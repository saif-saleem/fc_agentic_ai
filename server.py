# server.py
import os
import pathlib
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langserve import add_routes

from app.agent import agent_app, _responses_call
from app.utils import warmup_all

# ---------- Models ----------
class AgentInput(BaseModel):
    query: str
    selected_standard: str  # "gs" | "vcs" | "icr" | "plan_vivo" | "other"

class AgentOutput(BaseModel):
    answer: str
    evidence: list[dict] = []
    project_examples: list[dict] = []

class AskBody(BaseModel):
    query: str
    standard: str

# ---------- App ----------
app = FastAPI(
    title="Carbon GPT Agent Server",
    version="1.4",
    description="Strict RAG agent for carbon credit standards with weighted retrieval and project examples.",
)

# ---------- CORS ----------
allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = (
    ["*"] if allow_origins_env.strip() == "*"
    else [o.strip() for o in allow_origins_env.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------- Static / UI ----------
ROOT_DIR = pathlib.Path(__file__).parent
APP_DIR = ROOT_DIR / "app"
INDEX_HTML = APP_DIR / "index.html"

# (optional) mount whole app folder at /static if you later add images/css
app.mount("/static", StaticFiles(directory=str(APP_DIR)), name="static")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    """
    Serve the chat UI from the same domain.
    In index.html we auto-use window.origin as API base, so no hardcoding.
    """
    return FileResponse(str(INDEX_HTML))

# ---------- Startup warmup ----------
@app.on_event("startup")
async def _startup_warmup():
    try:
        warmup_all()
    except Exception as e:
        print(f"[startup] warmup_all failed: {e}")

    try:
        _responses_call("Warmup ping")
        print("[startup] Model warmup ping OK.")
    except Exception as e:
        print(f"[startup] Model warmup ping failed: {e}")

# ---------- Basic endpoints ----------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Carbon GPT Agent",
        "version": "1.4",
        "routes": ["/health", "/ui", "/ask", "/carbon-agent/invoke", "/carbon-agent/playground/"],
    }

@app.get("/health")
def health():
    return {"ok": True}

# ---------- LangServe route ----------
def _in_adapter(inp: Any) -> Dict[str, Any]:
    if isinstance(inp, dict):
        q = inp.get("query")
        std = inp.get("selected_standard") or inp.get("standard")
    else:
        q = getattr(inp, "query", None)
        std = getattr(inp, "selected_standard", None)

    if not isinstance(q, str) or not isinstance(std, str):
        raise ValueError("Both 'query' and 'selected_standard' must be strings.")
    return {"query": q, "selected_standard": std}

def _out_adapter(out: Any) -> AgentOutput:
    if isinstance(out, dict):
        return AgentOutput(
            answer=str(out.get("answer", "")),
            evidence=out.get("evidence", []) or [],
            project_examples=out.get("project_examples", []) or [],
        )
    return AgentOutput(answer=str(out), evidence=[], project_examples=[])

try:
    add_routes(
        app,
        agent_app,
        path="/carbon-agent",
        input_type=AgentInput,
        output_type=AgentOutput,
        config_keys=["configurable"],
        in_adapter=_in_adapter,
        out_adapter=_out_adapter,
    )
except TypeError:
    add_routes(
        app,
        agent_app,
        path="/carbon-agent",
        input_type=AgentInput,
        config_keys=["configurable"],
        in_adapter=_in_adapter,
    )

# ---------- Simple REST endpoint (optional guarded) ----------
@app.post("/ask")
def ask(body: AskBody, x_api_key: Optional[str] = Header(default=None)):
    required = os.getenv("PUBLIC_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        result = agent_app.invoke({"query": body.query, "selected_standard": body.standard})
        return _out_adapter(result).model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Local run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
