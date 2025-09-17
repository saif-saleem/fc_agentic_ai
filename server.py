# server.py
import os
from typing import Optional, Any, Dict
from pathlib import Path
import zipfile
import requests
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langserve import add_routes

# ----- Try importing your app pieces (these require embeddings later) -----
from app.agent import agent_app, _llm_call
from app.utils import warmup_all


APP_DIR = Path(__file__).parent / "app"      # /app/app in container
EMB_TARGETS = [
    APP_DIR / "embeddings_gs" / "chroma.sqlite3",
    APP_DIR / "embeddings_icr" / "chroma.sqlite3",
    APP_DIR / "embeddings_vcs" / "chroma.sqlite3",
    APP_DIR / "embeddings_plan_vivo" / "chroma.sqlite3",
    APP_DIR / "embeddings_other_documents" / "chroma.sqlite3",
]


def _ensure_embeddings():
    """
    If any Chroma DB is missing and EMBEDDINGS_ZIP_URL is set,
    download and unzip to app/ at startup. This replaces entrypoint.sh
    for the Python-runtime (non-Dockerfile) deploy.
    """
    need = any(not p.exists() for p in EMB_TARGETS)
    zip_url = os.getenv("EMBEDDINGS_ZIP_URL", "").strip()
    if not need:
        print("[startup] Embeddings found locally; skipping download.")
        return
    if not zip_url:
        print("[startup] ERROR: EMBEDDINGS_ZIP_URL not set AND embeddings not found.")
        # Don't crash the app; leave logs and let endpoints fail gracefully if needed
        return

    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = Path("/tmp/embeddings_bundle.zip")
    print(f"[startup] Downloading embeddings from {zip_url} ...")
    try:
        with requests.get(zip_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("[startup] Unzipping embeddings ...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(APP_DIR)
        print("[startup] Embeddings ready.")
    finally:
        tmp_zip.unlink(missing_ok=True)


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


# ---------- Lifespan Events ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[startup] Starting Carbon GPT Agent Server...")
    
    # Ensure embeddings are present (fallback if build-time step failed)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _ensure_embeddings)
    except Exception as e:
        print(f"[startup] _ensure_embeddings failed: {e}")

    # warmups (optional; best effort)
    try:
        warmup_all()
    except Exception as e:
        print(f"[startup] warmup_all failed: {e}")
    try:
        _llm_call("Warmup ping")
        print("[startup] Model warmup ping OK.")
    except Exception as e:
        print(f"[startup] Model warmup ping failed: {e}")
    
    print("[startup] Carbon GPT Agent Server started successfully!")
    yield
    # Shutdown
    print("[shutdown] Carbon GPT Agent Server shutting down...")


# ---------- App ----------
app = FastAPI(
    title="Carbon GPT Agent Server",
    version="1.4",
    description="Strict RAG agent for carbon credit standards with weighted retrieval and project examples.",
    lifespan=lifespan,
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


# ---------- Basic endpoints ----------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Carbon GPT Agent",
        "version": "1.4",
        "routes": ["/health", "/ask", "/carbon-agent/invoke", "/carbon-agent/playground/", "/ui"],
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


# Add langserve routes with proper adapters
try:
    # Try with modern parameter names
    add_routes(
        app,
        agent_app,
        path="/carbon-agent",
        input_type=AgentInput,
        output_type=AgentOutput,
        config_keys=["configurable"],
        input_adapter=_in_adapter,
        output_adapter=_out_adapter,
    )
except TypeError as e:
    print(f"First add_routes attempt failed: {e}")
    try:
        # Try with legacy parameter names
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
    except TypeError as e2:
        print(f"Second add_routes attempt failed: {e2}")
        # Fallback without adapters
        add_routes(
            app,
            agent_app,
            path="/carbon-agent",
            input_type=AgentInput,
            output_type=AgentOutput,
        )


# ---------- Simple REST endpoint (bypass playground) ----------
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


# ---------- Serve the UI (index.html) ----------
from fastapi.responses import HTMLResponse
@app.get("/ui", response_class=HTMLResponse)
def ui():
    index_path = APP_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------- Local run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
