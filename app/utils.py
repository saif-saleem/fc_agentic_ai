# app/utils.py
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import chromadb
from chromadb import PersistentClient

# ──────────────────────────────────────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────────────────────────────────────
# Your DB was built with ada-002 (1536-d). Keep queries on the same model.
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002") or "").strip().strip('"').strip("'")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────────────────────────────────────────────────────
# EXACT local-on-container paths (we will mount your Windows folders to these):
# icr:
#   app/embeddings_icr/standard_documents
#   app/embeddings_icr/project_documents
# plan_vivo:
#   app/embeddings_plan_vivo/standard_documents
#   app/embeddings_plan_vivo/project_documents
# vcs:
#   app/embeddings_vcs/standard_documents
#   app/embeddings_vcs/project_documents
# other:
#   app/embeddings_other_documents/carbon_market_general_documents
#   app/embeddings_other_documents/IPCC
# gs:
#   app/embeddings_gs/all_documents
# ──────────────────────────────────────────────────────────────────────────────
CHROMA_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "icr": [
        {"path": "app/embeddings_icr/standard_documents", "collection": None, "doc_type": "standard"},
        {"path": "app/embeddings_icr/project_documents",  "collection": None, "doc_type": "project"},
    ],
    "plan_vivo": [
        {"path": "app/embeddings_plan_vivo/standard_documents", "collection": None, "doc_type": "standard"},
        {"path": "app/embeddings_plan_vivo/project_documents",  "collection": None, "doc_type": "project"},
    ],
    "vcs": [
        {"path": "app/embeddings_vcs/standard_documents", "collection": None, "doc_type": "standard"},
        {"path": "app/embeddings_vcs/project_documents",  "collection": None, "doc_type": "project"},
    ],
    "other": [
        {"path": "app/embeddings_other_documents/carbon_market_general_documents", "collection": None, "doc_type": "standard"},
        {"path": "app/embeddings_other_documents/IPCC",                         "collection": None, "doc_type": "standard"},
    ],
    "gs": [
        {"path": "app/embeddings_gs/all_documents", "collection": None, "doc_type": "mixed"},
    ],
}

# caches
_CLIENTS: Dict[str, PersistentClient] = {}
_COLLECTIONS: Dict[str, Any] = {}

# Preferred collection names to try (in order), if spec["collection"] is None
_COLLECTION_CANDIDATES = [
    "langchain",
    "all_documents",
    "standard_documents",
    "project_documents",
    "carbon_market_general_documents",
    "IPCC",
]


def _get_openai() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = _get_openai()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _open_client(persist_path: str) -> Optional[PersistentClient]:
    # Must contain chroma.sqlite3
    sqlite_path = os.path.join(persist_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        print(f"[utils] Warning: no chroma.sqlite3 at {persist_path}")
        return None
    if persist_path in _CLIENTS:
        return _CLIENTS[persist_path]
    try:
        cli = chromadb.PersistentClient(path=persist_path)
        _CLIENTS[persist_path] = cli
        return cli
    except Exception as e:
        print(f"[utils] Error opening Chroma client at {persist_path}: {e}")
        return None


def _try_collections(cli: PersistentClient, wanted: Optional[str]):
    """Yield viable collection names to try, in order."""
    names = [c.name for c in cli.list_collections()]
    tried = set()

    # 1) the explicitly wanted name
    if wanted and wanted in names:
        yield wanted
        tried.add(wanted)

    # 2) our candidate list in order
    for cand in _COLLECTION_CANDIDATES:
        if cand in names and cand not in tried:
            yield cand
            tried.add(cand)

    # 3) finally, any remaining collections
    for name in names:
        if name not in tried:
            yield name


def _open_collection(persist_path: str, collection_name: Optional[str]):
    """
    Try the requested collection; if missing, try candidates (langchain, etc),
    then fall back to the first available collection in that DB.
    """
    key = f"{persist_path}:::{collection_name or '*'}"
    if key in _COLLECTIONS:
        return _COLLECTIONS[key]

    cli = _open_client(persist_path)
    if not cli:
        return None

    for cand in _try_collections(cli, collection_name):
        try:
            col = cli.get_collection(name=cand)
            _COLLECTIONS[key] = col
            if cand != (collection_name or ""):
                print(f"[utils] Using collection '{cand}' in {persist_path} (wanted '{collection_name}')")
            return col
        except Exception:
            continue

    print(f"[utils] No usable collection found in {persist_path}. Looked for '{collection_name}' or candidates.")
    return None


def _clean_excerpt(text: str, max_chars: int = 800) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    idx = max(cut.rfind("."), cut.rfind(" "))
    return (cut if idx < 100 else cut[:idx]).rstrip() + " …"


def _collect_hits(query: str, specs: List[Dict[str, Any]], k: int, thr: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    qvecs = _embed_texts([query])
    if not qvecs:
        return out
    qvec = qvecs[0]

    for spec in specs:
        col = _open_collection(spec["path"], spec.get("collection"))
        if not col:
            continue
        try:
            res = col.query(
                query_embeddings=[qvec],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            docs_list = res.get("documents", [[]])[0]
            metas_list = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs_list)

            def _sim(d):
                try:
                    return 1.0 / (1.0 + float(d))
                except Exception:
                    return 0.0

            for doc_text, meta, dist in zip(docs_list, metas_list, dists):
                sim = _sim(dist)
                if sim < thr:
                    continue
                out.append({
                    "page_content": _clean_excerpt(doc_text or ""),
                    "metadata": {
                        "source": (meta or {}).get("source", "Unknown.pdf"),
                        "clause": (meta or {}).get("clause", "N/A"),
                        "page": (meta or {}).get("page", "N/A"),
                        "doc_type": spec.get("doc_type", "mixed"),
                    },
                    "score": float(sim),
                })
        except Exception as e:
            print(f"[utils] Retrieval failed for {spec['path']}::{spec.get('collection')}: {e}")

    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def retrieve_context(
    query: str,
    selected_standard: Optional[str],
    top_k: int = 10,
    score_threshold: float = 0.55,
    min_hits_before_fallback: int = 0,
) -> List[Dict[str, Any]]:
    if not query:
        return []

    def _run_once(q: str, thr: float, k: int) -> List[Dict[str, Any]]:
        if selected_standard and selected_standard in CHROMA_SPECS:
            specs = CHROMA_SPECS[selected_standard]
        else:
            specs = [s for lst in CHROMA_SPECS.values() for s in lst]

        hits = _collect_hits(q, specs, k=k, thr=thr)

        def _order_key(d: Dict[str, Any]):
            t = d["metadata"].get("doc_type", "mixed")
            rank = {"standard": 0, "project": 1}.get(t, 2)
            return (rank, -d["score"])
        hits.sort(key=_order_key)
        return hits[:k]

    docs = _run_once(query, score_threshold, top_k)

    std_hits_count = sum(1 for d in docs if d["metadata"].get("doc_type") == "standard")
    if std_hits_count < max(3, min_hits_before_fallback or 0):
        looser_thr = max(0.40, score_threshold - 0.15)
        larger_k = max(20, top_k + 10)
        docs2 = _run_once(query, looser_thr, larger_k)

        merged: List[Dict[str, Any]] = []
        seen = set()
        for d in (docs + docs2):
            key = (d["page_content"], d["metadata"]["source"], d["metadata"]["clause"], d["metadata"]["page"])
            if key not in seen:
                seen.add(key)
                merged.append(d)
        docs = merged

    return docs


def warmup_all() -> None:
    try:
        _ = _embed_texts(["warmup"])
    except Exception as e:
        print(f"[warmup] Embedding init failed: {e}")

    for standard, specs in CHROMA_SPECS.items():
        for spec in specs:
            try:
                col = _open_collection(spec["path"], spec.get("collection"))
                if col:
                    qvec = _embed_texts(["warmup"])[0]
                    _ = col.query(query_embeddings=[qvec], n_results=1, include=["documents"])
            except Exception as e:
                print(f"[warmup] Chroma warmup failed for {spec['path']}::{spec.get('collection')}: {e}")
    print("[warmup] Completed embeddings + Chroma pre-initialization.")
