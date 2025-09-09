# app/agent.py
import os
from typing import List, Dict, Any, Tuple, Set
from openai import OpenAI
from app.utils import retrieve_context
from app.graph_state import GraphState
from langgraph.graph import StateGraph, END

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = (os.getenv("MODEL_NAME", "gpt-4.1") or "").strip()

# ───────────────────────── Domain glossary (edit/extend freely) ─────────────────────────
# Deterministic aliases (both directions are useful). Keep keys lowercase.
GLOSSARY_ALIASES: Dict[str, List[str]] = {
    "additionality": ["additional", "additional benefit", "incremental benefit"],
    "leakage": ["emissions displacement", "activity-shifting leakage", "market leakage"],
    "buffer pool": ["buffer reserve"],
    "arr": ["afforestation reforestation revegetation", "reforestation", "afforestation"],
    "ifm": ["improved forest management", "forest management"],
    "afolu": ["agriculture forestry and other land use", "agriculture forestry and land use"],
    "organic soil": ["organic soils", "histosol", "histosols", "peat", "peat soil", "peatland", "peatlands", "organic-rich soil"],
    "peat": ["peatland", "peatlands", "histosol", "histosols", "organic soil"],
    "peatland": ["peat", "histosols", "organic soil"],
    "verra": ["vcs", "verified carbon standard"],
    "vcs": ["verra", "verified carbon standard"],
    "gs": ["gold standard"],
    "icr": ["independent carbon registry"],
    "sdg": ["sustainable development goals"],
    "mr": ["monitoring report", "monitoring and reporting"],
    "pdd": ["project design document"],
    # add more as you notice gaps in your corpus
}

def _alias_expand(user_query: str) -> List[str]:
    q = user_query.strip()
    if not q:
        return []
    terms = set([q])
    lower_q = q.lower()
    # add aliases for any glossary key that appears as word or acronym
    for k, alist in GLOSSARY_ALIASES.items():
        if k in lower_q.split() or k in lower_q or f" {k} " in f" {lower_q} ":
            for a in alist:
                terms.add(q.replace(k, a))
                terms.add(f"{q} ({a})")
        for a in alist:
            if a in lower_q:
                terms.add(q.replace(a, k))
                terms.add(f"{q} ({k})")
    # also add a simple acronym expansion if the query is short & uppercase like "IFM"
    if q.isupper() and len(q) <= 6 and q.lower() in GLOSSARY_ALIASES:
        for a in GLOSSARY_ALIASES[q.lower()]:
            terms.add(a)
    return list(terms)

# ───────────────────────── Helpers ───────────────────────── #

def _format_meta_line(source: str, clause: str, page: str) -> str:
    parts = [f"PDF: {source}"]
    if clause and str(clause).strip().upper() != "N/A":
        parts.append(f"Clause: {clause}")
    parts.append(f"Page: {page}")
    return " | ".join(parts)

def _build_citation_template() -> str:
    return (
        "When citing, use one of these exact formats:\n"
        " - (PDF: <filename>, Clause: <clause>, Page: <page>)  # if clause exists\n"
        " - (PDF: <filename>, Page: <page>)                    # if clause is N/A\n"
    )

def _build_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for e in evidence:
        meta_line = _format_meta_line(e["source"], e["clause"], e["page"])
        lines.append(f"[E{e['id']}] {meta_line}\nExcerpt: {e['quote']}")
    return "\n\n---\n\n".join(lines)

def _split_docs_by_type(documents: List[Dict[str, Any]]):
    std, proj, mixed = [], [], []
    for d in documents:
        t = (d.get("metadata", {}) or {}).get("doc_type", "mixed")
        if t == "standard":
            std.append(d)
        elif t == "project":
            proj.append(d)
        else:
            mixed.append(d)
    return std, proj, mixed

def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        m = d.get("metadata", {})
        key = (
            (d.get("page_content") or "").strip(),
            str(m.get("source")),
            str(m.get("clause")),
            str(m.get("page")),
        )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# ───────────────────────── LLM helpers (gpt-4.1 via Responses API) ───────────────────────── #

def _responses_call(prompt: str, temperature: float = 0.0) -> str:
    resp = client.responses.create(model=MODEL_NAME, input=prompt, temperature=temperature)
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    try:
        return resp.output[0].content[0].text.strip()  # type: ignore[attr-defined]
    except Exception:
        return str(resp).strip()

def _generate_query_variants(query: str) -> List[str]:
    """
    Multi-query expansion: generate synonyms, abbreviations, and alternates.
    """
    prompt = f"""You expand search queries for carbon-credit standards RAG.
Return 5 short alternate phrasings, including likely abbreviations/expansions used in standards.
Only return a JSON array of strings, no prose.

Query: {query}"""
    try:
        raw = _responses_call(prompt)
        # be liberal in parsing: try to eval JSON-ish lists safely
        import json, ast
        try:
            alts = json.loads(raw)
        except Exception:
            alts = ast.literal_eval(raw)
        alts = [str(s).strip() for s in alts if str(s).strip()]
        return alts[:5]
    except Exception:
        # fallback: no variants
        return []

def _hyde_query(query: str) -> str:
    """
    HyDE: create a short hypothetical answer to use as a retrieval query when hits are low.
    """
    prompt = f"""Write a concise, factual 4-6 sentence note that *could* be an answer to this user question
in the style of a carbon-credit standard, with key terminology and context words. No citations.

User question: {query}
"""
    try:
        return _responses_call(prompt, temperature=0.2)
    except Exception:
        return query

# ───────────────────────── Nodes ───────────────────────── #

def retrieve_node(state: GraphState):
    """
    Retrieve candidate chunks from Chroma and prepare:
      - authoritative standard evidence
      - project examples
    With query expansion (aliases + LLM multi-query) and HyDE fallback.
    """
    print("--- 🧠 NODE: RETRIEVE ---")
    user_query = state["query"]
    standard = state["selected_standard"]

    # 1) build a set of queries
    queries: List[str] = [user_query]
    queries += _alias_expand(user_query)
    queries += _generate_query_variants(user_query)

    # Add keyword-only bare terms if present in the user query (helps exact term recall)
    tokens = ["organic soil", "histosol", "histosols", "peat", "peatland", "peatlands"]
    uq_lower = user_query.lower()
    for t in tokens:
        if t in uq_lower:
            queries.append(t)

    # keep unique, keep short junk out
    seen_q: Set[str] = set()
    cleaned: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q or len(q) < 2:
            continue
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            cleaned.append(q)
    queries = cleaned[:8]  # cap to avoid over-querying

    # 2) retrieve for each query, merge and dedupe
    all_docs: List[Dict[str, Any]] = []
    for q in queries:
        docs = retrieve_context(
            query=q,
            selected_standard=standard,
            top_k=8,
            score_threshold=0.55,
            min_hits_before_fallback=0,
        )
        all_docs.extend(docs)

    all_docs = _dedupe_docs(all_docs)

    # 3) if we still have very few authoritative hits, run HyDE and a lower-threshold pass
    std_docs, proj_docs, mixed_docs = _split_docs_by_type(all_docs)
    if len(std_docs) < 3:
        hyde = _hyde_query(user_query)
        hyde_docs = retrieve_context(
            query=hyde,
            selected_standard=standard,
            top_k=8,
            score_threshold=0.45,  # slightly looser
            min_hits_before_fallback=0,
        )
        all_docs.extend(hyde_docs)
        all_docs = _dedupe_docs(all_docs)
        std_docs, proj_docs, mixed_docs = _split_docs_by_type(all_docs)

    if not all_docs:
        return {
            "documents": [],
            "evidence": [],
            "project_examples": [],
            "answer": "I could not find any relevant information in the provided documents."
        }

    # Prefer authoritative (standard) for main evidence; if none, use mixed; finally project.
    preferred_for_answer = (std_docs or mixed_docs or proj_docs)

    # Top 5 evidence for the main answer
    evidence = []
    for i, d in enumerate(preferred_for_answer[:5], start=1):
        m = d["metadata"]
        evidence.append({
            "id": i,
            "quote": d["page_content"],
            "source": m.get("source", "Unknown.pdf"),
            "clause": m.get("clause", "N/A"),
            "page": m.get("page", "N/A"),
            "score": d["score"],
            "doc_type": m.get("doc_type", "mixed"),
        })

    # Up to 3 project examples (only if we actually have project docs)
    project_examples = []
    for j, d in enumerate(proj_docs[:3], start=1):
        m = d["metadata"]
        project_examples.append({
            "id": j,
            "quote": d["page_content"],
            "source": m.get("source", "Unknown.pdf"),
            "clause": m.get("clause", "N/A"),
            "page": m.get("page", "N/A"),
            "score": d["score"],
            "doc_type": "project",
        })

    return {"documents": all_docs, "evidence": evidence, "project_examples": project_examples}


def generate_node(state: GraphState):
    """
    Create an exhaustive answer using ONLY the provided evidence. The main body
    comes from authoritative (standard/mixed) evidence; project examples are appended.
    """
    print("--- 💬 NODE: GENERATE ---")
    query = state["query"]
    evidence = state.get("evidence", [])
    project_examples = state.get("project_examples", [])

    if not evidence:
        return {}

    evidence_block = _build_evidence_block(evidence)
    examples_block = _build_evidence_block(project_examples) if project_examples else ""
    citation_rule = _build_citation_template()

    prompt = f"""
You are a carbon credit standards assistant. Answer STRICTLY and ONLY from the EVIDENCE provided.
Do NOT use outside knowledge. If the evidence does not contain the answer, say:
"I could not find any relevant information in the provided documents."

OUTPUT FORMAT (very important):
- Use GitHub-flavoured **Markdown**.
- Start with a **H2** heading for the main title (## Answer).
- Use short paragraphs, bullet lists, and **bold** lead-ins.
- Use tables when listing structured items (e.g., parameters, thresholds).
- Keep inline citations *immediately after the sentence they support*, using the exact formats shown below.
- Add a final section **## Example projects** (only if examples are provided), as short bullets with citations.
- Do not wrap the whole answer in code fences.

{citation_rule}

Question:
{query}

EVIDENCE (authoritative):
{evidence_block}

PROJECT EXAMPLES (optional):
{examples_block}
"""

    try:
        # Use Responses API for gpt-4.1
        answer = _responses_call(prompt, temperature=0.0)
        return {"answer": answer}
    except Exception as e:
        return {"answer": "", "error": f"Model call failed: {e}"}

# ───────────────────── Graph wiring ───────────────────── #

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()
