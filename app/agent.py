# app/agent.py
import os
import json
import ast
from typing import List, Dict, Any, Tuple, Set
from openai import OpenAI, APIError
from app.utils import retrieve_context
from app.graph_state import GraphState
from langgraph.graph import StateGraph, END

# --- Client Initialization ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = (os.getenv("MODEL_NAME", "gpt-4.1") or "").strip()

# --- Domain Glossary ---
# Deterministic aliases for query expansion.
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
}

# --- Helper Functions ---

def _alias_expand(user_query: str) -> List[str]:
    """Expands a query with predefined aliases from the glossary."""
    q = user_query.strip()
    if not q:
        return []
    terms = {q}
    lower_q = q.lower()
    
    # Add aliases for any glossary key that appears as a word or acronym
    for k, alist in GLOSSARY_ALIASES.items():
        if k in lower_q.split() or f" {k} " in f" {lower_q} ":
            for a in alist:
                terms.add(q.replace(k, a))
                terms.add(f"{q} ({a})")
    
    # Add reverse aliases (e.g., "reforestation" -> "arr")
    for k, alist in GLOSSARY_ALIASES.items():
        for a in alist:
            if a in lower_q:
                terms.add(q.replace(a, k))
                terms.add(f"{q} ({k})")

    # Add simple acronym expansion if the query is a short uppercase term
    if q.isupper() and len(q) <= 6 and q.lower() in GLOSSARY_ALIASES:
        terms.update(GLOSSARY_ALIASES[q.lower()])
        
    return list(terms)

def _format_meta_line(source: str, clause: str, page: str) -> str:
    """Formats a metadata line for display."""
    parts = [f"PDF: {source}"]
    if clause and str(clause).strip().upper() != "N/A":
        parts.append(f"Clause: {clause}")
    parts.append(f"Page: {page}")
    return " | ".join(parts)

def _build_citation_template() -> str:
    """Creates the citation instruction block for the LLM prompt."""
    return (
        "When citing, use one of these exact formats:\n"
        " - (PDF: <filename>, Clause: <clause>, Page: <page>)  # if clause exists\n"
        " - (PDF: <filename>, Page: <page>)                    # if clause is N/A\n"
    )

def _build_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    """Constructs a formatted string of evidence blocks."""
    lines = []
    for e in evidence:
        meta_line = _format_meta_line(e.get("source", "Unknown"), e.get("clause", "N/A"), e.get("page", "N/A"))
        lines.append(f"[E{e['id']}] {meta_line}\nExcerpt: {e['quote']}")
    return "\n\n---\n\n".join(lines)

def _split_docs_by_type(documents: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    """Categorizes documents into standard, project, and mixed types."""
    std, proj, mixed = [], [], []
    for d in documents:
        doc_type = (d.get("metadata", {}) or {}).get("doc_type", "mixed")
        if doc_type == "standard":
            std.append(d)
        elif doc_type == "project":
            proj.append(d)
        else:
            mixed.append(d)
    return std, proj, mixed

def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate documents based on content and metadata."""
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

# --- LLM Helper Functions ---

def _llm_call(prompt: str, temperature: float = 0.0) -> str:
    """Makes a call to the OpenAI API and handles potential responses and errors."""
    try:
        # Assuming `client.responses.create` is a custom or preview API endpoint.
        # The standard method is `client.chat.completions.create`.
        # This code preserves the original `client.responses.create` structure.
        response = client.responses.create(model=MODEL_NAME, input=prompt, temperature=temperature)
        
        # Handle different possible response structures
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        if hasattr(response, "output") and response.output and hasattr(response.output[0], "content"):
            return response.output[0].content[0].text.strip()
        
        # Fallback for unexpected response structure
        return str(response).strip()
    except APIError as e:
        print(f"Error calling OpenAI API: {e}")
        return f"An error occurred while communicating with the model: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during the LLM call: {e}")
        return "An unexpected error occurred."


def _generate_query_variants(query: str) -> List[str]:
    """Uses the LLM to generate alternative phrasings for a search query."""
    prompt = f"""You expand search queries for carbon-credit standards RAG.
Return 5 short alternate phrasings, including likely abbreviations/expansions used in standards.
Only return a JSON array of strings, no prose.

Query: {query}"""
    try:
        raw_response = _llm_call(prompt)
        # Liberal parsing to handle JSON-like strings
        try:
            alts = json.loads(raw_response)
        except json.JSONDecodeError:
            alts = ast.literal_eval(raw_response)
        
        if isinstance(alts, list):
            return [str(s).strip() for s in alts if str(s).strip()][:5]
        return []
    except (ValueError, SyntaxError, Exception) as e:
        print(f"Failed to parse query variants from LLM response: {e}")
        return []

def _hyde_query(query: str) -> str:
    """Generates a hypothetical document (HyDE) to improve retrieval."""
    prompt = f"""Write a concise, factual 4-6 sentence note that *could* be an answer to this user question
in the style of a carbon-credit standard, with key terminology and context words. No citations.

User question: {query}"""
    return _llm_call(prompt, temperature=0.2)


# --- Graph Nodes ---

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieves documents from ChromaDB using query expansion and HyDE."""
    print("--- 🧠 NODE: RETRIEVE ---")
    user_query = state["query"]
    standard = state["selected_standard"]

    # 1. Build a set of queries for expansion
    queries = {user_query}
    queries.update(_alias_expand(user_query))
    queries.update(_generate_query_variants(user_query))

    tokens = ["organic soil", "histosol", "histosols", "peat", "peatland", "peatlands"]
    for t in tokens:
        if t in user_query.lower():
            queries.add(t)

    cleaned_queries = sorted([q for q in queries if q and len(q) > 1])[:8]

    # 2. Retrieve for each query, then merge and deduplicate
    all_docs: List[Dict[str, Any]] = []
    for q in cleaned_queries:
        docs = retrieve_context(
            query=q,
            selected_standard=standard,
            top_k=8,
            score_threshold=0.55,
        )
        all_docs.extend(docs)
    
    all_docs = _dedupe_docs(all_docs)

    # 3. If few authoritative hits, try HyDE as a fallback
    std_docs, proj_docs, mixed_docs = _split_docs_by_type(all_docs)
    if len(std_docs) < 3:
        hyde = _hyde_query(user_query)
        hyde_docs = retrieve_context(
            query=hyde,
            selected_standard=standard,
            top_k=8,
            score_threshold=0.45,  # Looser threshold for HyDE
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

    # 4. Prepare evidence and project examples for the next step
    preferred_for_answer = std_docs or mixed_docs or proj_docs

    evidence = []
    for i, d in enumerate(preferred_for_answer[:5], start=1):
        m = d.get("metadata", {})
        evidence.append({
            "id": i,
            "quote": d.get("page_content", ""),
            "source": m.get("source", "Unknown.pdf"),
            "clause": m.get("clause", "N/A"),
            "page": m.get("page", "N/A"),
            "score": d.get("score", 0.0),
            "doc_type": m.get("doc_type", "mixed"),
        })

    project_examples = []
    for j, d in enumerate(proj_docs[:3], start=1):
        m = d.get("metadata", {})
        project_examples.append({
            "id": j,
            "quote": d.get("page_content", ""),
            "source": m.get("source", "Unknown.pdf"),
            "clause": m.get("clause", "N/A"),
            "page": m.get("page", "N/A"),
            "score": d.get("score", 0.0),
            "doc_type": "project",
        })

    return {"documents": all_docs, "evidence": evidence, "project_examples": project_examples}


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generates a final answer based ONLY on the provided evidence."""
    print("--- 💬 NODE: GENERATE ---")
    query = state["query"]
    evidence = state.get("evidence", [])
    project_examples = state.get("project_examples", [])

    if not evidence:
        return {"answer": "I could not find any relevant information in the provided documents."}

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
    answer = _llm_call(prompt, temperature=0.0)
    return {"answer": answer}

# --- Graph Wiring ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent_app = workflow.compile()