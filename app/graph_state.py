# app/graph_state.py
from typing import List, TypedDict, Optional, Dict, Any

class GraphState(TypedDict):
    """
    State shared across graph nodes.
    """
    query: str
    selected_standard: str
    documents: List[Dict[str, Any]]            # raw retrieved docs
    evidence: List[Dict[str, Any]]             # curated authoritative evidence
    project_examples: List[Dict[str, Any]]     # curated project evidence (examples)
    answer: str                                 # final summarized answer
    error: Optional[str]
    rewrite_count: int
