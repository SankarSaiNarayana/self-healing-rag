from __future__ import annotations

import re


def classify_query(question: str) -> str:
    """
    Lightweight heuristic classifier:
    - factual: asks for specific fact/date/name/definition
    - conceptual: asks for explanation/why/how something works
    - multi-hop: asks to compare/synthesize across multiple things or steps
    """
    q = question.strip().lower()
    if not q:
        return "factual"

    # Multi-hop signals.
    if any(w in q for w in ["compare", "difference", "vs", "versus", "pros and cons", "tradeoff", "trade-off"]):
        return "multi-hop"
    if any(w in q for w in ["step", "steps", "process", "workflow", "pipeline"]) and "?" in q:
        return "multi-hop"
    if re.search(r"\b(and|then)\b.*\b(and|then)\b", q):
        return "multi-hop"

    # Conceptual signals.
    if q.startswith(("why ", "how ", "explain ", "what are the reasons", "what is the intuition")):
        return "conceptual"
    if any(w in q for w in ["intuition", "overview", "explain", "high level", "high-level", "concept"]):
        return "conceptual"

    return "factual"

