from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from rapidfuzz import fuzz

from core.store import get_all_chunks
from core.text import normalize_whitespace

C = TypeVar("C")


@lru_cache(maxsize=32)
def list_unique_sources(collection: str) -> tuple[str, ...]:
    rows = get_all_chunks(collection=collection)
    seen: set[str] = set()
    for r in rows:
        meta = r.get("metadata") or {}
        src = str(meta.get("source", "") or "").strip()
        if src:
            seen.add(src)
    return tuple(sorted(seen))


def infer_source_hint(*, question: str, collection: str, min_score: float = 0.42) -> tuple[str | None, float]:
    """
    Pick the most likely ingest `source` path for this question using fuzzy match against
    filenames (good for questions like 'What does Kubernetes.pdf say about pods?').
    """
    sources = list_unique_sources(collection)
    if not sources:
        return None, 0.0
    q = normalize_whitespace(question).lower()
    best: tuple[str | None, float] = (None, 0.0)
    for src in sources:
        base = Path(src).name.lower()
        s = max(
            fuzz.partial_ratio(q, src.lower()) / 100.0,
            fuzz.partial_ratio(q, base) / 100.0,
            fuzz.token_set_ratio(question, src) / 100.0,
            fuzz.token_set_ratio(question, base) / 100.0,
        )
        if s > best[1]:
            best = (src, float(s))
    if best[0] is not None and best[1] >= min_score:
        return best[0], best[1]
    return None, best[1]


def filter_chunks_by_source_hint(chunks: list[C], hint: str) -> tuple[list[C], bool]:
    """
    Keep chunks whose `source` contains `hint` (case-insensitive substring, e.g. 'Kubernetes.pdf').
    If nothing matches, return the original list and fallback=True.
    """
    if not hint.strip():
        return chunks, False
    h = hint.strip().lower()
    filt = [c for c in chunks if h in (getattr(c, "source", None) or "").lower()]
    if not filt:
        return chunks, True
    return filt, False


def invalidate_source_list_cache() -> None:
    """Clear cached source listings after index writes."""
    list_unique_sources.cache_clear()
