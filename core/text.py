from __future__ import annotations

import re
from typing import Iterable


_WS_RE = re.compile(r"\s+")


def normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    # Simple, language-agnostic heuristic.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def safe_join_lines(lines: Iterable[str]) -> str:
    return normalize_whitespace("\n".join(lines))

