from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.embeddings import embed_texts
from core.retriever import RetrievedChunk
from core.store import StoredChunk, get_all_chunks, query_collection, upsert_chunks
from core.text import normalize_whitespace


EPISODIC_COLLECTION_PREFIX = "mem_episodic__"


@dataclass(frozen=True)
class EpisodicItem:
    user_id: str
    question: str
    answer: str
    confidence: float
    faithfulness: float
    created_at: float


def _cid(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def collection_name(user_id: str) -> str:
    return f"{EPISODIC_COLLECTION_PREFIX}{user_id}"


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "unknown time"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "unknown time"


def write_episodic(*, user_id: str, question: str, answer: str, confidence: float, faithfulness: float) -> str:
    created_at = time.time()
    doc_id = _cid(user_id, "episodic", question)
    chunk_id = _cid(doc_id, str(created_at))
    text = f"Q: {question}\nA: {answer}"
    meta: dict[str, Any] = {
        "user_id": user_id,
        "kind": "episodic",
        "question": question,
        "answer": answer,
        "confidence": float(confidence),
        "faithfulness": float(faithfulness),
        "created_at": float(created_at),
    }
    emb = embed_texts([question])
    upsert_chunks(
        collection=collection_name(user_id),
        chunks=[StoredChunk(doc_id=doc_id, chunk_id=chunk_id, source="memory:episodic", text=text, metadata=meta)],
        embeddings=emb,
    )
    return chunk_id


def _parse_q_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip().upper().startswith("Q:"):
            return normalize_whitespace(line.replace("Q:", "", 1).strip())
    return ""


def read_episodic(*, user_id: str, question: str, k: int = 3) -> list[RetrievedChunk]:
    emb = embed_texts([question])[0]
    res = query_collection(collection=collection_name(user_id), query_embedding=emb, n_results=max(k * 4, k))
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    raw: list[RetrievedChunk] = []
    for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
        meta = meta or {}
        score = float(1.0 - dist) if dist is not None else 0.0
        raw.append(
            RetrievedChunk(
                doc_id=str(meta.get("user_id", user_id)),
                chunk_id=str(cid),
                source="memory:episodic",
                score=score,
                text=str(text or ""),
                metadata=dict(meta),
            )
        )
    # Dedupe by question text, keep best score / newest created_at
    best: dict[str, RetrievedChunk] = {}
    for ch in raw:
        qkey = _parse_q_line(ch.text) or ch.chunk_id
        prev = best.get(qkey)
        if prev is None:
            best[qkey] = ch
            continue
        ca = (ch.metadata or {}).get("created_at")
        pa = (prev.metadata or {}).get("created_at")
        if ch.score > prev.score + 1e-6:
            best[qkey] = ch
        elif ca is not None and pa is not None and float(ca) > float(pa):
            best[qkey] = ch
    deduped = sorted(best.values(), key=lambda x: float((x.metadata or {}).get("created_at", 0.0)), reverse=True)
    return deduped[:k]


def list_episodic_recent(*, user_id: str, limit: int = 40) -> list[dict[str, Any]]:
    rows = get_all_chunks(collection=collection_name(user_id))
    parsed: list[dict[str, Any]] = []
    for r in rows:
        meta = r.get("metadata") or {}
        text = str(r.get("text", ""))
        q = str(meta.get("question", _parse_q_line(text)))
        parsed.append(
            {
                "chunk_id": r.get("chunk_id"),
                "question": q,
                "answer_preview": str(meta.get("answer", ""))[:240],
                "created_at": float(meta.get("created_at", 0.0)),
                "confidence": float(meta.get("confidence", 0.0)),
                "faithfulness": float(meta.get("faithfulness", 0.0)),
            }
        )
    parsed.sort(key=lambda x: x["created_at"], reverse=True)
    return parsed[:limit]


def format_past_questions_answer(episodic: list[RetrievedChunk], *, max_items: int = 8) -> str:
    lines: list[str] = []
    for i, it in enumerate(episodic[:max_items], start=1):
        meta = it.metadata or {}
        q = str(meta.get("question", _parse_q_line(it.text)))
        ts = _format_ts(meta.get("created_at"))
        lines.append(f'{i}. ({ts}) {q}')
    return "Here are your recent past questions (deduplicated, newest first):\n" + "\n".join(lines)
