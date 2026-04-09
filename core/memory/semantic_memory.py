from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

from core.embeddings import embed_texts
from core.retriever import RetrievedChunk
from core.store import StoredChunk, query_collection, upsert_chunks
from core.text import normalize_whitespace


SEMANTIC_COLLECTION_PREFIX = "mem_semantic__"


@dataclass(frozen=True)
class SemanticFact:
    user_id: str
    fact: str
    source: str | None
    created_at: float


def _cid(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def collection_name(user_id: str) -> str:
    return f"{SEMANTIC_COLLECTION_PREFIX}{user_id}"


def write_facts(*, user_id: str, facts: Iterable[str], source: str | None = None) -> int:
    facts_list = [normalize_whitespace(f) for f in facts if normalize_whitespace(f)]
    if not facts_list:
        return 0

    created_at = time.time()
    chunks: list[StoredChunk] = []
    texts: list[str] = []
    for f in facts_list:
        doc_id = _cid(user_id, "semantic", f)
        chunk_id = _cid(doc_id, str(created_at))
        meta: dict[str, Any] = {
            "user_id": user_id,
            "kind": "semantic",
            "fact": f,
            "source": source,
            "created_at": float(created_at),
        }
        chunks.append(StoredChunk(doc_id=doc_id, chunk_id=chunk_id, source="memory:semantic", text=f, metadata=meta))
        texts.append(f)

    embs = embed_texts(texts)
    return upsert_chunks(collection=collection_name(user_id), chunks=chunks, embeddings=embs)


def read_facts(*, user_id: str, question: str, k: int = 5) -> list[RetrievedChunk]:
    emb = embed_texts([question])[0]
    res = query_collection(collection=collection_name(user_id), query_embedding=emb, n_results=k)
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
        meta = meta or {}
        score = float(1.0 - dist) if dist is not None else 0.0
        out.append(
            RetrievedChunk(
                doc_id=str(meta.get("user_id", user_id)),
                chunk_id=str(cid),
                source="memory:semantic",
                score=score,
                text=str(text or ""),
                metadata=dict(meta),
            )
        )
    return out

