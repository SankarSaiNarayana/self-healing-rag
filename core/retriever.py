from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from core.embeddings import embed_texts
from core.query_router import filter_chunks_by_source_hint, infer_source_hint
from core.settings import get_settings
from core.store import get_all_chunks, query_collection
from core.text import normalize_whitespace


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    source: str
    score: float
    text: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    max_score: float
    diagnostics: dict[str, Any]


@lru_cache(maxsize=16)
def _bm25_index(collection: str) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    rows = get_all_chunks(collection=collection)
    corpus = [_tokenize(r["text"]) for r in rows]
    bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([[]])
    return bm25, rows


def _rescale_0_1(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mn = min(xs)
    mx = max(xs)
    if math.isclose(mn, mx):
        # If there's only one candidate (or all equal), treat them as max-confidence within that pool.
        return [1.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def _vector_retrieve(collection: str, question: str, k: int) -> list[RetrievedChunk]:
    q_emb = embed_texts([question])[0]
    res = query_collection(collection=collection, query_embedding=q_emb, n_results=k)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
        meta = meta or {}
        # Chroma cosine distance: smaller is better. Convert to similarity-like.
        score = float(1.0 - dist) if dist is not None else 0.0
        chunks.append(
            RetrievedChunk(
                doc_id=str(meta.get("doc_id", "")),
                chunk_id=str(cid),
                source=str(meta.get("source", "")),
                score=score,
                text=str(text or ""),
                metadata=dict(meta),
            )
        )
    return chunks


def _bm25_retrieve(collection: str, question: str, k: int) -> list[RetrievedChunk]:
    bm25, rows = _bm25_index(collection)
    toks = _tokenize(question)
    if not rows:
        return []
    scores = bm25.get_scores(toks)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    picked = []
    for i in idxs:
        r = rows[i]
        meta = r.get("metadata") or {}
        picked.append(
            RetrievedChunk(
                doc_id=str(meta.get("doc_id", "")),
                chunk_id=str(r.get("chunk_id", "")),
                source=str(meta.get("source", "")),
                score=float(scores[i]),
                text=str(r.get("text", "")),
                metadata=dict(meta),
            )
        )
    return picked


def _rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []
    q = normalize_whitespace(question)
    rescored = []
    for c in chunks:
        # Token-set fuzz is a decent cheap reranker for demos.
        rr = fuzz.token_set_ratio(q, c.text) / 100.0
        rescored.append((0.70 * c.score + 0.30 * rr, rr, c))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return [
        RetrievedChunk(
            doc_id=c.doc_id,
            chunk_id=c.chunk_id,
            source=c.source,
            score=float(s),
            text=c.text,
            metadata=c.metadata,
        )
        for (s, _rr, c) in rescored
    ]


def _hybrid_merge(
    *,
    question: str,
    vector_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    alpha: float,
) -> list[RetrievedChunk]:
    # Normalize within each pool and merge by chunk_id.
    v_scores = _rescale_0_1([c.score for c in vector_chunks])
    b_scores = _rescale_0_1([c.score for c in bm25_chunks])

    merged: dict[str, dict[str, Any]] = {}
    for c, s in zip(vector_chunks, v_scores, strict=False):
        merged[c.chunk_id] = {"chunk": c, "v": s, "b": 0.0}
    for c, s in zip(bm25_chunks, b_scores, strict=False):
        if c.chunk_id in merged:
            merged[c.chunk_id]["b"] = max(float(merged[c.chunk_id]["b"]), float(s))
        else:
            merged[c.chunk_id] = {"chunk": c, "v": 0.0, "b": s}

    out: list[RetrievedChunk] = []
    for cid, row in merged.items():
        c = row["chunk"]
        score = alpha * float(row["v"]) + (1.0 - alpha) * float(row["b"])
        out.append(
            RetrievedChunk(
                doc_id=c.doc_id,
                chunk_id=cid,
                source=c.source,
                score=float(score),
                text=c.text,
                metadata=c.metadata,
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def expand_query(question: str, query_type: str, attempt: int) -> str:
    q = normalize_whitespace(question)
    if attempt <= 0:
        return q
    # Simple expansion heuristics; the goal is to change retrieval, not be "smart".
    if query_type == "multi-hop":
        return f"{q} key steps details sources"
    if query_type == "conceptual":
        return f"{q} explanation overview"
    return f"{q} exact details"


def retrieve(
    *,
    question: str,
    collection: str,
    query_type: str,
    top_k: int | None = None,
    bm25_k: int | None = None,
    rerank: bool = True,
    hybrid_alpha: float | None = None,
    source_hint: str | None = None,
    auto_source: bool = False,
) -> RetrievalResult:
    settings = get_settings()
    top_k = top_k or settings.top_k
    bm25_k = bm25_k or settings.bm25_k

    alpha = float(hybrid_alpha if hybrid_alpha is not None else settings.hybrid_alpha)
    q = normalize_whitespace(question)

    v = _vector_retrieve(collection, q, top_k)
    b = _bm25_retrieve(collection, q, bm25_k)
    merged = _hybrid_merge(question=q, vector_chunks=v, bm25_chunks=b, alpha=alpha)
    if rerank:
        merged = _rerank(q, merged)

    router: dict[str, Any] = {}
    hint = (source_hint or "").strip() or None
    if auto_source and not hint:
        inferred, conf = infer_source_hint(question=q, collection=collection)
        if inferred:
            hint = inferred
            router["mode"] = "auto"
            router["inferred_source"] = inferred
            router["inferred_confidence"] = round(conf, 4)
    elif hint:
        router["mode"] = "manual"
        router["source_hint"] = hint

    if hint:
        merged, fb = filter_chunks_by_source_hint(merged, hint)
        if fb:
            router["filter_fallback"] = True
            router["note"] = "source_hint matched no chunks; using unfiltered retrieval"

    max_score = merged[0].score if merged else 0.0
    slice_n = max(top_k, bm25_k, 1)
    return RetrievalResult(
        query=q,
        chunks=merged[:slice_n],
        max_score=float(max_score),
        diagnostics={
            "alpha": alpha,
            "top_k": top_k,
            "bm25_k": bm25_k,
            "rerank": rerank,
            "vector_candidates": len(v),
            "bm25_candidates": len(b),
            "router": router,
        },
    )


def retrieve_with_self_heal(
    *,
    question: str,
    collection: str,
    query_type: str,
    top_k: int | None = None,
    bm25_k: int | None = None,
    rerank: bool = True,
    hybrid_alpha: float | None = None,
    source_hint: str | None = None,
    auto_source: bool = False,
) -> tuple[RetrievalResult, int]:
    settings = get_settings()
    retries = 0
    last: RetrievalResult | None = None

    for attempt in range(settings.max_retrieval_retries + 1):
        q2 = expand_query(question, query_type, attempt)
        # As we retry, broaden a bit.
        tk = (top_k or settings.top_k) + 4 * attempt
        bk = (bm25_k or settings.bm25_k) + 6 * attempt
        res = retrieve(
            question=q2,
            collection=collection,
            query_type=query_type,
            top_k=tk,
            bm25_k=bk,
            rerank=rerank,
            hybrid_alpha=hybrid_alpha,
            source_hint=source_hint,
            auto_source=auto_source,
        )
        last = res
        if res.max_score >= float(settings.min_retrieval_score):
            return res, retries
        retries += 1

    return last or RetrievalResult(query=question, chunks=[], max_score=0.0, diagnostics={}), retries


def invalidate_bm25_cache() -> None:
    """BM25 index is memoized; call after writes so hybrid retrieval sees new chunks."""
    _bm25_index.cache_clear()

