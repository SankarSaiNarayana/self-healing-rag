from __future__ import annotations

from typing import Any

from core.vectorstore.base import StoredChunk
from core.vectorstore.factory import get_vectorstore


def upsert_chunks(*, collection: str, chunks: list[StoredChunk], embeddings: list[list[float]]) -> int:
    return get_vectorstore().upsert_chunks(collection=collection, chunks=chunks, embeddings=embeddings)


def query_collection(*, collection: str, query_embedding: list[float], n_results: int) -> dict[str, Any]:
    return get_vectorstore().query(collection=collection, query_embedding=query_embedding, n_results=n_results)


def get_all_chunks(*, collection: str) -> list[dict[str, Any]]:
    return get_vectorstore().get_all_chunks(collection=collection)


def delete_collection(*, collection: str) -> None:
    get_vectorstore().delete_collection(collection=collection)


def delete_chunks_for_sources(*, collection: str, sources: list[str]) -> int:
    """Remove indexed chunks whose ingested `source` matches (e.g. before re-uploading the same file)."""
    if not sources:
        return 0
    want = set(sources)
    vs = get_vectorstore()
    rows = vs.get_all_chunks(collection=collection)
    ids: list[str] = []
    for r in rows:
        meta = r.get("metadata") or {}
        if str(meta.get("source", "")) in want:
            ids.append(str(r["chunk_id"]))
    if not ids:
        return 0
    return vs.delete_chunk_ids(collection=collection, chunk_ids=ids)

