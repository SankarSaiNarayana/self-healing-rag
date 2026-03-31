from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from core.settings import get_settings


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any]


def _get_client() -> chromadb.PersistentClient:
    settings = get_settings()
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)


def get_collection(name: str) -> Collection:
    client = _get_client()
    # Use default embedding function because we precompute embeddings ourselves.
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    *,
    collection: str,
    chunks: list[StoredChunk],
    embeddings: list[list[float]],
) -> int:
    if not chunks:
        return 0
    col = get_collection(collection)
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "source": c.source,
            **(c.metadata or {}),
        }
        for c in chunks
    ]
    col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def query_collection(
    *,
    collection: str,
    query_embedding: list[float],
    n_results: int,
) -> dict[str, Any]:
    col = get_collection(collection)
    return col.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents", "metadatas", "distances"])


def get_all_chunks(*, collection: str) -> list[dict[str, Any]]:
    col = get_collection(collection)
    data = col.get(include=["documents", "metadatas"])
    ids: list[str] = data.get("ids") or []
    docs: list[str] = data.get("documents") or []
    metas: list[dict[str, Any]] = data.get("metadatas") or []
    out: list[dict[str, Any]] = []
    for i in range(min(len(ids), len(docs), len(metas))):
        out.append({"chunk_id": ids[i], "text": docs[i], "metadata": metas[i] or {}})
    return out

