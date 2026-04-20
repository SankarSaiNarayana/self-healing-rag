from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from core.settings import get_settings
from core.vectorstore.base import StoredChunk, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self) -> None:
        self._client: chromadb.PersistentClient | None = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            settings = get_settings()
            os.makedirs(settings.chroma_dir, exist_ok=True)
            # Avoid PostHog telemetry calls that break on some dependency versions (noisy ERROR logs).
            self._client = chromadb.PersistentClient(
                path=settings.chroma_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _collection(self, name: str) -> Collection:
        client = self._get_client()
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def upsert_chunks(self, *, collection: str, chunks: list[StoredChunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        col = self._collection(collection)
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

    def query(self, *, collection: str, query_embedding: list[float], n_results: int) -> dict[str, Any]:
        col = self._collection(collection)
        return col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def get_all_chunks(self, *, collection: str) -> list[dict[str, Any]]:
        col = self._collection(collection)
        data = col.get(include=["documents", "metadatas"])
        ids: list[str] = data.get("ids") or []
        docs: list[str] = data.get("documents") or []
        metas: list[dict[str, Any]] = data.get("metadatas") or []
        out: list[dict[str, Any]] = []
        for i in range(min(len(ids), len(docs), len(metas))):
            out.append({"chunk_id": ids[i], "text": docs[i], "metadata": metas[i] or {}})
        return out

    def delete_chunk_ids(self, *, collection: str, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        col = self._collection(collection)
        col.delete(ids=chunk_ids)
        return len(chunk_ids)

    def delete_collection(self, *, collection: str) -> None:
        client = self._get_client()
        try:
            client.delete_collection(name=collection)
        except Exception:
            pass

