from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from core.settings import get_settings
from core.vectorstore.base import StoredChunk, VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            s = get_settings()
            self._client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        return self._client

    def _name(self, collection: str) -> str:
        s = get_settings()
        return f"{s.qdrant_collection_prefix}{collection}"

    def _ensure_collection(self, name: str, dim: int) -> None:
        client = self._get_client()
        existing = {c.name for c in client.get_collections().collections}
        if name in existing:
            return
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert_chunks(self, *, collection: str, chunks: list[StoredChunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        name = self._name(collection)
        dim = len(embeddings[0]) if embeddings and embeddings[0] else 384
        self._ensure_collection(name, dim)

        points: list[PointStruct] = []
        for c, v in zip(chunks, embeddings, strict=False):
            payload = {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "source": c.source,
                "text": c.text,
                **(c.metadata or {}),
            }
            points.append(PointStruct(id=c.chunk_id, vector=v, payload=payload))

        self._get_client().upsert(collection_name=name, points=points)
        return len(points)

    def query(self, *, collection: str, query_embedding: list[float], n_results: int) -> dict[str, Any]:
        name = self._name(collection)
        client = self._get_client()
        hits = client.search(collection_name=name, query_vector=query_embedding, limit=n_results, with_payload=True)

        ids = [str(h.id) for h in hits]
        docs = [str((h.payload or {}).get("text", "")) for h in hits]
        metas = [dict(h.payload or {}) for h in hits]
        # Convert similarity score to a "distance" compatible with old interface: dist = 1 - sim
        dists = [float(1.0 - float(h.score)) for h in hits]
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    def get_all_chunks(self, *, collection: str) -> list[dict[str, Any]]:
        name = self._name(collection)
        client = self._get_client()
        out: list[dict[str, Any]] = []

        offset = None
        while True:
            points, offset = client.scroll(collection_name=name, limit=256, offset=offset, with_payload=True)
            for p in points:
                payload = dict(p.payload or {})
                out.append({"chunk_id": str(p.id), "text": str(payload.get("text", "")), "metadata": payload})
            if offset is None:
                break
        return out

    def delete_chunk_ids(self, *, collection: str, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        name = self._name(collection)
        client = self._get_client()
        client.delete(collection_name=name, points_selector=PointIdsList(points=chunk_ids))
        return len(chunk_ids)

    def delete_collection(self, *, collection: str) -> None:
        name = self._name(collection)
        client = self._get_client()
        try:
            client.delete_collection(collection_name=name)
        except Exception:
            pass

