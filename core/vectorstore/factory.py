from __future__ import annotations

from core.settings import get_settings
from core.vectorstore.base import VectorStore
from core.vectorstore.chroma_store import ChromaVectorStore
from core.vectorstore.qdrant_store import QdrantVectorStore


_STORE: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    global _STORE
    if _STORE is not None:
        return _STORE

    s = get_settings()
    if s.vector_db == "qdrant":
        _STORE = QdrantVectorStore()
    else:
        _STORE = ChromaVectorStore()
    return _STORE

