from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert_chunks(self, *, collection: str, chunks: list[StoredChunk], embeddings: list[list[float]]) -> int: ...
    def query(self, *, collection: str, query_embedding: list[float], n_results: int) -> dict[str, Any]: ...
    def get_all_chunks(self, *, collection: str) -> list[dict[str, Any]]: ...
    def delete_chunk_ids(self, *, collection: str, chunk_ids: list[str]) -> int: ...
    def delete_collection(self, *, collection: str) -> None: ...

