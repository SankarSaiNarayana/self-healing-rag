from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from core.embeddings import embed_texts
from core.store import StoredChunk, upsert_chunks
from core.text import chunk_text, safe_join_lines


@dataclass(frozen=True)
class IngestedDoc:
    doc_id: str
    source: str
    text: str


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def load_text_file(path: Path) -> str:
    # Keep it simple for demo: plaintext/markdown-ish.
    # PDFs can be added later via a proper parser.
    return path.read_text(encoding="utf-8", errors="ignore")


def load_docs(data_dir: Path) -> list[IngestedDoc]:
    docs: list[IngestedDoc] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            continue
        text = load_text_file(p)
        text = safe_join_lines(text.splitlines())
        if not text:
            continue
        source = str(p.relative_to(data_dir))
        doc_id = _hash_id(source, str(len(text)))
        docs.append(IngestedDoc(doc_id=doc_id, source=source, text=text))
    return docs


def ingest(*, data_dir: str, collection: str, chunk_size: int = 900, chunk_overlap: int = 150) -> int:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    docs = load_docs(data_path)
    if not docs:
        return 0

    all_chunks: list[StoredChunk] = []
    all_texts: list[str] = []

    for d in docs:
        chunks = chunk_text(d.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            chunk_id = _hash_id(d.doc_id, str(i), c[:80])
            all_chunks.append(
                StoredChunk(
                    doc_id=d.doc_id,
                    chunk_id=chunk_id,
                    source=d.source,
                    text=c,
                    metadata={"chunk_index": i, "char_len": len(c)},
                )
            )
            all_texts.append(c)

    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding"):
        embeddings.extend(embed_texts(all_texts[i : i + batch_size]))

    return upsert_chunks(collection=collection, chunks=all_chunks, embeddings=embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument("--data_dir", required=True, help="Directory containing files to ingest")
    parser.add_argument("--collection", default="docs", help="Chroma collection name")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "900")))
    parser.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "150")))
    args = parser.parse_args()

    n = ingest(
        data_dir=args.data_dir,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingested {n} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()

