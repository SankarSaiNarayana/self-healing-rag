from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from tqdm import tqdm

from core.embeddings import embed_texts
from core.query_router import invalidate_source_list_cache
from core.retriever import invalidate_bm25_cache
from core.store import StoredChunk, delete_chunks_for_sources, upsert_chunks
from core.text import chunk_text, safe_join_lines


@dataclass(frozen=True)
class IngestedDoc:
    doc_id: str
    source: str
    text: str


_ALLOWED_SUFFIX = {".txt", ".md", ".markdown", ".pdf"}


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def sanitize_upload_filename(raw: str) -> str:
    """Basename only; reject path tricks and unsupported types."""
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or len(name) > 180:
        raise ValueError("invalid filename")
    if name.startswith(".") or ".." in name:
        raise ValueError("invalid filename")
    for bad in ("/", "\\", "\x00"):
        if bad in name:
            raise ValueError("invalid filename")
    suf = Path(name).suffix.lower()
    if suf == ".markdown":
        suf = ".md"
    if suf not in _ALLOWED_SUFFIX:
        raise ValueError("unsupported file type (use .txt, .md, or .pdf)")
    return name


def load_text_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        if path.stat().st_size == 0:
            print(f"[ingest] skip (empty file): {path}", file=sys.stderr)
            return ""
        try:
            reader = PdfReader(str(path))
        except PdfReadError as e:
            print(f"[ingest] skip (unreadable PDF): {path} ({e})", file=sys.stderr)
            return ""
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_bytes(filename: str, data: bytes) -> str:
    suf = Path(filename).suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        if not data:
            return ""
        try:
            reader = PdfReader(BytesIO(data))
        except PdfReadError:
            return ""
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


def load_docs_from_dir(data_dir: Path) -> list[IngestedDoc]:
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


def _chunk_and_embed(
    docs: list[IngestedDoc],
    *,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    show_progress: bool,
) -> int:
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

    embeddings: list[list[float]] = []
    batch_size = 64
    it = range(0, len(all_texts), batch_size)
    if show_progress:
        it = tqdm(it, desc="Embedding")
    for i in it:
        embeddings.extend(embed_texts(all_texts[i : i + batch_size]))

    n = upsert_chunks(collection=collection, chunks=all_chunks, embeddings=embeddings)
    invalidate_bm25_cache()
    invalidate_source_list_cache()
    return n


def ingest_from_directory(
    *,
    data_dir: str,
    collection: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> int:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    docs = load_docs_from_dir(data_path)
    if not docs:
        return 0
    return _chunk_and_embed(docs, collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, show_progress=True)


def ingest_uploaded_files(
    *,
    collection: str,
    files: list[tuple[str, bytes]],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    source_prefix: str = "uploads/",
) -> dict[str, object]:
    """
    Ingest raw uploads into `collection`.

    Each file is stored under `source_prefix` + sanitized basename.
    Re-uploading the same basename replaces prior chunks for that source.
    """
    if not files:
        return {"chunk_count": 0, "sources": [], "skipped": []}

    prefix = source_prefix if source_prefix.endswith("/") else f"{source_prefix}/"
    skipped: list[dict[str, str]] = []
    docs: list[IngestedDoc] = []
    sources_to_replace: list[str] = []

    for raw_name, data in files:
        try:
            safe = sanitize_upload_filename(raw_name)
        except ValueError as e:
            skipped.append({"filename": raw_name, "reason": str(e)})
            continue
        source = f"{prefix}{safe}"
        text = load_document_bytes(safe, data)
        text = safe_join_lines(text.splitlines())
        if not text:
            skipped.append({"filename": raw_name, "reason": "empty or unreadable"})
            continue
        doc_id = _hash_id(source, str(len(text)))
        docs.append(IngestedDoc(doc_id=doc_id, source=source, text=text))
        sources_to_replace.append(source)

    if sources_to_replace:
        delete_chunks_for_sources(collection=collection, sources=sources_to_replace)

    if not docs:
        return {"chunk_count": 0, "sources": [], "skipped": skipped}

    n = _chunk_and_embed(docs, collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, show_progress=False)
    return {"chunk_count": n, "sources": [d.source for d in docs], "skipped": skipped}
