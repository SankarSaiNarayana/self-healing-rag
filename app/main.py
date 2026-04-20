import logging
import os
import re
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.deps import optional_api_key
from app.middleware import RequestIdMiddleware
from app.schemas import ClaimEvidence, IngestResponse, QueryRequest, QueryResponse, SkippedFile, SourceChunk
from core.health_checks import ping_llm_light, ping_vector_store
from core.llm import llm_enabled as llm_configured
from core.ingest_service import ingest_uploaded_files
from core.memory.clear_memory import clear_user_memory
from core.memory.episodic_memory import list_episodic_recent
from core.memory.procedural_memory import list_strategy_stats
from core.query_router import list_unique_sources
from core.rag_pipeline import run_self_healing_rag
from core.settings import get_settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

_settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Self-healing RAG", version="0.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)

_COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _validate_collection(name: str) -> str:
    n = name.strip()
    if not _COLLECTION_RE.fullmatch(n):
        raise HTTPException(
            status_code=400,
            detail="Collection must be 1–64 characters, start with a letter or digit, and contain only letters, digits, underscore, or hyphen.",
        )
    return n


async def _read_upload_to_limit(upload: UploadFile, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(512 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File {upload.filename!r} exceeds max size ({max_bytes} bytes)",
            )
    return bytes(buf)


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    vs = ping_vector_store()
    llm_ok = llm_configured()
    out: dict[str, Any] = {
        "status": "ok" if vs.get("ok") else "degraded",
        "llm_enabled": llm_ok,
        "llm": {
            "has_openai_base_url": bool(settings.openai_base_url),
            "has_openai_api_key": bool(settings.openai_api_key),
            "has_openai_model": bool((settings.openai_model or "").strip()),
            "model": settings.openai_model,
            "hint": (
                None
                if llm_ok
                else "Set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL in a .env file at the project root "
                "(or export them before starting uvicorn). Example for Groq: base "
                "https://api.groq.com/openai/v1 — without these, answers use extractive fallback only."
            ),
        },
        "vector_db": settings.vector_db,
        "memory_enabled": settings.memory_enabled,
        "vector_store": vs,
        "api_key_required": bool(settings.api_key),
    }
    if settings.health_check_llm:
        out["llm_probe"] = await ping_llm_light()
    return out


@app.get("/demo")
async def demo_page() -> FileResponse:
    path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    return FileResponse(path)


@app.get("/memory/{user_id}")
async def memory_debug(
    user_id: str,
    _: None = Depends(optional_api_key),
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "episodic": list_episodic_recent(user_id=user_id, limit=40),
        "procedural": list_strategy_stats(user_id=user_id, limit=50),
    }


@app.delete("/memory/{user_id}")
async def memory_clear(
    user_id: str,
    _: None = Depends(optional_api_key),
) -> dict[str, Any]:
    """Clear episodic + semantic vector memory and procedural strategy stats for this user (not document index)."""
    return clear_user_memory(user_id=user_id)


@app.get("/collections/{collection}/sources")
async def collection_sources(
    collection: str,
    _: None = Depends(optional_api_key),
) -> dict[str, Any]:
    """List unique `source` paths in the index (for choosing source_hint)."""
    col = _validate_collection(collection)
    sources = list(list_unique_sources(col))
    return {"collection": col, "sources": sources, "n": len(sources)}


@app.post("/collections/{collection}/documents", response_model=IngestResponse)
@limiter.limit(_settings.upload_rate_limit)
async def upload_documents(
    request: Request,
    collection: str,
    files: list[UploadFile] = File(),
    _: None = Depends(optional_api_key),
) -> IngestResponse:
    """Upload .txt / .md / .pdf files into the vector index for this collection, then query via POST /query."""
    col = _validate_collection(collection)
    settings = get_settings()
    if not files:
        raise HTTPException(
            status_code=400,
            detail='Add at least one file using the multipart field name "files" (repeat the field for multiple files).',
        )
    if len(files) > settings.upload_max_files:
        raise HTTPException(status_code=400, detail=f"Too many files (max {settings.upload_max_files}).")

    pairs: list[tuple[str, bytes]] = []
    total = 0
    for uf in files:
        raw_name = uf.filename or "unnamed"
        data = await _read_upload_to_limit(uf, settings.upload_max_bytes_per_file)
        total += len(data)
        if total > settings.upload_max_total_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload size exceeds {settings.upload_max_total_bytes} bytes.",
            )
        pairs.append((raw_name, data))

    out = ingest_uploaded_files(
        collection=col,
        files=pairs,
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )
    skipped_raw = out.get("skipped") or []
    skipped_models = [SkippedFile(filename=str(s.get("filename", "")), reason=str(s.get("reason", ""))) for s in skipped_raw]
    return IngestResponse(
        collection=col,
        chunk_count=int(out.get("chunk_count", 0)),
        sources=list(out.get("sources") or []),
        skipped=skipped_models,
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit(_settings.rate_limit)
async def query(
    request: Request,
    req: Annotated[QueryRequest, Body()],
    _: None = Depends(optional_api_key),
) -> QueryResponse:
    out = await run_self_healing_rag(
        question=req.question,
        collection=req.collection,
        user_id=req.user_id,
        top_k=req.top_k,
        bm25_k=req.bm25_k,
        rerank=req.rerank,
        source_hint=req.source_hint,
        auto_source=req.auto_source,
    )

    sources = [
        SourceChunk(
            doc_id=c.doc_id,
            chunk_id=c.chunk_id,
            source=c.source,
            score=float(c.score),
            text=c.text if req.return_context else None,
        )
        for c in out.sources
    ]

    claims = []
    for c in out.claims:
        src = None
        if c.best_chunk is not None:
            src = SourceChunk(
                doc_id=c.best_chunk.doc_id,
                chunk_id=c.best_chunk.chunk_id,
                source=c.best_chunk.source,
                score=float(c.best_chunk.score),
                text=c.best_chunk.text if req.return_context else None,
            )
        claims.append(ClaimEvidence(claim=c.claim, status=c.status, source=src, rationale=c.rationale))

    rid = getattr(request.state, "request_id", None)
    return QueryResponse(
        question=out.question,
        collection=out.collection,
        user_id=out.user_id,
        answer=out.answer,
        query_type=out.query_type,
        confidence=float(out.confidence),
        sources=sources,
        claims=claims,
        used_llm=bool(out.used_llm),
        retries=out.retries,
        warnings=out.warnings,
        request_id=str(rid) if rid else None,
        router=out.router,
    )
