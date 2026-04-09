import logging
import os
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.deps import optional_api_key
from app.middleware import RequestIdMiddleware
from app.schemas import ClaimEvidence, QueryRequest, QueryResponse, SourceChunk
from core.health_checks import ping_llm_light, ping_vector_store
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


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    llm_enabled = bool(settings.openai_base_url and settings.openai_api_key)
    vs = ping_vector_store()
    out: dict[str, Any] = {
        "status": "ok" if vs.get("ok") else "degraded",
        "llm_enabled": llm_enabled,
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
    sources = list(list_unique_sources(collection))
    return {"collection": collection, "sources": sources, "n": len(sources)}


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
