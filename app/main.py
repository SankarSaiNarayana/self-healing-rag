from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse
from app.schemas import ClaimEvidence, SourceChunk
from core.rag_pipeline import run_self_healing_rag
from core.settings import get_settings


app = FastAPI(title="Self-healing RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    llm_enabled = bool(settings.openai_base_url and settings.openai_api_key)
    return {"status": "ok", "llm_enabled": str(llm_enabled).lower()}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    out = await run_self_healing_rag(
        question=req.question,
        collection=req.collection,
        top_k=req.top_k,
        bm25_k=req.bm25_k,
        rerank=req.rerank,
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

    return QueryResponse(
        question=out.question,
        collection=out.collection,
        answer=out.answer,
        query_type=out.query_type,
        confidence=float(out.confidence),
        sources=sources,
        claims=claims,
        used_llm=bool(out.used_llm),
        retries=out.retries,
    )

