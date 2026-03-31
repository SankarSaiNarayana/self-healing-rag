from __future__ import annotations

from dataclasses import dataclass

from core.classifier import classify_query
from core.hallucination_detector import ClaimCheck, verify_claims
from core.llm import ChatMessage, chat_complete, llm_enabled
from core.retriever import RetrievedChunk, RetrievalResult, retrieve_with_self_heal
from core.settings import get_settings
from core.text import normalize_whitespace


@dataclass(frozen=True)
class PipelineOutput:
    question: str
    collection: str
    query_type: str
    answer: str
    sources: list[RetrievedChunk]
    claims: list[ClaimCheck]
    confidence: float
    used_llm: bool
    retries: dict[str, int]
    retrieval: RetrievalResult


def _format_context(chunks: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        block = f"[{i}] source={ch.source} chunk_id={ch.chunk_id}\n{ch.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


async def _generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool]:
    q = normalize_whitespace(question)
    if not llm_enabled():
        # Extractive fallback: stitch top chunks with minimal framing.
        ctx = "\n\n".join([f"- {normalize_whitespace(c.text)[:350]}" for c in chunks[:4]])
        if not ctx:
            return "I couldn't find relevant sources for that question in the collection.", False
        return f"From the available sources, here are the most relevant excerpts:\n{ctx}", False

    context = _format_context(chunks)
    system = (
        "You are a careful RAG assistant. Answer using ONLY the provided context.\n"
        "If the context is insufficient, say what is missing.\n"
        "Write 3-8 sentences. Avoid speculation.\n"
    )
    user = f"Question: {q}\n\nContext:\n{context}"
    text = await chat_complete([ChatMessage(role='system', content=system), ChatMessage(role='user', content=user)], temperature=0.2)
    return normalize_whitespace(text), True


def _confidence_from_claims(claims: list[ClaimCheck], retrieval_max_score: float) -> float:
    if not claims:
        return max(0.0, min(1.0, float(retrieval_max_score)))
    weights = {"verified": 1.0, "uncertain": 0.55, "hallucinated": 0.0}
    base = sum(weights.get(c.status, 0.0) for c in claims) / max(1, len(claims))
    # Incorporate retrieval confidence to penalize low-quality retrieval.
    return max(0.0, min(1.0, 0.70 * base + 0.30 * float(retrieval_max_score)))


async def run_self_healing_rag(
    *,
    question: str,
    collection: str,
    top_k: int | None = None,
    bm25_k: int | None = None,
    rerank: bool = True,
) -> PipelineOutput:
    settings = get_settings()
    q = normalize_whitespace(question)
    qtype = classify_query(q)

    retrieval, retrieval_retries = retrieve_with_self_heal(
        question=q,
        collection=collection,
        query_type=qtype,
        top_k=top_k,
        bm25_k=bm25_k,
        rerank=rerank,
    )

    verification_retries = 0
    used_llm = False

    answer, used_llm = await _generate_answer(q, retrieval.chunks)
    claims = verify_claims(answer=answer, chunks=retrieval.chunks)

    # Self-healing verification loop: if many claims are hallucinated, broaden retrieval and regenerate.
    for attempt in range(settings.max_verification_retries):
        hallucinated = [c for c in claims if c.status == "hallucinated"]
        if not claims:
            break
        if len(hallucinated) <= max(1, len(claims) // 4):
            break

        verification_retries += 1
        # Broaden retrieval and re-generate.
        retrieval, _ = retrieve_with_self_heal(
            question=f"{q} provide evidence with sources",
            collection=collection,
            query_type=qtype,
            top_k=(top_k or settings.top_k) + 6 + 4 * attempt,
            bm25_k=(bm25_k or settings.bm25_k) + 10 + 6 * attempt,
            rerank=rerank,
        )
        answer, used_llm = await _generate_answer(q, retrieval.chunks)
        claims = verify_claims(answer=answer, chunks=retrieval.chunks)

    confidence = _confidence_from_claims(claims, retrieval.max_score)
    return PipelineOutput(
        question=q,
        collection=collection,
        query_type=qtype,
        answer=answer,
        sources=retrieval.chunks,
        claims=claims,
        confidence=float(confidence),
        used_llm=bool(used_llm),
        retries={"retrieval": int(retrieval_retries), "verification": int(verification_retries)},
        retrieval=retrieval,
    )

