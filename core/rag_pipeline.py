from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.classifier import classify_query
from core.hallucination_detector import ClaimCheck, verify_claims
from core.llm import ChatMessage, LLMError, chat_complete, llm_enabled
from core.memory.episodic_memory import format_past_questions_answer, read_episodic, write_episodic
from core.memory.procedural_memory import Strategy, choose_strategy, update_strategy_stats
from core.memory.semantic_memory import read_facts, write_facts
from core.retriever import RetrievedChunk, RetrievalResult, retrieve_with_self_heal
from core.settings import get_settings
from core.text import normalize_whitespace


@dataclass(frozen=True)
class PipelineOutput:
    question: str
    collection: str
    user_id: str
    query_type: str
    answer: str
    sources: list[RetrievedChunk]
    claims: list[ClaimCheck]
    confidence: float
    used_llm: bool
    retries: dict[str, int]
    retrieval: RetrievalResult
    memory_used: dict[str, int]
    warnings: list[str]
    router: dict[str, Any] | None = None


_MEMORY_QUERY_RE = re.compile(r"\b(what did i ask before|previous questions?|what have i asked|my last question)\b", re.I)
_K8S_TOPIC = re.compile(r"\b(kubernetes|k8s)\b", re.I)

# Stored semantic "facts" can echo prior model hedges; re-injecting them verifies the hedge and poisons the next answer.
_TOXIC_SEMANTIC_SUBSTR = (
    "not available in the provided context",
    "insufficient context",
    "insufficient information",
    "however, a clear definition",
    "no clear definition",
    "a clear definition of kubernetes is not",
)


def _is_toxic_semantic_echo(text: str) -> bool:
    t = normalize_whitespace(text).lower()
    return any(s in t for s in _TOXIC_SEMANTIC_SUBSTR)


def _mentions_k8s(text: str) -> bool:
    t = normalize_whitespace(text).lower()
    return "kubernetes" in t or "k8s" in t or " kube" in t or t.startswith("kube")


def _is_docker_centric_memory_blob(text: str) -> bool:
    """Prior Docker Q/A or facts that should not steer a Kubernetes answer."""
    if _mentions_k8s(text):
        return False
    t = normalize_whitespace(text).lower()
    if "docker" not in t:
        return False
    return (
        "os-level virtualization" in t
        or "os level virtualization" in t
        or "platform-as-a-service" in t
        or "what is docker" in t
        or "containers are lightweight" in t
    )


def _filter_semantic_for_prompt(question: str, semantic: list[RetrievedChunk]) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for c in semantic:
        if _is_toxic_semantic_echo(c.text):
            continue
        if _K8S_TOPIC.search(question) and _is_docker_centric_memory_blob(c.text):
            continue
        out.append(c)
    return out


def _filter_episodic_for_prompt(question: str, episodic: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not _K8S_TOPIC.search(question):
        return episodic
    return [c for c in episodic if not _is_docker_centric_memory_blob(c.text)]


def _spurious_docker_line_for_k8s_question(question: str, claim: str) -> bool:
    if not _K8S_TOPIC.search(question):
        return False
    if _mentions_k8s(claim) or "pod" in claim.lower() or "cluster" in claim.lower():
        return False
    cl = normalize_whitespace(claim).lower()
    return bool(
        re.search(
            r"docker|os[- ]level virtualization|platform-as-a-service|containers are lightweight",
            cl,
        )
    )


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


def _format_memory(episodic: list[RetrievedChunk], semantic: list[RetrievedChunk], *, max_chars: int = 3000) -> str:
    blocks: list[str] = []
    if episodic:
        blocks.append("Episodic memory (past Q/A):")
        for i, it in enumerate(episodic[:3], start=1):
            blocks.append(f"(E{i}) {it.text}")
    if semantic:
        blocks.append("Semantic memory (facts):")
        for i, it in enumerate(semantic[:6], start=1):
            blocks.append(f"(S{i}) {it.text}")
    txt = "\n".join(blocks).strip()
    return txt[:max_chars]


async def _generate_answer(question: str, chunks: list[RetrievedChunk], memory_text: str | None = None) -> tuple[str, bool, list[str]]:
    q = normalize_whitespace(question)
    warnings: list[str] = []

    def fallback(reason: str) -> tuple[str, bool, list[str]]:
        ctx = "\n\n".join([f"- {normalize_whitespace(c.text)[:350]}" for c in chunks[:4]])
        if not ctx:
            return "I couldn't find relevant sources for that question in the collection.", False, [reason, "extractive_fallback:no_context"]
        return f"From the available sources, here are the most relevant excerpts:\n{ctx}", False, [reason, "extractive_fallback"]

    if not llm_enabled():
        return fallback("llm_not_configured")

    context = _format_context(chunks)
    k8s_extra = ""
    if _K8S_TOPIC.search(q):
        k8s_extra = (
            "The user is asking about Kubernetes. Do not paste in generic container/Docker marketing "
            "phrases (e.g. 'OS-level virtualization', 'platform-as-a-service', 'containers are lightweight') "
            "unless that exact wording appears in the Context. Prefer terms actually present (Pods, Services, "
            "cluster, CNI, Deployments, etc.).\n"
        )
    system = (
        k8s_extra
        + "You are a careful RAG assistant. Answer using ONLY the provided context.\n"
        "You may use the provided memory as user-specific preference/facts, but DO NOT invent citations.\n"
        "If you truly cannot answer from the context, say what is missing—briefly.\n"
        "For 'what is X' questions, synthesize a concise explanation from whatever the context says about "
        "X's role, parts, or behavior—even if there is no single formal definition sentence.\n"
        "If you already explained from the context, do not add a final sentence that claims the context "
        "lacks a definition or is insufficient (that contradicts a good synthesis).\n"
        "Write 3-8 sentences. Avoid speculation.\n"
    )
    mem = f"\n\nMemory:\n{memory_text}" if memory_text else ""
    user = f"Question: {q}\n\nContext:\n{context}{mem}"
    try:
        text = await chat_complete(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            temperature=0.2,
        )
        return normalize_whitespace(text), True, warnings
    except LLMError as e:
        code = getattr(e, "status_code", None)
        detail = f"llm_error:{code}" if code else "llm_error"
        a, u, w = fallback(detail)
        return a, u, w + [detail, f"llm_message:{str(e)[:120]}"]


def _confidence_from_claims(claims: list[ClaimCheck], retrieval_max_score: float) -> float:
    if not claims:
        return max(0.0, min(1.0, float(retrieval_max_score)))
    weights = {"verified": 1.0, "uncertain": 0.55, "hallucinated": 0.0}
    base = sum(weights.get(c.status, 0.0) for c in claims) / max(1, len(claims))
    return max(0.0, min(1.0, 0.70 * base + 0.30 * float(retrieval_max_score)))


async def run_self_healing_rag(
    *,
    question: str,
    collection: str,
    user_id: str = "demo",
    top_k: int | None = None,
    bm25_k: int | None = None,
    rerank: bool = True,
    source_hint: str | None = None,
    auto_source: bool = False,
) -> PipelineOutput:
    settings = get_settings()
    q = normalize_whitespace(question)
    qtype = classify_query(q)

    episodic_mem: list[RetrievedChunk] = []
    semantic_mem: list[RetrievedChunk] = []
    memory_used = {"episodic": 0, "semantic": 0, "procedural": 0}
    memory_text: str | None = None

    strategy: Strategy | None = None
    if settings.memory_enabled:
        strategy = choose_strategy(user_id=user_id, query_type=qtype)
        memory_used["procedural"] = 1
        episodic_mem = read_episodic(user_id=user_id, question=q, k=min(8, settings.memory_top_k * 2))
        semantic_mem = read_facts(user_id=user_id, question=q, k=min(6, settings.memory_top_k + 2))
        memory_used["episodic"] = len(episodic_mem)
        memory_used["semantic"] = len(semantic_mem)

        epi_prompt = _filter_episodic_for_prompt(q, episodic_mem)
        sem_prompt = _filter_semantic_for_prompt(q, semantic_mem)
        memory_text = _format_memory(epi_prompt, sem_prompt) if (epi_prompt or sem_prompt) else None

    if settings.memory_enabled and _MEMORY_QUERY_RE.search(q):
        if episodic_mem:
            answer = format_past_questions_answer(episodic_mem, max_items=settings.memory_top_k)
            claims: list[ClaimCheck] = []
            confidence = 0.85
        else:
            answer = "I don't have any prior questions saved for this user yet."
            claims = []
            confidence = 0.3

        if settings.memory_enabled:
            write_episodic(user_id=user_id, question=q, answer=answer, confidence=confidence, faithfulness=1.0)

        return PipelineOutput(
            question=q,
            collection=collection,
            user_id=user_id,
            query_type=qtype,
            answer=answer,
            sources=episodic_mem,
            claims=claims,
            confidence=float(confidence),
            used_llm=False,
            retries={"retrieval": 0, "verification": 0},
            retrieval=RetrievalResult(query=q, chunks=[], max_score=0.0, diagnostics={"mode": "memory_only"}),
            memory_used=memory_used,
            warnings=["mode:memory_query_short_circuit"],
            router=None,
        )

    retrieval, retrieval_retries = retrieve_with_self_heal(
        question=q,
        collection=collection,
        query_type=qtype,
        top_k=(top_k if strategy is None else (top_k or settings.top_k) + strategy.top_k_boost),
        bm25_k=(bm25_k if strategy is None else (bm25_k or settings.bm25_k) + strategy.bm25_k_boost),
        rerank=(rerank if strategy is None else strategy.rerank),
        hybrid_alpha=(None if strategy is None else strategy.hybrid_alpha),
        source_hint=source_hint,
        auto_source=auto_source,
    )

    verification_retries = 0
    answer, used_llm, gen_warnings = await _generate_answer(q, retrieval.chunks, memory_text=memory_text)
    # Verify against retrieved documents only. Episodic and semantic memory echo prior answers and
    # falsely "verify" hedges or recycle bad sentences.
    evidence_chunks = retrieval.chunks
    claims = verify_claims(answer=answer, chunks=evidence_chunks)

    for attempt in range(settings.max_verification_retries):
        hallucinated = [c for c in claims if c.status == "hallucinated"]
        if not claims:
            break
        if len(hallucinated) <= max(1, len(claims) // 4):
            break

        verification_retries += 1
        retrieval, _ = retrieve_with_self_heal(
            question=f"{q} provide evidence with sources",
            collection=collection,
            query_type=qtype,
            top_k=(top_k or settings.top_k) + 6 + 4 * attempt,
            bm25_k=(bm25_k or settings.bm25_k) + 10 + 6 * attempt,
            rerank=(rerank if strategy is None else strategy.rerank),
            hybrid_alpha=(None if strategy is None else strategy.hybrid_alpha),
            source_hint=source_hint,
            auto_source=auto_source,
        )
        answer, used_llm, gen_warnings = await _generate_answer(q, retrieval.chunks, memory_text=memory_text)
        evidence_chunks = retrieval.chunks
        claims = verify_claims(answer=answer, chunks=evidence_chunks)

    confidence = _confidence_from_claims(claims, retrieval.max_score)
    all_warnings = list(gen_warnings)
    router_meta = retrieval.diagnostics.get("router") if retrieval.diagnostics else None
    if isinstance(router_meta, dict) and router_meta.get("filter_fallback"):
        all_warnings.append("router:source_hint_matched_no_chunks_used_full_retrieval")

    faithfulness = sum(1.0 for c in claims if c.status == "verified") / max(1, len(claims)) if claims else 0.0
    if settings.memory_enabled:
        write_episodic(
            user_id=user_id,
            question=q,
            answer=answer,
            confidence=float(confidence),
            faithfulness=float(faithfulness),
        )
        verified_facts = [c.claim for c in claims if c.status == "verified"][:10]
        verified_facts = [
            f
            for f in verified_facts
            if len(f) <= 400 and not _is_toxic_semantic_echo(f) and not _spurious_docker_line_for_k8s_question(q, f)
        ]
        if verified_facts:
            write_facts(user_id=user_id, facts=verified_facts, source=collection)
        if strategy is not None:
            update_strategy_stats(
                user_id=user_id,
                query_type=qtype,
                strategy=strategy,
                faithfulness=float(faithfulness),
                retrieval_retries=int(retrieval_retries),
                verification_retries=int(verification_retries),
            )

    return PipelineOutput(
        question=q,
        collection=collection,
        user_id=user_id,
        query_type=qtype,
        answer=answer,
        sources=retrieval.chunks,
        claims=claims,
        confidence=float(confidence),
        used_llm=bool(used_llm),
        retries={"retrieval": int(retrieval_retries), "verification": int(verification_retries)},
        retrieval=retrieval,
        memory_used=memory_used,
        warnings=all_warnings,
        router=router_meta if isinstance(router_meta, dict) and router_meta else None,
    )
