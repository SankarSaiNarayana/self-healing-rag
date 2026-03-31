from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from core.retriever import RetrievedChunk
from core.text import normalize_whitespace, split_sentences


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    status: str  # "verified" | "uncertain" | "hallucinated"
    best_chunk: RetrievedChunk | None
    support_score: float
    rationale: str | None = None


def extract_claims(answer: str) -> list[str]:
    sents = split_sentences(answer)
    # Remove very short fragments.
    return [s.strip() for s in sents if len(s.strip()) >= 12]


def _support_score(claim: str, chunk_text: str) -> float:
    # Cheap lexical support proxy; good enough for demo without extra models.
    # token_set_ratio is robust to reordering.
    return fuzz.token_set_ratio(claim, chunk_text) / 100.0


def verify_claims(
    *,
    answer: str,
    chunks: list[RetrievedChunk],
    verified_threshold: float = 0.72,
    uncertain_threshold: float = 0.58,
) -> list[ClaimCheck]:
    claims = extract_claims(answer)
    out: list[ClaimCheck] = []
    if not claims:
        return out

    ctx = chunks or []
    for claim in claims:
        c = normalize_whitespace(claim)
        best: RetrievedChunk | None = None
        best_score = 0.0
        for ch in ctx:
            sc = _support_score(c, ch.text)
            if sc > best_score:
                best_score = sc
                best = ch

        if best_score >= verified_threshold:
            status = "verified"
            rationale = None
        elif best_score >= uncertain_threshold:
            status = "uncertain"
            rationale = "Partial match to sources; may need more context."
        else:
            status = "hallucinated"
            rationale = "No sufficiently similar supporting source found."

        out.append(ClaimCheck(claim=c, status=status, best_chunk=best, support_score=float(best_score), rationale=rationale))

    return out

