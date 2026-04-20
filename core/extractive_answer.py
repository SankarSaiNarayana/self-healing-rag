from __future__ import annotations

import re
from collections.abc import Sequence

from rapidfuzz import fuzz

from core.text import normalize_whitespace, split_sentences

# Repeated “please review this doc” lines add noise and no technical content.
_BOILERPLATE = re.compile(
    r"(^|\b)(hi team[, ]*)?(please )?review (the )?documentation.*(missed|anything that i missed|what i can learn)\.?\s*$",
    re.I,
)


def _question_terms(q: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", q.lower()) if t not in _STOPWORDS}


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "she",
        "use",
        "her",
        "what",
        "when",
        "with",
        "have",
        "this",
        "that",
        "from",
        "they",
        "been",
        "into",
        "more",
        "some",
        "than",
        "them",
        "these",
        "will",
        "your",
        "about",
        "after",
        "also",
        "could",
        "does",
        "each",
        "just",
        "like",
        "make",
        "many",
        "most",
        "much",
        "only",
        "such",
        "their",
        "there",
        "then",
        "very",
        "were",
        "which",
        "would",
        "should",
        "could",
        "tell",
        "know",
        "want",
        "need",
        "help",
        "team",
        "please",
        "review",
        "let",
        "know",
        "learn",
        "add",
        "missed",
    }
)


def _skip_sentence(s: str) -> bool:
    t = normalize_whitespace(s)
    if len(t) < 28:
        return True
    if _BOILERPLATE.search(t):
        return True
    # Same greeting line repeated in many chunks
    if t.lower().startswith("hi team") and "review" in t.lower() and len(t) < 180:
        return True
    return False


def _rescale_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo + 1e-9:
        return [1.0 for _ in scores]
    return [(x - lo) / (hi - lo) for x in scores]


def synthesize_extractive_answer(
    *,
    question: str,
    chunks: Sequence[object],
    max_sentences: int = 8,
    max_chars: int = 1600,
) -> str:
    """
    Build a short, readable answer from retrieved chunks without an LLM.
    Picks and lightly orders sentences by relevance to the question; dedupes near-duplicates.
    """
    q = normalize_whitespace(question)
    if not q or not chunks:
        return ""

    q_terms = _question_terms(q)
    q_low = q.lower()

    scores_list = [float(getattr(c, "score", 0.0) or 0.0) for c in chunks]
    norm_chunk = _rescale_scores(scores_list)

    candidates: list[tuple[float, str]] = []
    for i, ch in enumerate(chunks[:10]):
        text = normalize_whitespace(str(getattr(ch, "text", "") or ""))
        if not text:
            continue
        w_chunk = norm_chunk[i] if i < len(norm_chunk) else 0.5
        for sent in split_sentences(text):
            if _skip_sentence(sent):
                continue
            sl = sent.lower()
            if q_terms:
                overlap = min(1.0, sum(1 for t in q_terms if t in sl) / float(len(q_terms)))
            else:
                overlap = 0.0
            fr = fuzz.token_set_ratio(q_low, sl) / 100.0
            pr = fuzz.partial_ratio(q_low, sl) / 100.0
            score = 0.38 * fr + 0.22 * pr + 0.25 * overlap + 0.15 * w_chunk
            candidates.append((score, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)

    picked: list[str] = []
    picked_lower: list[str] = []
    total_len = 0
    for _sc, sent in candidates:
        s_norm = normalize_whitespace(sent)
        if not s_norm:
            continue
        sl = s_norm.lower()
        if any(fuzz.ratio(sl, p) > 90 for p in picked_lower):
            continue
        picked.append(s_norm)
        picked_lower.append(sl)
        total_len += len(s_norm) + 1
        if len(picked) >= max_sentences or total_len >= max_chars:
            break

    if not picked:
        for ch in chunks[:3]:
            text = normalize_whitespace(str(getattr(ch, "text", "") or ""))
            for sent in split_sentences(text):
                if len(sent) >= 40 and not _skip_sentence(sent):
                    picked.append(normalize_whitespace(sent))
                    if len(picked) >= 4:
                        break
            if len(picked) >= 4:
                break

    if not picked:
        return ""

    body = " ".join(picked)
    body = normalize_whitespace(body)[:max_chars].strip()
    intro = (
        "Here is a concise summary built only from the passages that best match your question "
        "(wording is taken from your documents; nothing was invented by a model):"
    )
    return f"{intro}\n\n{body}"
