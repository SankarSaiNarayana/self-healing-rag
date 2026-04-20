from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Default HF cache + one-time download is noisy at INFO; keep warnings visible.
for _name in ("sentence_transformers", "transformers", "torch"):
    logging.getLogger(_name).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Prefer local cache (no network). If weights are missing, allow a one-time HF download.
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, local_files_only=False)


def _hash_embed(text: str, *, dim: int = 384) -> list[float]:
    # Deterministic, network-free fallback embedding for demos.
    # Not semantically strong, but good enough to exercise the pipeline end-to-end.
    v = np.zeros((dim,), dtype=np.float32)
    bs = text.encode("utf-8", errors="ignore")
    for i, b in enumerate(bs):
        v[(b + i) % dim] += 1.0
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v.tolist()


def embed_texts(texts: list[str], *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[list[float]]:
    if not texts:
        return []
    try:
        model = get_embedder(model_name)
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if isinstance(emb, list):
            return emb
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32).tolist()
        return [list(map(float, row)) for row in emb]
    except Exception:
        return [_hash_embed(t) for t in texts]

