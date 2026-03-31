from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Prefer local-only load so demos work without external network.
    # If the model isn't cached locally, SentenceTransformer may raise.
    return SentenceTransformer(model_name, local_files_only=True)


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

