from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field

from dotenv import load_dotenv


# Ensure `.env` is loaded for local/dev runs (uvicorn, scripts, etc.).
# Environment variables already set in the shell still take precedence.
load_dotenv(override=False)


def _get_env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def _get_env_int(name: str, default: int) -> int:
    v = _get_env(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    v = _get_env(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_base_url: str | None = field(default_factory=lambda: _get_env("OPENAI_BASE_URL"))
    openai_api_key: str | None = field(default_factory=lambda: _get_env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: (_get_env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"))

    chroma_dir: str = field(default_factory=lambda: (_get_env("CHROMA_DIR", ".chroma") or ".chroma"))

    top_k: int = field(default_factory=lambda: _get_env_int("TOP_K", 8))
    bm25_k: int = field(default_factory=lambda: _get_env_int("BM25_K", 12))
    hybrid_alpha: float = field(default_factory=lambda: _get_env_float("HYBRID_ALPHA", 0.55))
    min_retrieval_score: float = field(default_factory=lambda: _get_env_float("MIN_RETRIEVAL_SCORE", 0.25))

    max_retrieval_retries: int = field(default_factory=lambda: _get_env_int("MAX_RETRIEVAL_RETRIES", 2))
    max_verification_retries: int = field(default_factory=lambda: _get_env_int("MAX_VERIFICATION_RETRIES", 2))


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    # Recreate to reflect latest environment (useful with reload/dev).
    return Settings()

