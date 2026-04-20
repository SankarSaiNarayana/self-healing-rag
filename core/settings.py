from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from dotenv import load_dotenv

# Repo-root `.env` (works even when uvicorn is started from another cwd).
_REPO_ROOT = Path(__file__).resolve().parent.parent


# Ensure `.env` is loaded for local/dev runs (uvicorn, scripts, etc.).
# Variables already in the process environment are not overwritten (override=False).
load_dotenv(_REPO_ROOT / ".env", override=False)
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

    # ---- Memory (prod knobs) ----
    memory_enabled: bool = field(default_factory=lambda: (_get_env("MEMORY_ENABLED", "true") or "true").lower() == "true")
    memory_top_k: int = field(default_factory=lambda: _get_env_int("MEMORY_TOP_K", 4))
    procedural_db_path: str = field(default_factory=lambda: (_get_env("PROCEDURAL_DB_PATH", "storage/procedural_memory.sqlite") or "storage/procedural_memory.sqlite"))

    # ---- Vector DB backend ----
    vector_db: str = field(default_factory=lambda: (_get_env("VECTOR_DB", "chroma") or "chroma").lower())  # "chroma" | "qdrant"
    qdrant_url: str = field(default_factory=lambda: (_get_env("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: _get_env("QDRANT_API_KEY"))
    qdrant_collection_prefix: str = field(default_factory=lambda: (_get_env("QDRANT_COLLECTION_PREFIX", "rag__") or "rag__"))

    # ---- API / ops ----
    api_key: str | None = field(default_factory=lambda: _get_env("API_KEY"))
    rate_limit: str = field(default_factory=lambda: (_get_env("RATE_LIMIT", "120/minute") or "120/minute"))
    cors_origins: str = field(default_factory=lambda: (_get_env("CORS_ORIGINS", "*") or "*"))
    health_check_llm: bool = field(default_factory=lambda: (_get_env("HEALTH_CHECK_LLM", "false") or "false").lower() == "true")

    # ---- Document upload (API) ----
    upload_max_files: int = field(default_factory=lambda: _get_env_int("UPLOAD_MAX_FILES", 12))
    upload_max_bytes_per_file: int = field(default_factory=lambda: _get_env_int("UPLOAD_MAX_BYTES_PER_FILE", 12_000_000))
    upload_max_total_bytes: int = field(default_factory=lambda: _get_env_int("UPLOAD_MAX_TOTAL_BYTES", 48_000_000))
    upload_rate_limit: str = field(default_factory=lambda: (_get_env("UPLOAD_RATE_LIMIT", "20/minute") or "20/minute"))


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    # Recreate to reflect latest environment (useful with reload/dev).
    return Settings()

