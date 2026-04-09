from __future__ import annotations


def test_import_app() -> None:
    from app.main import app

    assert app.title


def test_settings_defaults() -> None:
    from core.settings import get_settings

    s = get_settings()
    assert s.vector_db in ("chroma", "qdrant")
