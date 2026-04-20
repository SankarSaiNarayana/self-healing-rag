from __future__ import annotations


def test_upload_documents_and_query(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma_upload"))
    from core.vectorstore.factory import reset_vectorstore

    reset_vectorstore()

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    content = b"UniqueIngestTokenOmega753 rag upload smoke test paragraph."
    res = client.post(
        "/collections/smoke_upload/documents",
        files=[("files", ("smoke_note.txt", content, "text/plain"))],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["chunk_count"] >= 1
    assert any("uploads/smoke_note.txt" in s for s in body["sources"])

    q = client.post(
        "/query",
        json={
            "question": "What is UniqueIngestTokenOmega753?",
            "collection": "smoke_upload",
            "return_context": True,
        },
    )
    assert q.status_code == 200, q.text
    data = q.json()
    assert "UniqueIngestTokenOmega753" in data.get("answer", "") or any(
        "UniqueIngestTokenOmega753" in (s.get("text") or "") for s in data.get("sources", [])
    )


def test_import_app() -> None:
    from app.main import app

    assert app.title


def test_extractive_answer_is_prose_not_bullet_dump() -> None:
    from core.extractive_answer import synthesize_extractive_answer
    from core.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            doc_id="1",
            chunk_id="a",
            source="notes.txt",
            score=0.9,
            text=(
                "Hi team, Please review the documentation and let me know what I can learn. "
                "docker start [container_id] starts a stopped container. "
                "docker restart restarts a running container."
            ),
        ),
        RetrievedChunk(
            doc_id="1",
            chunk_id="b",
            source="notes.txt",
            score=0.5,
            text="Hi team, Please review the documentation and let me know what I can learn.",
        ),
    ]
    out = synthesize_extractive_answer(question="How do I start a stopped Docker container?", chunks=chunks)
    assert "From the available sources" not in out
    assert "docker start" in out.lower()
    assert "- docker" not in out  # not a bullet list of raw chunks


def test_settings_defaults() -> None:
    from core.settings import get_settings

    s = get_settings()
    assert s.vector_db in ("chroma", "qdrant")
