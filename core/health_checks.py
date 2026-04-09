from __future__ import annotations

from typing import Any

from core.settings import get_settings


def ping_vector_store() -> dict[str, Any]:
    s = get_settings()
    try:
        if s.vector_db == "qdrant":
            from qdrant_client import QdrantClient

            client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
            cols = client.get_collections()
            return {"ok": True, "backend": "qdrant", "collections": len(cols.collections)}
        from core.vectorstore.chroma_store import ChromaVectorStore

        ChromaVectorStore()._get_client()
        return {"ok": True, "backend": "chroma"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


async def ping_llm_light() -> dict[str, Any]:
    """Optional tiny probe when HEALTH_CHECK_LLM=true (adds latency + uses quota)."""
    from core.llm import ChatMessage, chat_complete, llm_enabled

    if not llm_enabled():
        return {"ok": False, "skipped": True, "reason": "llm_not_configured"}
    try:
        await chat_complete(
            [ChatMessage(role="user", content="Reply with exactly: ok")],
            temperature=0.0,
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
