from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from core.settings import get_settings


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def llm_enabled() -> bool:
    s = get_settings()
    return bool(s.openai_base_url and s.openai_api_key and s.openai_model)


async def chat_complete(messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
    s = get_settings()
    if not llm_enabled():
        raise RuntimeError("LLM not configured. Set OPENAI_BASE_URL and OPENAI_API_KEY.")

    url = s.openai_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": s.openai_model,
        "temperature": temperature,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }

    headers = {
        "Authorization": f"Bearer {s.openai_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, headers=headers, content=json.dumps(payload))
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            body = None
            try:
                body = e.response.text[:2000]
            except Exception:
                body = None
            raise LLMError(
                f"LLM request failed ({e.response.status_code}). Check OPENAI_* env and provider status.",
                status_code=int(e.response.status_code),
                body=body,
            ) from e
        except httpx.RequestError as e:
            raise LLMError(f"LLM request error: {type(e).__name__}") from e

