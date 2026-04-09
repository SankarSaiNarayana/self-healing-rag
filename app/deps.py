from __future__ import annotations

from fastapi import Header, HTTPException, status

from core.settings import get_settings


async def optional_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    s = get_settings()
    if not s.api_key:
        return
    if not x_api_key or x_api_key != s.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
