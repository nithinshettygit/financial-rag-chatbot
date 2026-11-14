import asyncio
import hashlib
import hmac
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Response, Security
from fastapi.security import APIKeyHeader
from redis.exceptions import RedisError

from app.config import settings
from app.utils.cache import get_redis_client
from app.utils.logger import logger

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


def validate_security_settings() -> None:
    if not settings.CHAT_API_KEYS:
        raise RuntimeError("CHAT_API_KEYS must contain at least one API key")
    if any(not client_name.strip() for client_name in settings.CHAT_API_KEYS):
        raise RuntimeError("CHAT_API_KEYS client names must not be empty")
    if any(len(api_key.encode("utf-8")) < 32 for api_key in settings.CHAT_API_KEYS.values()):
        raise RuntimeError("Every CHAT_API_KEYS value must be at least 32 bytes")


def authenticate_api_key(api_key: str | None, configured_keys: dict[str, str]) -> str:
    if not configured_keys:
        raise HTTPException(status_code=503, detail="Chat authentication is not configured")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    matched = False
    for configured_key in configured_keys.values():
        matched = hmac.compare_digest(
            api_key.encode("utf-8"),
            configured_key.encode("utf-8"),
        ) or matched
    if not matched:
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def require_chat_api_key(
    api_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> str:
    return authenticate_api_key(api_key, settings.CHAT_API_KEYS)


async def enforce_chat_rate_limit(
    response: Response,
    key_digest: Annotated[str, Depends(require_chat_api_key)],
) -> None:
    now = int(time.time())
    window_seconds = settings.CHAT_RATE_LIMIT_WINDOW_SECONDS
    window = now // window_seconds
    reset_at = (window + 1) * window_seconds
    retry_after = max(1, reset_at - now)
    redis_key = f"financial-rag:chat-rate:v1:{key_digest}:{window}"

    try:
        count = await asyncio.to_thread(
            get_redis_client().eval,
            RATE_LIMIT_SCRIPT,
            1,
            redis_key,
            retry_after,
        )
    except RedisError as error:
        logger.error("Chat rate limiter unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Chat rate limiting is temporarily unavailable",
        ) from error

    limit = settings.CHAT_RATE_LIMIT_REQUESTS
    remaining = max(0, limit - int(count))
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_at),
    }
    if int(count) > limit:
        headers["Retry-After"] = str(retry_after)
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

    response.headers.update(headers)
