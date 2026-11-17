import asyncio
import hashlib
import hmac
import ipaddress
import secrets
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from redis.exceptions import RedisError

from app.config import settings
from app.utils.cache import get_redis_client
from app.utils.logger import logger

SESSION_COOKIE = "financial_rag_sid"

RATE_LIMIT_SCRIPT = """
local session_count = tonumber(redis.call('GET', KEYS[1]) or '0')
local ip_count = tonumber(redis.call('GET', KEYS[2]) or '0')
local session_limit = tonumber(ARGV[1])
local ip_limit = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

if session_count >= session_limit or ip_count >= ip_limit then
    return {session_count, ip_count, 0}
end

session_count = redis.call('INCR', KEYS[1])
if session_count == 1 then
    redis.call('EXPIRE', KEYS[1], ttl)
end

ip_count = redis.call('INCR', KEYS[2])
if ip_count == 1 then
    redis.call('EXPIRE', KEYS[2], ttl)
end

return {session_count, ip_count, 1}
"""


def _sign_session_id(session_id: str) -> str:
    digest = hmac.new(
        settings.SESSION_HMAC_SECRET.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def create_session_token() -> str:
    session_id = secrets.token_urlsafe(32)
    return f"{session_id}.{_sign_session_id(session_id)}"


def verify_session_token(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    session_id, signature = token.rsplit(".", 1)
    if not session_id or not signature:
        return None
    expected = _sign_session_id(session_id)
    if not hmac.compare_digest(signature, expected):
        return None
    return session_id


def ensure_session(request: Request, response: Response) -> str:
    existing = verify_session_token(request.cookies.get(SESSION_COOKIE))
    if existing:
        request.state.session_id = existing
        return existing

    token = create_session_token()
    session_id = token.split(".", 1)[0]
    request.state.session_id = session_id
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=settings.SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )
    return session_id


def normalize_client_ip(request: Request) -> str:
    # Nginx is configured to overwrite X-Forwarded-For with the immediate client
    # address. This avoids trusting arbitrary client-supplied forwarding headers.
    raw_ip = request.headers.get("X-Real-IP") or request.client.host
    try:
        ip = ipaddress.ip_address(raw_ip.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid client IP address")

    if ip.version == 6:
        network = ipaddress.ip_network(f"{ip}/64", strict=False)
        return str(network.network_address) + "/64"
    return str(ip)


async def enforce_chat_rate_limit(request: Request, response: Response) -> None:
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid chat session")

    client_ip = normalize_client_ip(request)
    now = int(time.time())
    window = now // settings.CHAT_RATE_LIMIT_WINDOW_SECONDS
    reset_at = (window + 1) * settings.CHAT_RATE_LIMIT_WINDOW_SECONDS
    retry_after = max(1, reset_at - now)

    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    ip_digest = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()
    session_key = f"financial-rag:rate:v2:session:{session_digest}:{window}"
    ip_key = f"financial-rag:rate:v2:ip:{ip_digest}:{window}"

    try:
        result = await asyncio.to_thread(
            get_redis_client().eval,
            RATE_LIMIT_SCRIPT,
            2,
            session_key,
            ip_key,
            settings.CHAT_SESSION_RATE_LIMIT_REQUESTS,
            settings.CHAT_IP_RATE_LIMIT_REQUESTS,
            retry_after,
        )
    except RedisError as error:
        logger.error("Chat rate limiter unavailable: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Chat rate limiting is temporarily unavailable",
        ) from error

    session_count, ip_count, allowed = map(int, result)
    session_remaining = max(0, settings.CHAT_SESSION_RATE_LIMIT_REQUESTS - session_count)
    ip_remaining = max(0, settings.CHAT_IP_RATE_LIMIT_REQUESTS - ip_count)

    response.headers["X-RateLimit-Session-Limit"] = str(settings.CHAT_SESSION_RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Session-Remaining"] = str(session_remaining)
    response.headers["X-RateLimit-IP-Limit"] = str(settings.CHAT_IP_RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-IP-Remaining"] = str(ip_remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_at)

    if not allowed:
        response.headers["Retry-After"] = str(retry_after)
        detail = "Session rate limit exceeded" if session_count >= settings.CHAT_SESSION_RATE_LIMIT_REQUESTS else "IP rate limit exceeded"
        raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after)})


def validate_security_settings() -> None:
    secret = settings.SESSION_HMAC_SECRET.encode("utf-8")
    if len(secret) < 32:
        raise RuntimeError("SESSION_HMAC_SECRET must be at least 32 bytes")

    if settings.SESSION_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
        raise RuntimeError("SESSION_COOKIE_SAMESITE must be lax, strict, or none")
    if settings.SESSION_COOKIE_SAMESITE == "none" and not settings.SESSION_COOKIE_SECURE:
        raise RuntimeError("Secure cookies are required when SameSite=None")
