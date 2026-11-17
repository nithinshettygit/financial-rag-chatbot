import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException, Request, Response
from redis.exceptions import RedisError

from app.config import settings
from app.security import (
    create_session_token,
    enforce_chat_rate_limit,
    validate_security_settings,
    verify_session_token,
)


class FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def eval(self, _script: str, _key_count: int, session_key: str, ip_key: str,
             session_limit: int, ip_limit: int, _ttl: int) -> list[int]:
        with self.lock:
            session_count = self.counts.get(session_key, 0)
            ip_count = self.counts.get(ip_key, 0)
            if session_count >= session_limit or ip_count >= ip_limit:
                return [session_count, ip_count, 0]
            session_count += 1
            ip_count += 1
            self.counts[session_key] = session_count
            self.counts[ip_key] = ip_count
            return [session_count, ip_count, 1]


class BrokenRedis:
    def eval(self, *_args: object) -> int:
        raise RedisError("offline")


def make_request(session_id: str = "session-one", ip: str = "203.0.113.8") -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/v1/chat",
        "headers": [(b"x-real-ip", ip.encode())],
        "client": (ip, 1234),
        "state": {"session_id": session_id},
    })


class SecurityTests(unittest.TestCase):
    def test_signed_session_token_rejects_tampering(self) -> None:
        with patch.object(settings, "SESSION_HMAC_SECRET", "s" * 32):
            token = create_session_token()
            session_id = token.split(".", 1)[0]
            self.assertEqual(verify_session_token(token), session_id)
            self.assertIsNone(verify_session_token(token + "tampered"))
            self.assertIsNone(verify_session_token(None))

    def test_chat_rate_limit_blocks_at_the_session_limit(self) -> None:
        redis_client = FakeRedis()
        with (
            patch("app.security.get_redis_client", return_value=redis_client),
            patch.object(settings, "CHAT_SESSION_RATE_LIMIT_REQUESTS", 2),
            patch.object(settings, "CHAT_IP_RATE_LIMIT_REQUESTS", 5),
            patch.object(settings, "CHAT_RATE_LIMIT_WINDOW_SECONDS", 60),
            patch("app.security.time.time", return_value=1_700_000_000),
        ):
            first_response, second_response = Response(), Response()
            asyncio.run(enforce_chat_rate_limit(make_request(), first_response))
            asyncio.run(enforce_chat_rate_limit(make_request(), second_response))
            self.assertEqual(second_response.headers["X-RateLimit-Session-Remaining"], "0")

            with self.assertRaises(HTTPException) as limited:
                asyncio.run(enforce_chat_rate_limit(make_request(), Response()))
            self.assertEqual(limited.exception.status_code, 429)
            self.assertIn("Retry-After", limited.exception.headers)

    def test_redis_rate_limit_failure_fails_closed(self) -> None:
        with patch("app.security.get_redis_client", return_value=BrokenRedis()):
            with self.assertRaises(HTTPException) as unavailable:
                asyncio.run(enforce_chat_rate_limit(make_request(), Response()))
        self.assertEqual(unavailable.exception.status_code, 503)

    def test_security_and_credentials_are_validated_without_provider_calls(self) -> None:
        with (
            patch.object(settings, "GROQ_API_KEY", "gsk_" + "a" * 24),
            patch.object(settings, "FALLBACK_GROQ_API_KEY", "gsk_" + "b" * 24),
            patch.object(settings, "QDRANT_URL", None),
            patch.object(settings, "QDRANT_API_KEY", None),
            patch.object(settings, "SESSION_HMAC_SECRET", "s" * 32),
            patch.object(settings, "SESSION_COOKIE_SAMESITE", "lax"),
            patch.object(settings, "SESSION_COOKIE_SECURE", True),
        ):
            validate_security_settings()

    def test_placeholder_credentials_are_rejected_locally(self) -> None:
        with (
            patch.object(settings, "GROQ_API_KEY", "replace-with-a-valid-groq-api-key"),
            patch.object(settings, "FALLBACK_GROQ_API_KEY", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                validate_security_settings()

if __name__ == "__main__":
    unittest.main()
