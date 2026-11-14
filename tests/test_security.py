import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from app.config import settings
from app.security import (
    authenticate_api_key,
    enforce_chat_rate_limit,
    validate_security_settings,
)


class FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def eval(self, _script: str, _key_count: int, key: str, _ttl: int) -> int:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]


class BrokenRedis:
    def eval(self, *_args: object) -> int:
        raise RedisError("offline")


class SecurityTests(unittest.TestCase):
    def test_chat_route_authenticates_and_limits_each_api_key(self) -> None:
        app = FastAPI()

        @app.post("/v1/chat", dependencies=[Depends(enforce_chat_rate_limit)])
        async def chat() -> dict[str, bool]:
            return {"ok": True}

        key_one = "a" * 32
        key_two = "b" * 32
        redis_client = FakeRedis()
        with (
            patch.object(settings, "CHAT_API_KEYS", {"web": key_one, "integration": key_two}),
            patch.object(settings, "CHAT_RATE_LIMIT_REQUESTS", 1),
            patch.object(settings, "CHAT_RATE_LIMIT_WINDOW_SECONDS", 60),
            patch("app.security.get_redis_client", return_value=redis_client),
            patch("app.security.time", SimpleNamespace(time=lambda: 1_700_000_000)),
        ):
            client = TestClient(app)
            self.assertEqual(client.post("/v1/chat").status_code, 401)
            self.assertEqual(
                client.post("/v1/chat", headers={"X-API-Key": key_one}).status_code,
                200,
            )
            self.assertEqual(
                client.post("/v1/chat", headers={"X-API-Key": key_two}).status_code,
                200,
            )
            limited = client.post("/v1/chat", headers={"X-API-Key": key_one})
            self.assertEqual(limited.status_code, 429)
            self.assertIn("Retry-After", limited.headers)

    def test_startup_rejects_missing_api_key_configuration(self) -> None:
        with patch.object(settings, "CHAT_API_KEYS", {}):
            with self.assertRaises(RuntimeError):
                validate_security_settings()
        with patch.object(settings, "CHAT_API_KEYS", {"web": "too-short"}):
            with self.assertRaises(RuntimeError):
                validate_security_settings()

    def test_api_keys_are_authenticated_without_returning_the_secret(self) -> None:
        key = "a" * 32

        digest = authenticate_api_key(key, {"web": key})

        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, key)

    def test_missing_or_invalid_api_keys_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as missing:
            authenticate_api_key(None, {"web": "a" * 32})
        self.assertEqual(missing.exception.status_code, 401)

        with self.assertRaises(HTTPException) as invalid:
            authenticate_api_key("wrong", {"web": "a" * 32})
        self.assertEqual(invalid.exception.status_code, 401)

    def test_rate_limit_blocks_requests_after_the_configured_limit(self) -> None:
        redis_client = FakeRedis()
        with (
            patch("app.security.get_redis_client", return_value=redis_client),
            patch.object(settings, "CHAT_RATE_LIMIT_REQUESTS", 1),
            patch.object(settings, "CHAT_RATE_LIMIT_WINDOW_SECONDS", 60),
            patch("app.security.time", SimpleNamespace(time=lambda: 1_700_000_000)),
        ):
            first_response = Response()
            asyncio.run(enforce_chat_rate_limit(first_response, "b" * 64))
            self.assertEqual(first_response.headers["X-RateLimit-Remaining"], "0")

            with self.assertRaises(HTTPException) as limited:
                asyncio.run(enforce_chat_rate_limit(Response(), "b" * 64))
            self.assertEqual(limited.exception.status_code, 429)
            self.assertIn("Retry-After", limited.exception.headers)

    def test_redis_failure_fails_closed(self) -> None:
        with patch("app.security.get_redis_client", return_value=BrokenRedis()):
            with self.assertRaises(HTTPException) as unavailable:
                asyncio.run(enforce_chat_rate_limit(Response(), "b" * 64))
        self.assertEqual(unavailable.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
