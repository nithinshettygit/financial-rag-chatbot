import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from redis.exceptions import RedisError

from app.config import settings
from app.queue import DistributedRAGQueue, QUEUE_ACQUIRE_SCRIPT, RELEASE_SCRIPT, RENEW_SCRIPT


class AtomicQueueRedis:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.waiting: dict[str, int] = {}
        self.active: dict[str, int] = {}
        self.max_active = 0

    def eval(self, script: str, _key_count: int, *args: object) -> int:
        with self.lock:
            if script == QUEUE_ACQUIRE_SCRIPT:
                _waiting_key, _active_key, member, now_ms, wait_ttl_ms, lease_ms, max_active, max_waiters = args
                member = str(member)
                now_ms = int(now_ms)
                self.waiting = {key: score for key, score in self.waiting.items() if score >= now_ms - int(wait_ttl_ms)}
                self.active = {key: score for key, score in self.active.items() if score > now_ms}
                if member not in self.waiting:
                    if len(self.waiting) >= int(max_waiters):
                        return -1
                    self.waiting[member] = now_ms
                ordered = sorted(self.waiting, key=lambda key: (self.waiting[key], key))
                if len(self.active) < int(max_active) and ordered[0] == member:
                    del self.waiting[member]
                    self.active[member] = now_ms + int(lease_ms)
                    self.max_active = max(self.max_active, len(self.active))
                    return 1
                return 0

            if script == RELEASE_SCRIPT:
                _waiting_key, _active_key, member = args
                member = str(member)
                self.waiting.pop(member, None)
                self.active.pop(member, None)
                return 1

            if script == RENEW_SCRIPT:
                _active_key, member, expires_at = args
                member, expires_at = str(member), int(expires_at)
                if member in self.active:
                    self.active[member] = expires_at
                    return 1
                return 0

            raise AssertionError("Unexpected queue script")


class UnavailableRedis:
    def eval(self, *_args: object) -> int:
        raise RedisError("offline")


class QueueStressTests(unittest.TestCase):
    def test_concurrent_requests_never_exceed_configured_slots(self) -> None:
        redis_client = AtomicQueueRedis()
        queue = DistributedRAGQueue()

        async def exercise_queue() -> None:
            async def request() -> None:
                lease = await queue.acquire()
                await asyncio.sleep(0.01)
                await queue.release(lease)

            await asyncio.gather(*(request() for _ in range(10)))

        with (
            patch("app.queue.get_redis_client", return_value=redis_client),
            patch.object(settings, "MAX_CONCURRENT_RAG", 2),
            patch.object(settings, "MAX_QUEUE_WAITERS", 20),
            patch.object(settings, "RAG_QUEUE_MAX_WAIT_SECONDS", 3),
            patch.object(settings, "RAG_QUEUE_POLL_INTERVAL_SECONDS", 0.001),
            patch.object(settings, "RAG_CONCURRENCY_LEASE_SECONDS", 2),
        ):
            asyncio.run(exercise_queue())

        self.assertEqual(redis_client.max_active, 2)
        self.assertFalse(redis_client.active)
        self.assertFalse(redis_client.waiting)

    def test_queue_fails_closed_when_redis_is_unavailable(self) -> None:
        queue = DistributedRAGQueue()
        with patch("app.queue.get_redis_client", return_value=UnavailableRedis()):
            with self.assertRaises(HTTPException) as unavailable:
                asyncio.run(queue.acquire())
        self.assertEqual(unavailable.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
