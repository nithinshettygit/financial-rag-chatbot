import asyncio
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException
from redis.exceptions import RedisError

from app.config import settings
from app.utils.cache import get_redis_client
from app.utils.logger import logger

QUEUE_ACQUIRE_SCRIPT = """
local queue_key = KEYS[1]
local active_key = KEYS[2]
local member = ARGV[1]
local now_ms = tonumber(ARGV[2])
local wait_ttl_ms = tonumber(ARGV[3])
local lease_ms = tonumber(ARGV[4])
local max_active = tonumber(ARGV[5])
local max_waiters = tonumber(ARGV[6])

-- Remove expired waiting requests and expired active leases.
redis.call('ZREMRANGEBYSCORE', queue_key, '-inf', now_ms - wait_ttl_ms)
redis.call('ZREMRANGEBYSCORE', active_key, '-inf', now_ms)

local existing = redis.call('ZSCORE', queue_key, member)
if not existing then
    if redis.call('ZCARD', queue_key) >= max_waiters then
        return -1
    end
    redis.call('ZADD', queue_key, now_ms, member)
end

local active_count = redis.call('ZCARD', active_key)
local rank = redis.call('ZRANK', queue_key, member)

if active_count < max_active and rank == 0 then
    redis.call('ZREM', queue_key, member)
    redis.call('ZADD', active_key, now_ms + lease_ms, member)
    return 1
end

return 0
"""

RELEASE_SCRIPT = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""


@dataclass(frozen=True)
class QueueLease:
    member: str


class DistributedRAGQueue:
    def __init__(self) -> None:
        self.queue_key = "financial-rag:rag-queue:v1:waiting"
        self.active_key = "financial-rag:rag-queue:v1:active"

    async def acquire(self) -> QueueLease:
        member = uuid.uuid4().hex
        deadline = time.monotonic() + settings.RAG_QUEUE_MAX_WAIT_SECONDS

        while True:
            now_ms = int(time.time() * 1000)
            try:
                result = await asyncio.to_thread(
                    get_redis_client().eval,
                    QUEUE_ACQUIRE_SCRIPT,
                    2,
                    self.queue_key,
                    self.active_key,
                    member,
                    now_ms,
                    settings.RAG_QUEUE_MAX_WAIT_SECONDS * 1000,
                    settings.RAG_CONCURRENCY_LEASE_SECONDS * 1000,
                    settings.MAX_CONCURRENT_RAG,
                    settings.MAX_QUEUE_WAITERS,
                )
            except RedisError as error:
                logger.error("RAG queue unavailable: %s", error)
                raise HTTPException(
                    status_code=503,
                    detail="RAG concurrency control is temporarily unavailable",
                ) from error

            result = int(result)
            if result == 1:
                return QueueLease(member=member)
            if result == -1:
                raise HTTPException(
                    status_code=429,
                    detail="RAG queue is full. Please retry shortly.",
                    headers={"Retry-After": str(settings.RAG_QUEUE_RETRY_AFTER_SECONDS)},
                )

            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=429,
                    detail="RAG queue wait time exceeded. Please retry shortly.",
                    headers={"Retry-After": str(settings.RAG_QUEUE_RETRY_AFTER_SECONDS)},
                )

            await asyncio.sleep(settings.RAG_QUEUE_POLL_INTERVAL_SECONDS)

    async def release(self, lease: QueueLease) -> None:
        try:
            await asyncio.to_thread(
                get_redis_client().eval,
                RELEASE_SCRIPT,
                2,
                self.queue_key,
                self.active_key,
                lease.member,
            )
        except RedisError:
            # The lease has a TTL-like expiry encoded in the active ZSET score,
            # so Redis failure during release cannot permanently consume a slot.
            logger.exception("Failed to release RAG queue lease %s", lease.member)


rag_queue = DistributedRAGQueue()
