import asyncio
import hashlib
import re
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app, chat_endpoint
from app.queue import QueueLease
from app.schemas import ChatRequest, ChatResponse, SessionState


class RequestTrackingTests(unittest.TestCase):
    def test_request_id_is_returned_and_logs_omit_query_string(self) -> None:
        with patch("app.main.logger.info") as log_info:
            response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        request_id = response.headers["X-Request-ID"]
        self.assertRegex(request_id, re.compile(r"^[0-9a-f]{32}$"))

        request_log = next(
            call for call in log_info.call_args_list if call.args == ("HTTP request",)
        )
        metadata = request_log.kwargs["extra"]["extra_data"]
        self.assertEqual(metadata["request_id"], request_id)
        self.assertEqual(metadata["method"], "GET")
        self.assertEqual(metadata["path"], "/health")
        self.assertEqual(metadata["status_code"], 200)
        self.assertGreaterEqual(metadata["duration_ms"], 0)
        self.assertEqual(
            set(metadata),
            {"request_id", "method", "path", "status_code", "duration_ms"},
        )
        self.assertNotIn("?", str(metadata))

    def test_chat_span_records_query_digest_instead_of_query_text(self) -> None:
        query = "quarterly revenue"
        request = ChatRequest(session_id="test-session", query=query)
        api_request = SimpleNamespace(state=SimpleNamespace(session_id="test-session"))
        lease = QueueLease(member="lease-id")
        result = ChatResponse(
            session_id="test-session",
            answer="answer",
            intent="rag_search",
            search_query=query,
            state=SessionState(),
            retrieved_chunks=[],
        )

        with (
            patch("app.main.logfire.span", return_value=nullcontext()) as span,
            patch("app.main.run_rag_pipeline_async", new_callable=AsyncMock, return_value=result),
            patch("app.main.rag_queue.acquire", new_callable=AsyncMock, return_value=lease),
            patch("app.main.rag_queue.renew", new_callable=AsyncMock),
            patch("app.main.rag_queue.release", new_callable=AsyncMock),
            patch.object(app.state, "ready", True, create=True),
        ):
            asyncio.run(chat_endpoint(api_request, request))

        attributes = span.call_args.kwargs
        self.assertEqual(attributes["query_length"], len(query))
        self.assertEqual(attributes["query_hash"], hashlib.sha256(query.encode()).hexdigest())
        self.assertNotIn(query, str(attributes))

    def test_timed_out_pipeline_keeps_lease_until_work_finishes(self) -> None:
        async def scenario() -> None:
            finished = asyncio.Event()
            released = asyncio.Event()
            lease = QueueLease(member="lease-id")

            async def slow_pipeline(_request):
                await finished.wait()
                return ChatResponse(
                    session_id="test-session",
                    answer="answer",
                    intent="rag_search",
                    search_query="revenue",
                    state=SessionState(),
                    retrieved_chunks=[],
                )

            async def release(_lease):
                released.set()

            api_request = SimpleNamespace(state=SimpleNamespace(session_id="test-session"))
            request = ChatRequest(session_id="test-session", query="revenue")
            with (
                patch("app.main.logfire.span", return_value=nullcontext()),
                patch("app.main.run_rag_pipeline_async", side_effect=slow_pipeline),
                patch("app.main.rag_queue.acquire", new_callable=AsyncMock, return_value=lease),
                patch("app.main.rag_queue.renew", new_callable=AsyncMock),
                patch("app.main.rag_queue.release", side_effect=release),
                patch("app.main.REQUEST_TIMEOUT_S", 0.01),
                patch.object(app.state, "ready", True, create=True),
            ):
                with self.assertRaises(HTTPException) as timed_out:
                    await chat_endpoint(api_request, request)
                self.assertEqual(timed_out.exception.status_code, 504)
                self.assertFalse(released.is_set())
                finished.set()
                await asyncio.wait_for(released.wait(), timeout=1)

        asyncio.run(scenario())

    def test_request_deadline_includes_queue_wait(self) -> None:
        async def scenario() -> None:
            async def slow_acquire():
                await asyncio.sleep(0.1)
                return QueueLease(member="lease-id")

            api_request = SimpleNamespace(state=SimpleNamespace(session_id="test-session"))
            request = ChatRequest(session_id="test-session", query="revenue")
            with (
                patch("app.main.rag_queue.acquire", side_effect=slow_acquire),
                patch("app.main.run_rag_pipeline_async", new_callable=AsyncMock) as pipeline,
                patch("app.main.logfire.span", return_value=nullcontext()),
                patch("app.main.REQUEST_TIMEOUT_S", 0.01),
                patch.object(app.state, "ready", True, create=True),
            ):
                with self.assertRaises(HTTPException) as timed_out:
                    await chat_endpoint(api_request, request)

            self.assertEqual(timed_out.exception.status_code, 504)
            pipeline.assert_not_called()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
