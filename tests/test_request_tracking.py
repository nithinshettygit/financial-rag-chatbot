import asyncio
import hashlib
import re
import unittest
from contextlib import nullcontext
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app, chat_endpoint
from app.schemas import ChatRequest


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

        with (
            patch("app.main.logfire.span", return_value=nullcontext()) as span,
            patch("app.main.run_rag_pipeline_async", new_callable=AsyncMock),
            patch.object(app.state, "ready", True, create=True),
        ):
            asyncio.run(chat_endpoint(request))

        attributes = span.call_args.kwargs
        self.assertEqual(attributes["query_length"], len(query))
        self.assertEqual(attributes["query_hash"], hashlib.sha256(query.encode()).hexdigest())
        self.assertNotIn(query, str(attributes))


if __name__ == "__main__":
    unittest.main()
