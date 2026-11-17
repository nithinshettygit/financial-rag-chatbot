import asyncio
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError
from qdrant_client.http import models

from app.schemas import ChatRequest, ChatTurn
from app.services.retrieval import _safe_call
from app.utils.cache import build_scope
from app.utils.guardrails import _check_rails, check_input, check_output
from app.utils.llm import chat_with_retry, extract_completion_text
from app.utils.router import validate_router_response
from app.utils.vector import QdrantVectorRetriever, VectorStoreError


class ResilienceTests(unittest.TestCase):
    def test_request_rejects_blank_queries_and_oversized_history(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(query="   ")

        with self.assertRaises(ValidationError):
            ChatRequest(query="revenue", history=[ChatTurn(role="user", content="x")] * 13)

        with self.assertRaises(ValidationError):
            ChatTurn(role="assistant", content="   ")

        with self.assertRaises(ValidationError):
            ChatRequest(query="revenue", state={"company": "A" * 201})

    def test_cache_scope_isolated_by_document_id(self) -> None:
        first = build_scope("rag_search", ["Acme"], [2025], "corpus", document_ids=["filing-a"])
        second = build_scope("rag_search", ["Acme"], [2025], "corpus", document_ids=["filing-b"])

        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_qdrant_failure_is_not_reported_as_empty_results(self) -> None:
        retriever = object.__new__(QdrantVectorRetriever)
        retriever.client = SimpleNamespace(collection_exists=Mock(side_effect=RuntimeError("offline")))

        with self.assertRaises(VectorStoreError):
            retriever.list_documents()

    def test_qdrant_filter_indexes_are_created_once(self) -> None:
        create_index = Mock()
        retriever = object.__new__(QdrantVectorRetriever)
        retriever.client = SimpleNamespace(create_payload_index=create_index)
        retriever._payload_indexes_ready = False
        retriever._payload_index_lock = threading.Lock()

        retriever._ensure_payload_indexes()
        retriever._ensure_payload_indexes()

        self.assertEqual(create_index.call_count, 4)
        self.assertEqual(
            {call.kwargs["field_name"] for call in create_index.call_args_list},
            {"document_id", "company", "fiscal_year", "fiscal_years"},
        )
        fiscal_years_index = next(
            call for call in create_index.call_args_list
            if call.kwargs["field_name"] == "fiscal_years"
        )
        self.assertEqual(
            fiscal_years_index.kwargs["field_schema"],
            models.PayloadSchemaType.INTEGER,
        )

    def test_qdrant_filter_index_failure_is_explicit(self) -> None:
        retriever = object.__new__(QdrantVectorRetriever)
        retriever.client = SimpleNamespace(
            create_payload_index=Mock(side_effect=RuntimeError("index unavailable"))
        )
        retriever._payload_indexes_ready = False
        retriever._payload_index_lock = threading.Lock()

        with self.assertRaises(VectorStoreError):
            retriever._ensure_payload_indexes()

    def test_retrieval_propagates_qdrant_failure(self) -> None:
        def unavailable() -> list[dict[str, str]]:
            raise VectorStoreError("offline")

        with self.assertRaises(VectorStoreError):
            asyncio.run(_safe_call(unavailable))

    def test_groq_completion_requires_nonempty_assistant_text(self) -> None:
        valid = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  answer  "))]
        )
        self.assertEqual(extract_completion_text(valid), "answer")

        for response in (
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                extract_completion_text(response)

    def test_groq_authentication_failure_uses_configured_fallback(self) -> None:
        class FakeAuthenticationError(Exception):
            pass

        primary_create = Mock(side_effect=FakeAuthenticationError())
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"))]
        )
        fallback_create = Mock(return_value=fallback_response)
        clients = [
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=primary_create))),
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fallback_create))),
        ]
        with (
            patch("app.utils.llm.AuthenticationError", FakeAuthenticationError),
            patch.dict(os.environ, {"LLM_MAX_ATTEMPTS": "2", "LLM_MAX_WAIT_S": "0"}),
        ):
            result = chat_with_retry(clients, model="test-model", messages=[])

        self.assertIs(result, fallback_response)
        primary_create.assert_called_once()
        fallback_create.assert_called_once()

    def test_empty_groq_completion_retries_with_fallback_key(self) -> None:
        empty_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        primary_create = Mock(return_value=empty_response)
        fallback_create = Mock(return_value=fallback_response)
        clients = [
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=primary_create))),
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fallback_create))),
        ]
        with patch.dict(os.environ, {"LLM_MAX_ATTEMPTS": "2", "LLM_MAX_WAIT_S": "0"}):
            result = chat_with_retry(clients, model="test-model", messages=[])

        self.assertIs(result, fallback_response)
        primary_create.assert_called_once()
        fallback_create.assert_called_once()

    def test_router_response_shape_is_checked_before_planning(self) -> None:
        valid = {"intent": "rag_search", "entities": ["Acme"], "fiscal_years": [2025], "state": {}}
        self.assertEqual(validate_router_response(valid), valid)
        self.assertEqual(validate_router_response({"entities": "Acme"}), {})
        self.assertEqual(validate_router_response({"state": {"company": []}}), {})

    def test_guardrails_fail_closed_on_service_errors(self) -> None:
        with (
            patch("app.utils.guardrails.settings.GUARDRAILS_ENABLED", True),
            patch("app.utils.guardrails.settings.GUARDRAILS_FAIL_OPEN", False),
            patch("app.utils.guardrails.settings.FALLBACK_GROQ_API_KEY", None),
            patch("app.utils.guardrails.get_rails", side_effect=RuntimeError("offline")),
            patch("app.utils.guardrails._self_check_input_prompt", return_value="check {{ user_input }}"),
            patch("app.utils.guardrails.count_tokens", return_value=1),
        ):
            input_result = asyncio.run(check_input("revenue"))
            output_result = asyncio.run(check_output("revenue", "answer", "evidence"))

        self.assertFalse(input_result.allowed)
        self.assertEqual(input_result.rail, "guardrail_error")
        self.assertFalse(output_result.allowed)
        self.assertEqual(output_result.rail, "guardrail_error")

    def test_guardrails_retry_with_fallback_key_but_keep_policy_checks(self) -> None:
        primary = SimpleNamespace(check_async=AsyncMock(side_effect=asyncio.TimeoutError()))
        fallback_result = object()
        fallback = SimpleNamespace(check_async=AsyncMock(return_value=fallback_result))
        with (
            patch("app.utils.guardrails.settings.GROQ_API_KEY", "gsk_" + "a" * 24),
            patch("app.utils.guardrails.settings.FALLBACK_GROQ_API_KEY", "gsk_" + "b" * 24),
            patch("app.utils.guardrails.settings.GUARDRAILS_TIMEOUT_S", 0.1),
            patch("app.utils.guardrails.get_rails", return_value=primary),
            patch("app.utils.guardrails._get_fallback_rails", return_value=fallback),
        ):
            result = asyncio.run(_check_rails([], ["input"]))

        self.assertIs(result, fallback_result)
        primary.check_async.assert_awaited_once()
        fallback.check_async.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
