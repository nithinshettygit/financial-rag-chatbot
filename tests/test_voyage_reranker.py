import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config import settings
from app.retrieval.voyage_reranker import VoyageReranker


class VoyageRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "chunk_id": "chunk-1",
                "entity": "Duolingo",
                "source_file": "duolingo-10k.pdf",
                "headings": ["Revenue", "Subscriptions"],
                "chunk_type": "table_fact",
                "metadata": {
                    "financial_metric": "Revenue",
                    "fiscal_years": [2024],
                    "columns": ["Subscriptions", "Advertising"],
                },
                "content": "Subscriptions were the largest source.",
                "rrf_score": 0.02,
            },
            {
                "chunk_id": "chunk-2",
                "content": "Advertising revenue details.",
                "rrf_score": 0.01,
            },
        ]

    @patch("app.retrieval.voyage_reranker.voyageai.Client")
    def test_rerank_maps_voyage_indices_to_original_candidates(self, client_type: Mock) -> None:
        client = client_type.return_value
        client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=1, relevance_score=0.91)]
        )
        with patch.object(settings, "VOYAGE_API_KEY", "voyage-test-key"):
            reranker = VoyageReranker()

        result = reranker.rerank("revenue sources", self.candidates, top_k=1)

        self.assertEqual([item["chunk_id"] for item in result], ["chunk-2"])
        self.assertEqual(result[0]["final_score"], 0.91)
        self.assertEqual(result[0]["rerank_score"], 0.91)
        kwargs = client.rerank.call_args.kwargs
        self.assertEqual(kwargs["query"], "revenue sources")
        self.assertEqual(kwargs["model"], settings.RERANKER_MODEL)
        self.assertEqual(kwargs["top_k"], 1)
        self.assertIn("Financial Metric: Revenue", kwargs["documents"][0])
        self.assertIn("Fiscal Years: 2024", kwargs["documents"][0])

    @patch("app.retrieval.voyage_reranker.voyageai.Client")
    def test_api_failure_falls_back_to_initial_candidate_order(self, client_type: Mock) -> None:
        client_type.return_value.rerank.side_effect = TimeoutError("request timed out")
        with patch.object(settings, "VOYAGE_API_KEY", "voyage-test-key"):
            reranker = VoyageReranker()

        result = reranker.rerank("revenue sources", self.candidates, top_k=1)

        self.assertEqual([item["chunk_id"] for item in result], ["chunk-1"])
        self.assertEqual(result[0]["final_score"], self.candidates[0]["rrf_score"])

    def test_missing_api_key_fails_explicitly(self) -> None:
        with patch.object(settings, "VOYAGE_API_KEY", None):
            with self.assertRaisesRegex(RuntimeError, "VOYAGE_API_KEY"):
                VoyageReranker()
