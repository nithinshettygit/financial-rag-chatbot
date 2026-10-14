import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
import httpx
from openai import AsyncOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from app.pipeline import run_rag_pipeline_async, ChatRequest
from app.config import settings


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("evaluation/dataset/ragas_dataset_cleaned.json")
CACHE_PATH = Path("ragas_samples_cache.json")
REPORT_PATH = Path("ragas_evaluation_report.csv")

JUDGE_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_CONTEXTS = 5
MAX_CONTEXT_CHARS = 1500

MAX_WORKERS = 1
JUDGE_TIMEOUT = 60
MAX_RETRIES = 2

logger = logging.getLogger(__name__)
_RETRY_DURATION_RE = re.compile(
    r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE
)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
_DURATION_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


# ============================================================
# FAILOVER CLIENT BUILDER
# ============================================================

class GroqFailoverTransport(httpx.AsyncBaseTransport):
    """Retry rate-limited or unavailable Groq models using configured keys/models."""

    def __init__(self, api_keys: list[str]):
        self._api_keys = api_keys
        self._transport = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            payload = json.loads(await request.aread())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await self._transport.handle_async_request(request)

        original_model = payload.get("model", JUDGE_MODELS[0])
        candidates = [
            (api_key, model)
            for model in JUDGE_MODELS
            for api_key in self._api_keys
        ]
        candidates = [
            candidate for candidate in candidates
            if candidate != (self._api_keys[0], original_model)
        ]
        candidates.insert(0, (self._api_keys[0], original_model))

        last_response = None
        retry_delay = 0.0
        for index, (api_key, model) in enumerate(candidates):
            attempt_payload = {**payload, "model": model}
            headers = httpx.Headers(request.headers)
            headers["authorization"] = f"Bearer {api_key}"
            headers.pop("content-length", None)

            attempt = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=json.dumps(attempt_payload).encode("utf-8"),
                extensions=request.extensions.copy(),
            )
            response = await self._transport.handle_async_request(attempt)
            response_body = await response.aread()
            status_code = response.status_code
            response_headers = httpx.Headers(response.headers)
            extensions = response.extensions.copy()
            await response.aclose()

            if not self._should_fail_over(status_code, response_body):
                return self._make_response(
                    status_code, response_headers, response_body, attempt, extensions
                )

            if status_code == 429:
                retry_delay = max(
                    retry_delay,
                    self._get_retry_delay(response_headers, response_body),
                )

            last_response = self._make_response(
                status_code, response_headers, response_body, attempt, extensions
            )
            if index < len(candidates) - 1:
                logger.warning(
                    "Groq judge returned HTTP %d for %s; trying the next configured key/model",
                    status_code,
                    model,
                )

        if last_response is not None and last_response.status_code == 429 and retry_delay:
            logger.warning(
                "All configured Groq key/model options are rate limited; sleeping %.2fs",
                retry_delay,
            )
            await asyncio.sleep(retry_delay)

        return last_response

    @staticmethod
    def _get_retry_delay(headers: httpx.Headers, body: bytes) -> float:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass

        try:
            message = str(json.loads(body).get("error", {}).get("message", ""))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            return 0.0

        match = _RETRY_DURATION_RE.search(message)
        if not match:
            return 0.0
        return sum(
            float(value) * _DURATION_SECONDS[unit.lower()]
            for value, unit in _DURATION_PART_RE.findall(match.group(1))
        )

    @staticmethod
    def _should_fail_over(status_code: int, body: bytes) -> bool:
        if status_code == 429:
            return True
        if status_code not in (400, 404):
            return False
        try:
            error = json.loads(body).get("error", {})
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return error.get("code") in {"model_not_found", "model_decommissioned"}

    @staticmethod
    def _make_response(
        status_code: int,
        headers: httpx.Headers,
        body: bytes,
        request: httpx.Request,
        extensions: dict[str, Any],
    ) -> httpx.Response:
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)
        headers.pop("transfer-encoding", None)
        return httpx.Response(
            status_code,
            headers=headers,
            content=body,
            request=request,
            extensions=extensions,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


def build_judge_llm():
    primary_key = (settings.GROQ_API_KEY or "").strip()
    fallback_key = (settings.FALLBACK_GROQ_API_KEY or "").strip()

    if not primary_key:
        raise ValueError("GROQ_API_KEY is missing.")

    api_keys = list(dict.fromkeys(key for key in (primary_key, fallback_key) if key))
    transport = GroqFailoverTransport(api_keys)
    http_client = httpx.AsyncClient(timeout=JUDGE_TIMEOUT, transport=transport)
    groq_client = AsyncOpenAI(
        api_key=primary_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=JUDGE_TIMEOUT,
        max_retries=0,
        http_client=http_client,
    )

    return llm_factory(
        model=JUDGE_MODELS[0],
        provider="openai",
        client=groq_client,
    )


def get_metrics(judge_llm, embeddings):
    """Configure Ragas evaluate()-compatible metric instances."""
    answer_relevancy.strictness = 1
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
    return metrics


# ============================================================
# JSON AND SAMPLE VALIDATION
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_samples(samples: Any) -> list[dict]:
    if not isinstance(samples, list):
        raise ValueError("Samples must be a list.")

    required = {
        "id", "user_input", "retrieved_contexts",
        "response", "reference",
    }

    valid = []

    for sample in samples:
        if not isinstance(sample, dict) or not required.issubset(sample):
            continue

        if not all(
            isinstance(sample[key], str) and sample[key].strip()
            for key in ("user_input", "response", "reference")
        ):
            continue

        contexts = sample["retrieved_contexts"]

        if not isinstance(contexts, list):
            continue

        contexts = [
            context for context in contexts
            if isinstance(context, str) and context.strip()
        ]

        if not contexts:
            continue

        valid.append({
            **sample,
            "retrieved_contexts": contexts[:MAX_CONTEXTS],
        })

    return valid


# ============================================================
# STEP 1: RUN RAG PIPELINE
# ============================================================

async def collect_samples(items: list[dict]) -> tuple[list[dict], list[dict]]:
    samples = []
    problems = []
    total = len(items)

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            problems.append({
                "id": f"item_{index}",
                "issue": "Invalid dataset record",
            })
            continue

        item_id = item.get("id", f"item_{index}")

        if not item.get("answerable", True):
            print(f"[{index}/{total}] Skipping unanswerable {item_id}", flush=True)
            continue

        query = item.get("user_input")
        reference = item.get("reference")

        if not isinstance(query, str) or not query.strip():
            problems.append({"id": item_id, "issue": "Missing user_input"})
            continue

        if not isinstance(reference, str) or not reference.strip():
            problems.append({"id": item_id, "issue": "Missing reference"})
            continue

        try:
            request = ChatRequest(
                query=query,
                session_id=f"eval_{item_id}",
            )

            output = await run_rag_pipeline_async(request)
            answer = getattr(output, "answer", None)
            retrieved = getattr(output, "retrieved_chunks", None) or []

            contexts = [
                chunk.content[:MAX_CONTEXT_CHARS]
                for chunk in retrieved[:MAX_CONTEXTS]
                if isinstance(getattr(chunk, "content", None), str)
                and chunk.content.strip()
            ]

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Pipeline returned an empty answer.")

            if not contexts:
                raise ValueError("No usable contexts retrieved.")

            samples.append({
                "id": item_id,
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": reference,
            })

            print(f"[{index}/{total}] Processed {item_id}", flush=True)

        except Exception as exc:
            issue = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{total}] Failed {item_id}: {issue}", flush=True)
            problems.append({"id": item_id, "issue": issue})

    return samples, problems


# ============================================================
# STEP 2: LOAD OR CACHE PIPELINE OUTPUTS
# ============================================================

async def get_samples(refresh: bool) -> tuple[list[dict], list[dict]]:
    if CACHE_PATH.exists() and not refresh:
        print(f"Loading cache: {CACHE_PATH}", flush=True)

        cache = load_json(CACHE_PATH)

        if not isinstance(cache, dict):
            raise ValueError("Invalid cache structure.")

        samples = validate_samples(cache.get("samples", []))
        problems = cache.get("problems", [])

        if not samples:
            raise RuntimeError("No valid cached samples. Run with --refresh.")

        print(f"Loaded {len(samples)} cached samples.", flush=True)
        return samples, problems

    items = load_json(DATASET_PATH)

    if not isinstance(items, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    print(f"Running pipeline for {len(items)} records...", flush=True)
    samples, problems = await collect_samples(items)

    if not samples:
        raise RuntimeError("No pipeline samples succeeded.")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {"samples": samples, "problems": problems},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved cache: {CACHE_PATH}", flush=True)
    return samples, problems


# ============================================================
# STEP 3: RAGAS EVALUATION
# ============================================================

def evaluate_samples(samples: list[dict]):
    judge_llm = build_judge_llm()

    print("Loading Hugging Face embedding model...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    metrics = get_metrics(judge_llm, embeddings)

    ids = [sample["id"] for sample in samples]

    dataset = Dataset.from_list([
        {
            "user_input": sample["user_input"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "response": sample["response"],
            "reference": sample["reference"],
        }
        for sample in samples
    ])

    print(
        f"Starting evaluation: {len(dataset)} samples, "
        f"{len(metrics)} metrics, {MAX_WORKERS} worker(s).",
        flush=True,
    )

    run_config = RunConfig(
        max_workers=MAX_WORKERS,
        timeout=JUDGE_TIMEOUT,
        max_retries=MAX_RETRIES,
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=True,
        show_progress=True,
    )

    print("\n=== Ragas Results ===", flush=True)
    print(results, flush=True)

    df = results.to_pandas()
    df.insert(0, "id", ids)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print(f"Report saved: {REPORT_PATH.resolve()}", flush=True)

    for metric_name in [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]:
        if metric_name in df.columns:
            print(f"{metric_name}: {df[metric_name].mean():.4f}", flush=True)

    return results


# ============================================================
# MAIN
# ============================================================

async def main():
    refresh = "--refresh" in sys.argv

    samples, problems = await get_samples(refresh)

    # Keep synchronous Ragas evaluation outside the active event loop.
    await asyncio.to_thread(evaluate_samples, samples)

    if problems:
        print(f"\nExcluded/failed pipeline records: {len(problems)}", flush=True)
        for problem in problems:
            print(f"  {problem['id']}: {problem['issue']}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(
            f"\nEvaluation failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise