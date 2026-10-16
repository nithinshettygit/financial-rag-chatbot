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
# CONFIGURATION & LOGGING SETUP
# ============================================================
DATASET_PATH = Path("evaluation/dataset/ragas_dataset_cleaned_all.json")
CACHE_PATH = Path("evaluation/cache/ragas_samples_cache.json")
REPORT_PATH = Path("evaluation/report/ragas_evaluation_report.csv")
# Updated with active, standard Groq judge models
JUDGE_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_CONTEXTS = 5
MAX_CONTEXT_CHARS = 1500
MAX_WORKERS = 1  # Strict single worker to avoid Groq TPM limits
MAX_CONCURRENT_PIPELINE = 2
JUDGE_TIMEOUT = 60
MAX_RETRIES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ragas_eval")

_DURATION_REGEX = re.compile(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE)
_DURATION_MAP = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

# ============================================================
# STREAMLINED FAILOVER & RETRY TRANSPORT
# ============================================================
class GroqFailoverTransport(httpx.AsyncBaseTransport):
    """Handles Groq model failovers and rate-limit backoff sleeps with proper re-attempts."""

    def __init__(self, api_keys: list[str]):
        self._api_keys = api_keys
        self._transport = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            payload = json.loads(await request.aread())
        except Exception:
            return await self._transport.handle_async_request(request)

        orig_model = payload.get("model", JUDGE_MODELS[0])
        candidates = [(k, m) for m in JUDGE_MODELS for k in self._api_keys]
        candidates = [(self._api_keys[0], orig_model)] + [c for c in candidates if c != (self._api_keys[0], orig_model)]

        max_outer_attempts = 3
        for outer_attempt in range(max_outer_attempts):
            last_resp, retry_delay = None, 0.0
            for i, (api_key, model) in enumerate(candidates):
                headers = httpx.Headers(request.headers)
                headers["authorization"] = f"Bearer {api_key}"
                headers.pop("content-length", None)

                attempt = httpx.Request(
                    request.method, request.url, headers=headers,
                    content=json.dumps({**payload, "model": model}).encode("utf-8"),
                    extensions=request.extensions.copy(),
                )
                try:
                    resp = await self._transport.handle_async_request(attempt)
                    body = await resp.aread()
                    status, resp_headers, extensions = resp.status_code, httpx.Headers(resp.headers), resp.extensions.copy()
                    await resp.aclose()
                except httpx.RequestError as exc:
                    logger.warning("Transport request error with model %s: %s", model, exc)
                    continue

                if status != 429 and not self._is_model_error(status, body):
                    return self._build_response(status, resp_headers, body, attempt, extensions)

                if status == 429:
                    retry_delay = max(retry_delay, self._parse_retry_delay(resp_headers, body))

                last_resp = self._build_response(status, resp_headers, body, attempt, extensions)
                if i < len(candidates) - 1:
                    logger.warning("Groq judge failed with HTTP %d for model %s. Failing over...", status, model)

            # If all candidates hit 429, sleep for the required delay and retry the outer loop
            if last_resp and last_resp.status_code == 429:
                sleep_time = retry_delay if retry_delay > 0 else 5.0 * (outer_attempt + 1)
                logger.warning("All Groq key/model options rate-limited (Attempt %d/%d); sleeping for %.2fs", outer_attempt + 1, max_outer_attempts, sleep_time)
                await asyncio.sleep(sleep_time)
                continue

            return last_resp

        return last_resp or httpx.Response(500, json={"error": {"message": "All failover and retry attempts exhausted"}}, request=request)

    @staticmethod
    def _parse_retry_delay(headers: httpx.Headers, body: bytes) -> float:
        if retry_after := headers.get("retry-after"):
            try: return max(0.0, float(retry_after))
            except ValueError: pass
        try:
            msg = str(json.loads(body).get("error", {}).get("message", ""))
            match = _DURATION_REGEX.search(msg)
            if match:
                return sum(float(v) * _DURATION_MAP[u.lower()] for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1), re.IGNORECASE))
        except Exception:
            pass
        return 5.0

    @staticmethod
    def _is_model_error(status: int, body: bytes) -> bool:
        if status not in (400, 404): return False
        try:
            return json.loads(body).get("error", {}).get("code") in {"model_not_found", "model_decommissioned"}
        except Exception:
            return False

    @staticmethod
    def _build_response(status, headers, body, request, extensions) -> httpx.Response:
        for h in ("content-encoding", "content-length", "transfer-encoding"):
            headers.pop(h, None)
        return httpx.Response(status, headers=headers, content=body, request=request, extensions=extensions)

    async def aclose(self):
        await self._transport.aclose()


def build_judge_llm():
    primary = (settings.GROQ_API_KEY or "").strip()
    fallback = (settings.FALLBACK_GROQ_API_KEY or "").strip()
    if not primary:
        raise ValueError("GROQ_API_KEY is missing in settings.")

    api_keys = list(dict.fromkeys(k for k in (primary, fallback) if k))
    client = AsyncOpenAI(
        api_key=primary,
        base_url="https://api.groq.com/openai/v1",
        timeout=JUDGE_TIMEOUT,
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=JUDGE_TIMEOUT, transport=GroqFailoverTransport(api_keys)),
    )
    return llm_factory(model=JUDGE_MODELS[0], provider="openai", client=client)


# ============================================================
# DATA VALIDATION & PIPELINE COLLECTION
# ============================================================
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Dataset or cache file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON from %s: %s", path, exc)
        raise


def validate_samples(samples: list) -> list[dict]:
    valid = []
    for s in samples:
        if not isinstance(s, dict) or not {"id", "user_input", "retrieved_contexts", "response", "reference"}.issubset(s):
            continue
        if not all(isinstance(s[k], str) and s[k].strip() for k in ("user_input", "response", "reference")):
            continue
        contexts = [c for c in s.get("retrieved_contexts", []) if isinstance(c, str) and c.strip()]
        if not contexts:
            continue
        valid.append({**s, "retrieved_contexts": contexts[:MAX_CONTEXTS]})
    return valid


async def process_single_item(item: dict, index: int, total: int, semaphore: asyncio.Semaphore) -> tuple[dict | None, dict | None]:
    if not isinstance(item, dict) or not item.get("answerable", True):
        return None, None

    item_id = item.get("id", f"item_{index}")
    query, reference = item.get("user_input"), item.get("reference")

    if not query or not reference:
        return None, {"id": item_id, "issue": "Missing query or reference"}

    async with semaphore:
        try:
            output = await run_rag_pipeline_async(ChatRequest(query=query, session_id=f"eval_{item_id}"))
            answer = getattr(output, "answer", None)
            retrieved = getattr(output, "retrieved_chunks", None) or []
            contexts = [c.content[:MAX_CONTEXT_CHARS] for c in retrieved[:MAX_CONTEXTS] if getattr(c, "content", "").strip()]

            if not answer or not contexts:
                raise ValueError("Empty pipeline answer or retrieved contexts.")

            logger.info("[%d/%d] Successfully processed item %s", index, total, item_id)
            return {
                "id": item_id,
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": reference
            }, None
        except Exception as exc:
            logger.error("[%d/%d] Failed processing item %s: %s", index, total, item_id, exc)
            return None, {"id": item_id, "issue": f"{type(exc).__name__}: {exc}"}


async def collect_samples(items: list[dict]) -> tuple[list[dict], list[dict]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINE)
    total = len(items)
    
    tasks = [process_single_item(item, i, total, semaphore) for i, item in enumerate(items, 1)]
    results = await asyncio.gather(*tasks)

    samples = [res[0] for res in results if res[0] is not None]
    problems = [res[1] for res in results if res[1] is not None]
    return samples, problems


async def get_samples(refresh: bool) -> tuple[list[dict], list[dict]]:
    if CACHE_PATH.exists() and not refresh:
        logger.info("Loading evaluation cache from %s", CACHE_PATH)
        try:
            cache = load_json(CACHE_PATH)
            samples = validate_samples(cache.get("samples", []))
            if samples:
                return samples, cache.get("problems", [])
        except Exception as exc:
            logger.warning("Cache loading failed (%s). Re-running pipeline collection.", exc)

    items = load_json(DATASET_PATH)
    samples, problems = await collect_samples(items)
    if not samples:
        raise RuntimeError("No pipeline samples succeeded during collection.")

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump({"samples": samples, "problems": problems}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cache file: %s", exc)

    return samples, problems


# ============================================================
# EVALUATION
# ============================================================
def evaluate_samples(samples: list[dict]):
    logger.info("Initializing judge LLM and embeddings...")
    judge_llm = build_judge_llm()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    for m in (context_precision, context_recall, faithfulness, answer_relevancy):
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    dataset = Dataset.from_list([{k: s[k] for k in ("user_input", "retrieved_contexts", "response", "reference")} for s in samples])
    
    logger.info("Starting Ragas evaluation on %d samples with max_workers=%d...", len(samples), MAX_WORKERS)
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=MAX_WORKERS, timeout=JUDGE_TIMEOUT, max_retries=MAX_RETRIES),
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        logger.error("Critical error during Ragas evaluation execution: %s", exc)
        raise

    df = results.to_pandas()
    df.insert(0, "id", [s["id"] for s in samples])
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    logger.info("Ragas Results Summary:\n%s", results)
    logger.info("Evaluation report successfully saved to %s", REPORT_PATH.resolve())
    return results


# ============================================================
# MAIN
# ============================================================
async def main():
    refresh_cache = "--refresh" in sys.argv
    samples, problems = await get_samples(refresh_cache)
    await asyncio.to_thread(evaluate_samples, samples)
    if problems:
        logger.warning("Completed with %d excluded or failed pipeline records.", len(problems))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation process interrupted by user.")
    except Exception as exc:
        logger.critical("Evaluation pipeline failed fatally: %s: %s", type(exc).__name__, exc, exc_info=True)
        sys.exit(1)