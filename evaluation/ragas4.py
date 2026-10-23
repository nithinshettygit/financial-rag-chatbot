import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import datasets.arrow_dataset
datasets.arrow_dataset.generate_fingerprint = lambda self, *args, **kwargs: "dummy_fingerprint"

import httpx
import pandas as pd

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from app.pipeline import run_rag_pipeline_async, ChatRequest
from app.config import settings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

# ============================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================
DATASET_PATH = Path("evaluation/dataset/ragas_dataset_cleaned_all.json")
CACHE_PATH = Path("evaluation/ragas_samples_cache.json")
REPORT_PATH = Path("evaluation/report/ragas_evaluation_report.csv")

JUDGE_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_CONTEXTS = 5
MAX_CONTEXT_CHARS = 1500
MAX_CONCURRENT_PIPELINE = 2
JUDGE_TIMEOUT = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ragas_eval")

_DURATION_REGEX = re.compile(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE)
_DURATION_MAP = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

# ============================================================
# CUSTOM FAILOVER & RETRY TRANSPORT (DUAL KEYS & DUAL MODELS)
# ============================================================
class GroqFailoverTransport(httpx.AsyncBaseTransport):
    """Handles Groq multi-key/multi-model failover with exponential rate-limit backoff."""

    def __init__(self, api_keys: list[str]):
        self._api_keys = [k for k in api_keys if k and k.strip()]
        self._transport = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            payload = json.loads(await request.aread())
        except Exception:
            return await self._transport.handle_async_request(request)

        orig_model = payload.get("model", JUDGE_MODELS[0])
        
        # Build candidate pairs of (API Key, Model)
        candidates = [(k, m) for m in JUDGE_MODELS for k in self._api_keys]
        if self._api_keys:
            candidates = [(self._api_keys[0], orig_model)] + [c for c in candidates if c != (self._api_keys[0], orig_model)]

        max_outer_attempts = 4
        for outer_attempt in range(max_outer_attempts):
            last_resp, retry_delay = None, 0.0
            
            for i, (api_key, model) in enumerate(candidates):
                headers = httpx.Headers(request.headers)
                headers["authorization"] = f"Bearer {api_key}"
                headers.pop("content-length", None)

                attempt = httpx.Request(
                    request.method,
                    request.url,
                    headers=headers,
                    content=json.dumps({**payload, "model": model}).encode("utf-8"),
                    extensions=request.extensions.copy(),
                )

                try:
                    resp = await self._transport.handle_async_request(attempt)
                    body = await resp.aread()
                    status, resp_headers, extensions = resp.status_code, httpx.Headers(resp.headers), resp.extensions.copy()
                    await resp.aclose()
                except httpx.RequestError as exc:
                    logger.warning("[Transport Exception] Model: %s | Error: %s", model, exc)
                    continue

                if status == 200:
                    return self._build_response(status, resp_headers, body, attempt, extensions)

                if status == 429:
                    parsed_delay = self._parse_retry_delay(resp_headers, body)
                    retry_delay = max(retry_delay, parsed_delay)
                    logger.warning(
                        "[429 Rate-Limited] Key ending ...%s | Model: %s | Retry delay requested: %.2fs",
                        api_key[-4:], model, parsed_delay
                    )

                elif self._is_model_error(status, body):
                    logger.error("[Model Error HTTP %d] Model %s is invalid or decommissioned.", status, model)

                else:
                    logger.warning("[HTTP %d Failure] Key ending ...%s | Model: %s", status, api_key[-4:], model)

                last_resp = self._build_response(status, resp_headers, body, attempt, extensions)
                if i < len(candidates) - 1:
                    logger.info("Failing over to next available key/model option...")

            # If all candidates failed with rate limits, perform backoff sleep
            if last_resp and last_resp.status_code == 429:
                sleep_time = retry_delay if retry_delay > 0 else 5.0 * (outer_attempt + 1)
                logger.warning(
                    "[All Failover Options Exhausted] Attempt %d/%d rate limited. Sleeping for %.2fs...",
                    outer_attempt + 1, max_outer_attempts, sleep_time
                )
                await asyncio.sleep(sleep_time)
                continue

            return last_resp or httpx.Response(500, json={"error": {"message": "All failovers failed"}}, request=request)

        return last_resp or httpx.Response(500, json={"error": {"message": "All retry attempts exhausted"}}, request=request)

    @staticmethod
    def _parse_retry_delay(headers: httpx.Headers, body: bytes) -> float:
        if retry_after := headers.get("retry-after"):
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
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
        if status not in (400, 404):
            return False
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
    primary = (getattr(settings, "GROQ_API_KEY", "") or "").strip()
    fallback = (getattr(settings, "GROQ_API_KEY_FALLBACK", "") or "").strip()
    
    api_keys = [k for k in (primary, fallback) if k]
    if not api_keys:
        raise ValueError("No valid GROQ_API_KEY found in settings.")

    logger.info("Initializing Judge LLM with %d Groq API key(s)...", len(api_keys))
    
    custom_transport = GroqFailoverTransport(api_keys=api_keys)
    http_client = httpx.AsyncClient(transport=custom_transport, timeout=JUDGE_TIMEOUT)

    chat_model = ChatOpenAI(
        model=JUDGE_MODELS[0],
        openai_api_key=api_keys[0],
        openai_api_base="https://api.groq.com/openai/v1",
        timeout=JUDGE_TIMEOUT,
        max_retries=0,
        async_client=http_client,
    )
    return LangchainLLMWrapper(chat_model)


# ============================================================
# DATA VALIDATION & PIPELINE COLLECTION
# ============================================================
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Dataset or cache file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        return None, {"id": item_id, "issue": "Missing user_input or reference"}

    async with semaphore:
        try:
            output = await run_rag_pipeline_async(ChatRequest(query=query, session_id=f"eval_{item_id}"))
            answer = getattr(output, "answer", None)
            retrieved = getattr(output, "retrieved_chunks", None) or []
            contexts = [c.content[:MAX_CONTEXT_CHARS] for c in retrieved[:MAX_CONTEXTS] if getattr(c, "content", "").strip()]

            if not answer or not contexts:
                raise ValueError("Pipeline returned empty answer or contexts.")

            logger.info("[%d/%d] Pipeline processed item %s successfully.", index, total, item_id)
            return {
                "id": item_id,
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": reference
            }, None
        except Exception as exc:
            logger.error("[%d/%d] Pipeline failed for item %s: %s", index, total, item_id, exc)
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
        logger.info("Loading cached pipeline samples from %s", CACHE_PATH)
        try:
            cache = load_json(CACHE_PATH)
            samples = validate_samples(cache.get("samples", []))
            if samples:
                return samples, cache.get("problems", [])
        except Exception as exc:
            logger.warning("Failed to load cache (%s). Re-running pipeline execution...", exc)

    items = load_json(DATASET_PATH)
    samples, problems = await collect_samples(items)
    if not samples:
        raise RuntimeError("No pipeline samples succeeded.")

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump({"samples": samples, "problems": problems}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write evaluation cache: %s", exc)

    return samples, problems


# ============================================================
# ISOLATED METRIC EVALUATION LOOP
# ============================================================
def evaluate_single_sample_sync(metric, sample: dict) -> float:
    """Runs single metric scoring inside a dedicated thread and fresh task context."""
    async def _run():
        eval_sample = SingleTurnSample(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            response=sample["response"],
            reference=sample["reference"]
        )
        if hasattr(metric, "single_turn_ascore"):
            score = await metric.single_turn_ascore(eval_sample)
        else:
            score = await metric.ascore(eval_sample)
        return float(score) if score is not None else 0.0

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.warning("[Metric Exception] Sample: %s | Metric: %s | Error: %s", sample.get("id"), metric.name, exc)
        return float("nan")


async def evaluate_samples_robust(samples: list[dict]):
    logger.info("Setting up Judge LLM and HuggingFace Embeddings...")
    judge_llm = build_judge_llm()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    for m in metrics:
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    logger.info("Beginning evaluation across %d samples and %d metrics...", len(samples), len(metrics))
    
    results_data = []
    for idx, s in enumerate(samples, 1):
        logger.info("[%d/%d] Evaluating Sample ID: %s", idx, len(samples), s["id"])
        row_result = {"id": s["id"]}
        
        for m in metrics:
            # Delegate each metric evaluation to isolated thread/loop context
            score = await asyncio.to_thread(evaluate_single_sample_sync, m, s)
            row_result[m.name] = score
            logger.info("  └─ %s: %.4f", m.name, score)
            await asyncio.sleep(0.2)
            
        results_data.append(row_result)

    df = pd.DataFrame(results_data)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    logger.info("\n========== EVALUATION SUMMARY METRICS ==========\n%s\n===============================================", df.mean(numeric_only=True))
    logger.info("Evaluation report saved successfully to %s", REPORT_PATH.resolve())
    return df


# ============================================================
# MAIN EXECUTION ENTRYPOINT
# ============================================================
async def main():
    refresh_cache = "--refresh" in sys.argv
    samples, problems = await get_samples(refresh_cache)
    
    await evaluate_samples_robust(samples)

    if problems:
        logger.warning("Pipeline collection finished with %d excluded/failed items.", len(problems))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation process manually stopped.")
    except Exception as exc:
        logger.critical("Fatal error encountered during execution: %s: %s", type(exc).__name__, exc, exc_info=True)
        sys.exit(1)