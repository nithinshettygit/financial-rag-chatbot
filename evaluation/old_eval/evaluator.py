"""End-to-end RAG evaluation harness.

Runs every dataset question through the real pipeline (in-process, async), scores retrieval,
answer quality and abstention, then writes a reproducible run folder:

    evaluation/results/<run_id>/
        results.jsonl      one line per question, appended as it finishes (crash-safe, resumable)
        results.json       full per-question results in dataset order
        summary.json       aggregate metrics, per-type breakdown, latency, failure breakdown, run config
        failures.json      every non-passing question with a root-cause category
    evaluation/results/latest_summary.json   (copy; use as --baseline for regression gating)

Dataset item (all fields except `question` optional; legacy fields still work):
    {"id": "q001", "question": "...", "question_type": "single_fact|multi_metric|comparative|multi_year|followup|unanswerable",
     "answerable": true,
     "expected_answer": "...",
     "expected_facts": ["Revenue 2023 was $X"]  or  [{"fact": "Revenue 2023", "values": ["531.1", "531,100"]}],
     "gold_sources": [{"document": "duolingo", "pages": [45], "section": "Consolidated Statements of Operations"}],
     "entities": ["Duolingo", "Nike"],
     "history": [{"role": "user", "content": "..."}], "state": {"company": "Duolingo"}}
Legacy `source_page` / `source_reference` are converted to a single gold source.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app import pipeline as P
from app.pipeline import run_rag_pipeline_async, warmup_pipeline
from app.schemas import ChatRequest, ChatTurn, SessionState

from evaluation import metrics as M
from evaluation.answer_judge import JUDGE_MODEL, judge_answer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "dataset" / "evaluation_questions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


@dataclass
class EvalConfig:
    dataset_path: Path = DEFAULT_DATASET
    results_dir: Path = DEFAULT_RESULTS_DIR
    run_name: Optional[str] = None
    concurrency: int = 1                 # pipeline calls in flight; keep 1 on low-TPM Groq tiers
    judge_concurrency: int = 1
    timeout_s: float = float(os.getenv("EVAL_TIMEOUT_S", "180"))  # generous: includes rate-limit waits
    retry_passes: int = 1                # re-run errored questions this many times after a cooldown
    retry_cooldown_s: float = 45.0
    limit: Optional[int] = None
    question_types: Optional[List[str]] = None
    resume: bool = False
    use_judge: bool = True               # False = retrieval-only run (fast, free)
    retrieval_k: str = "ctx"             # retrieval score uses recall@<k>; "ctx" = what the LLM saw
    pass_correctness: float = 0.8
    pass_faithfulness: float = 0.8
    fail_under: Optional[float] = None   # gate: overall score (percent)
    max_error_rate: float = 0.05         # gate: share of errored/timed-out questions
    baseline_path: Optional[Path] = None
    regression_tolerance: float = 2.0    # percentage points


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _normalize_facts(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = []
    for f in raw.get("expected_facts") or []:
        if isinstance(f, str):
            facts.append({"fact": f, "values": []})
        elif isinstance(f, dict) and f.get("fact"):
            facts.append({"fact": str(f["fact"]), "values": [str(v) for v in f.get("values") or []]})
    if not facts and raw.get("expected_answer"):
        facts.append({"fact": str(raw["expected_answer"]), "values": []})
    return facts


def load_dataset(path: Path, limit: Optional[int] = None, types: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], str]:
    blob = Path(path).read_bytes()
    raw = json.loads(blob.decode("utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("questions", [])

    items, seen = [], set()
    for i, r in enumerate(raw):
        qid = str(r.get("id") or f"q{i + 1:03d}")
        if qid in seen:
            raise ValueError(f"Duplicate question id: {qid}")
        seen.add(qid)
        if not r.get("question"):
            raise ValueError(f"Question {qid} has no text")

        answerable = bool(r.get("answerable", True))
        gold = M.normalize_gold_sources(r)
        qtype = r.get("question_type") or ("unanswerable" if not answerable else "single_fact")
        if answerable and not gold:
            logger.warning("%s: answerable but no gold sources -> retrieval will not be scored", qid)

        items.append(
            {
                "id": qid,
                "question": r["question"],
                "question_type": qtype,
                "answerable": answerable,
                "expected_answer": r.get("expected_answer", ""),
                "expected_facts": _normalize_facts(r) if answerable else [],
                "gold_sources": gold,
                "entities": r.get("entities") or [],
                "history": r.get("history") or [],
                "state": r.get("state") or {},
            }
        )

    if types:
        items = [x for x in items if x["question_type"] in set(types)]
    if limit:
        items = items[:limit]
    return items, hashlib.sha256(blob).hexdigest()[:16]


def _dump(model: Any) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


# --------------------------------------------------------------------------- #
# Scoring one result
# --------------------------------------------------------------------------- #
def _retrieval_key(cfg: EvalConfig) -> str:
    return f"recall@{cfg.retrieval_k}"


def score_result(r: Dict[str, Any], cfg: EvalConfig) -> None:
    """Adds score (0-1 or None), passed (bool or None) and failure_category."""
    r["score"], r["passed"], r["failure_category"] = None, None, None

    if "error" in r:
        r.update(score=0.0, passed=False, failure_category=r.get("error_type", "error"))
        return

    if not r["answerable"]:
        ok = bool(r.get("abstained"))
        r.update(score=1.0 if ok else 0.0, passed=ok, failure_category=None if ok else "hallucinated_answer")
        return

    rm = r.get("retrieval_metrics")
    recall = rm.get(_retrieval_key(cfg)) if rm else None
    aq = r.get("answer_quality")
    judged = bool(aq) and not aq.get("judge_failed")

    correctness = aq.get("correctness") if judged else None
    faithfulness = aq.get("faithfulness") if judged else None
    quality = None
    if judged and correctness is not None:
        quality = (correctness + faithfulness + aq["relevancy"]) / 3

    if quality is not None and recall is not None:
        score = 0.3 * recall + 0.7 * quality
    elif quality is not None:
        score = quality
    else:
        score = recall  # retrieval-only run (or judge failure with gold)

    if cfg.use_judge and aq and aq.get("judge_failed"):
        r["score"], r["passed"], r["failure_category"] = None, None, "judge_failed"
        return

    if quality is not None:
        passed = (
            correctness >= cfg.pass_correctness
            and faithfulness >= cfg.pass_faithfulness
            and not r.get("abstained")
        )
    elif recall is not None:
        passed = recall >= 0.999
    else:
        passed = None

    r["score"], r["passed"] = score, passed
    if passed is False:
        full_recall = rm.get("recall@ctx") if rm else None
        if full_recall is not None and full_recall < 0.999:
            r["failure_category"] = "retrieval_miss" if full_recall == 0 else "partial_retrieval"
        elif r.get("abstained") or (aq and aq.get("partially_abstained")):
            r["failure_category"] = "false_abstention"
        elif faithfulness is not None and faithfulness < cfg.pass_faithfulness:
            r["failure_category"] = "unfaithful_answer"
        else:
            r["failure_category"] = "generation_error"


# --------------------------------------------------------------------------- #
# Evaluate one question
# --------------------------------------------------------------------------- #
async def evaluate_one(item: Dict[str, Any], cfg: EvalConfig, run_id: str, judge_sem: asyncio.Semaphore) -> Dict[str, Any]:
    base = {
        "id": item["id"],
        "question": item["question"],
        "question_type": item["question_type"],
        "answerable": item["answerable"],
        "expected_answer": item["expected_answer"],
        "expected_facts": item["expected_facts"],
        "gold_sources": item["gold_sources"],
        "entities": item["entities"],
    }
    t0 = time.perf_counter()

    try:
        request = ChatRequest(
            session_id=f"eval-{run_id}-{item['id']}",
            query=item["question"],
            history=[ChatTurn(**t) for t in item["history"]],
            state=SessionState(**item["state"]),
        )
        response = await asyncio.wait_for(run_rag_pipeline_async(request), timeout=cfg.timeout_s)
    except asyncio.TimeoutError:
        r = {**base, "error": f"timeout after {cfg.timeout_s:.0f}s", "error_type": "timeout",
             "latency_s": time.perf_counter() - t0}
        score_result(r, cfg)
        return r
    except Exception as exc:
        logger.exception("Pipeline failed for %s", item["id"])
        limited = "429" in str(exc) or "rate_limit" in str(exc).lower()
        r = {**base, "error": str(exc), "error_type": "rate_limited" if limited else "exception",
             "latency_s": time.perf_counter() - t0}
        score_result(r, cfg)
        return r

    latency = time.perf_counter() - t0
    chunks = [_dump(c) for c in response.retrieved_chunks]
    context = "\n\n".join(c.get("content", "") for c in chunks)

    r: Dict[str, Any] = {
        **base,
        "actual_answer": response.answer,
        "intent": response.intent,
        "search_query": response.search_query,
        "state": _dump(response.state),
        "source_map": response.source_map,
        "latency_s": latency,
        "retrieved_chunks": [
            {**{k: v for k, v in c.items() if k != "content"}, "content_preview": c.get("content", "")[:500]}
            for c in chunks
        ],
    }

    if item["answerable"]:
        if item["gold_sources"]:
            r["retrieval_metrics"] = M.calculate_retrieval_metrics(chunks, item["gold_sources"])
        cov = M.entity_coverage(chunks, item["entities"])
        if cov is not None:
            r["entity_coverage"] = cov
        numeric = M.numeric_fact_recall(response.answer, item["expected_facts"])
        if numeric:
            r["numeric_check"] = numeric

    r["abstained_regex"] = M.is_abstention(response.answer)
    judge = None
    if cfg.use_judge:
        async with judge_sem:
            judge = await asyncio.to_thread(
                judge_answer,
                question=item["question"],
                expected_answer=item["expected_answer"],
                expected_facts=[f["fact"] for f in item["expected_facts"]],
                actual_answer=response.answer,
                context=context,
                answerable=item["answerable"],
            )
        r["answer_quality"] = judge

    r["abstained"] = judge["abstained"] if judge and not judge["judge_failed"] else r["abstained_regex"]
    score_result(r, cfg)
    return r


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x * 100, 2)


def _grade(score: Optional[float]) -> str:
    if score is None:
        return "n/a"
    for threshold, letter in ((0.90, "A"), (0.80, "B"), (0.70, "C"), (0.60, "D")):
        if score >= threshold:
            return letter
    return "F"


def _group_stats(rows: List[Dict[str, Any]], cfg: EvalConfig) -> Dict[str, Any]:
    ans = [r for r in rows if r["answerable"] and "error" not in r]
    judged = [r for r in ans if r.get("answer_quality") and not r["answer_quality"]["judge_failed"]]
    return {
        "n": len(rows),
        "score": _pct(M.mean(r["score"] for r in rows)),
        "pass_rate": _pct(M.mean(None if r["passed"] is None else float(r["passed"]) for r in rows)),
        "recall": _pct(M.mean(r["retrieval_metrics"].get(_retrieval_key(cfg)) for r in ans if r.get("retrieval_metrics"))),
        "correctness": _pct(M.mean(r["answer_quality"]["correctness"] for r in judged)),
        "faithfulness": _pct(M.mean(r["answer_quality"]["faithfulness"] for r in judged)),
        "errors": sum(1 for r in rows if "error" in r),
    }


def build_summary(results: List[Dict[str, Any]], cfg: EvalConfig, wall_time_s: float) -> Dict[str, Any]:
    n = len(results)
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]

    retrieval = M.aggregate_retrieval_metrics(ans)
    judged = [r for r in ans if r.get("answer_quality") and not r["answer_quality"]["judge_failed"]]
    judge_failed = sum(1 for r in ok if r.get("answer_quality") and r["answer_quality"]["judge_failed"])

    correctness = M.mean(r["answer_quality"]["correctness"] for r in judged)
    faithfulness = M.mean(r["answer_quality"]["faithfulness"] for r in judged)
    relevancy = M.mean(r["answer_quality"]["relevancy"] for r in judged)
    aq_parts = [x for x in (correctness, faithfulness, relevancy) if x is not None]
    aq_overall = sum(aq_parts) / len(aq_parts) if aq_parts else None

    abst_correct = M.mean(float(bool(r.get("abstained"))) for r in unans)
    false_abst = M.mean(float(bool(r.get("abstained"))) for r in ans)
    partial_abst = M.mean(float(bool(r.get("answer_quality", {}) and r["answer_quality"].get("partially_abstained"))) for r in judged)

    overall = M.mean(r["score"] for r in results)
    pass_rate = M.mean(None if r["passed"] is None else float(r["passed"]) for r in results)

    lat_ok = [r["latency_s"] for r in ok]
    lat_all = [r["latency_s"] for r in results if "latency_s" in r]

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    failure_breakdown = Counter(r["failure_category"] for r in results if r.get("failure_category"))

    return {
        "run": {"wall_time_s": round(wall_time_s, 1), "concurrency": cfg.concurrency, "judge_enabled": cfg.use_judge},
        "dataset": {
            "total_questions": n,
            "answerable": len(ans),
            "unanswerable": len(unans),
            "errors": len(errors),
            "timeouts": sum(1 for r in errors if r.get("error_type") == "timeout"),
            "rate_limited": sum(1 for r in errors if r.get("error_type") == "rate_limited"),
            "judged": len(judged),
            "error_rate": _pct(len(errors) / n) if n else None,
            "judge_failed": judge_failed,
            "retrieval_scored": sum(1 for r in ans if r.get("retrieval_metrics")),
        },
        "retrieval": {k: _pct(v) for k, v in retrieval.items()},
        "entity_coverage": _pct(M.mean(r.get("entity_coverage") for r in ans)),
        "answer_quality": {
            "correctness": _pct(correctness),
            "faithfulness": _pct(faithfulness),
            "relevancy": _pct(relevancy),
            "overall": _pct(aq_overall),
            "numeric_fact_recall": _pct(M.mean(r["numeric_check"]["recall"] for r in ans if r.get("numeric_check"))),
        },
        "abstention": {
            "correct_abstention_rate": _pct(abst_correct),
            "false_answer_rate": _pct(None if abst_correct is None else 1 - abst_correct),
            "false_abstention_rate": _pct(false_abst),
            "partial_abstention_rate": _pct(partial_abst),
        },
        "latency": {k: (None if v is None else round(v, 2)) for k, v in M.latency_stats(lat_ok).items()}
        | {"p95_incl_timeouts_s": None if not lat_all else round(M.percentile(lat_all, 95), 2)},
        "overall": {
            "overall_score": _pct(overall),
            "pass_rate": _pct(pass_rate),
            "grade": _grade(overall),
            "status": "PASS" if overall is not None and overall >= 0.75 else "NEEDS IMPROVEMENT",
        },
        "by_question_type": {t: _group_stats(rows, cfg) for t, rows in sorted(by_type.items())},
        "failure_breakdown": dict(failure_breakdown),
    }


# --------------------------------------------------------------------------- #
# Baseline comparison / quality gate
# --------------------------------------------------------------------------- #
_GATE_METRICS = [
    ("overall", "overall_score", True, "abs"),
    ("overall", "pass_rate", True, "abs"),
    ("retrieval", "recall@ctx", True, "abs"),
    ("answer_quality", "correctness", True, "abs"),
    ("answer_quality", "faithfulness", True, "abs"),
    ("abstention", "correct_abstention_rate", True, "abs"),
    ("abstention", "false_abstention_rate", False, "abs"),
    ("latency", "p95_s", False, "rel"),
]


def compare_to_baseline(summary: Dict[str, Any], baseline: Dict[str, Any], tol: float) -> Dict[str, Any]:
    rows, regressions = [], []
    for section, key, higher_better, mode in _GATE_METRICS:
        cur = summary.get(section, {}).get(key)
        old = baseline.get(section, {}).get(key)
        if cur is None or old is None:
            continue
        delta = cur - old
        worse_by = -delta if higher_better else delta
        bad = worse_by > tol if mode == "abs" else (old > 0 and worse_by / old > 0.25)
        rows.append({"metric": f"{section}.{key}", "baseline": old, "current": cur, "delta": round(delta, 2), "regression": bad})
        if bad:
            regressions.append(f"{section}.{key}: {old} -> {cur}")
    return {"rows": rows, "regressions": regressions}


def _load_baseline(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _snapshot(cfg: EvalConfig, dataset_hash: str, n: int) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(cfg.dataset_path), "sha256_16": dataset_hash, "questions": n},
        "models": {
            "generator": settings.LLM_MODEL,
            "judge": JUDGE_MODEL if cfg.use_judge else None,
            "embedding": getattr(settings, "EMBEDDING_MODEL", None),
            "reranker": getattr(P, "RERANKER_MODEL", None),
        },
        "retrieval": {
            k: getattr(P, k, None)
            for k in ("RERANK_CANDIDATES", "FINAL_TOP_K", "MAX_FINAL_TOP_K", "RERANK_PER_SQ",
                      "RERANK_MAX_PAIRS", "MIN_STRICT_YIELD", "RERANK_MAX_LENGTH")
        },
        "eval_config": {k: str(v) for k, v in asdict(cfg).items()},
    }


async def run_evaluation_async(cfg: Optional[EvalConfig] = None) -> Dict[str, Any]:
    cfg = cfg or EvalConfig()
    items, dataset_hash = load_dataset(cfg.dataset_path, cfg.limit, cfg.question_types)
    if not items:
        raise ValueError("Dataset is empty after filtering")
    if cfg.use_judge and JUDGE_MODEL == settings.LLM_MODEL:
        logger.warning("Judge model == generator model (%s). Set EVAL_JUDGE_MODEL to a different model to reduce bias.", JUDGE_MODEL)

    if cfg.resume and not cfg.run_name:
        raise ValueError("--resume requires --run-name")
    run_id = cfg.run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"

    done: Dict[str, Dict[str, Any]] = {}
    if cfg.resume and jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "error" not in rec:  # re-run errored/timed-out questions
                    done[rec["id"]] = rec
        logger.info("Resuming: %d/%d questions already evaluated", len(done), len(items))
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()), encoding="utf-8")

    todo = [it for it in items if it["id"] not in done]
    logger.info("Warming up pipeline (loading models)...")
    await warmup_pipeline()

    sem = asyncio.Semaphore(cfg.concurrency)
    judge_sem = asyncio.Semaphore(cfg.judge_concurrency)
    completed = len(done)
    start = time.perf_counter()

    async def _run(item: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        async with sem:
            r = await evaluate_one(item, cfg, run_id, judge_sem)
        completed += 1
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("[%d/%d] %s %s (%.1fs)", completed, len(items), r["id"],
                    "PASS" if r["passed"] else ("ERROR" if "error" in r else "FAIL" if r["passed"] is False else "n/a"),
                    r.get("latency_s", 0.0))
        return r

    new_results = await asyncio.gather(*[_run(it) for it in todo])
    by_id = {**done, **{r["id"]: r for r in new_results}}

    for p in range(cfg.retry_passes):
        failed = [it for it in todo if "error" in by_id[it["id"]]]
        if not failed:
            break
        logger.warning("Retry pass %d: re-running %d errored questions after %.0fs cooldown", p + 1, len(failed), cfg.retry_cooldown_s)
        await asyncio.sleep(cfg.retry_cooldown_s)
        for r in await asyncio.gather(*[_run(it) for it in failed]):
            by_id[r["id"]] = r
    wall = time.perf_counter() - start

    results = [by_id[it["id"]] for it in items]  # dataset order
    for r in results:  # refresh scores if thresholds changed between resume runs
        score_result(r, cfg)

    summary = build_summary(results, cfg, wall)
    summary["config"] = _snapshot(cfg, dataset_hash, len(items))

    # ---- quality gate ----
    reasons: List[str] = []
    overall = summary["overall"]["overall_score"]
    if cfg.fail_under is not None and (overall is None or overall < cfg.fail_under):
        reasons.append(f"overall score {overall} < fail-under {cfg.fail_under}")
    err_rate = (summary["dataset"]["error_rate"] or 0) / 100
    if err_rate > cfg.max_error_rate:
        reasons.append(f"error rate {err_rate:.1%} > max {cfg.max_error_rate:.1%}")
    if cfg.baseline_path:
        comparison = compare_to_baseline(summary, _load_baseline(cfg.baseline_path), cfg.regression_tolerance)
        summary["baseline_comparison"] = comparison
        reasons += [f"regression {x}" for x in comparison["regressions"]]
    gate = {"passed": not reasons, "reasons": reasons}
    summary["gate"] = gate

    failures = [
        {k: r.get(k) for k in ("id", "question_type", "question", "failure_category", "expected_answer",
                               "actual_answer", "error", "retrieval_metrics", "answer_quality", "search_query")}
        for r in results if r.get("passed") is False
    ]

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.results_dir / "latest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(summary, run_dir)
    return {"results": results, "summary": summary, "gate": gate, "run_dir": run_dir}


def run_evaluation(cfg: Optional[EvalConfig] = None) -> Dict[str, Any]:
    return asyncio.run(run_evaluation_async(cfg))


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def _f(v: Optional[float], suffix: str = "%") -> str:
    return "n/a" if v is None else f"{v:.2f}{suffix}"


def print_report(summary: Dict[str, Any], run_dir: Path) -> None:
    d, rt, aq = summary["dataset"], summary["retrieval"], summary["answer_quality"]
    ab, lat, ov = summary["abstention"], summary["latency"], summary["overall"]
    line = "-" * 64

    print("\n" + "=" * 64)
    print("                 FINANCIAL RAG EVALUATION")
    print("=" * 64)
    bad = d["errors"] + d["judge_failed"]
    if d["total_questions"] and bad / d["total_questions"] > 0.10:
        print(f"!! RESULTS NOT RELIABLE: {d['errors']} pipeline errors ({d['rate_limited']} rate-limited) and "
              f"{d['judge_failed']} judge failures. Quality metrics below cover only {d['judged']} judged questions.")
    print(f"Questions {d['total_questions']} | answerable {d['answerable']} | unanswerable {d['unanswerable']}"
          f" | errors {d['errors']} (timeouts {d['timeouts']}) | judge failures {d['judge_failed']}")

    print(f"\nRETRIEVAL  (scored on {d['retrieval_scored']} questions with gold sources)\n{line}")
    for k in (1, 3, 5, 10):
        print(f"@{k:<3} hit {_f(rt.get(f'hit@{k}')):>8} | recall {_f(rt.get(f'recall@{k}')):>8} | "
              f"precision {_f(rt.get(f'precision@{k}')):>8} | nDCG {_f(rt.get(f'ndcg@{k}')):>8}")
    print(f"@ctx recall {_f(rt.get('recall@ctx'))}  (gold sources present in what the LLM saw)")
    print(f"MRR {_f(rt.get('mrr'))} | entity coverage {_f(summary['entity_coverage'])}")

    print(f"\nANSWER QUALITY (LLM judge)\n{line}")
    print(f"Correctness {_f(aq['correctness'])} | Faithfulness {_f(aq['faithfulness'])} | "
          f"Relevancy {_f(aq['relevancy'])} | Overall {_f(aq['overall'])}")
    print(f"Numeric fact recall (deterministic cross-check) {_f(aq['numeric_fact_recall'])}")

    print(f"\nABSTENTION\n{line}")
    print(f"Correct abstention {_f(ab['correct_abstention_rate'])} | hallucinated answers {_f(ab['false_answer_rate'])}")
    print(f"False abstention on answerable {_f(ab['false_abstention_rate'])} | partial {_f(ab['partial_abstention_rate'])}")

    print(f"\nLATENCY\n{line}")
    print(f"mean {_f(lat['mean_s'], 's')} | p50 {_f(lat['p50_s'], 's')} | p95 {_f(lat['p95_s'], 's')} | max {_f(lat['max_s'], 's')}")

    print(f"\nBY QUESTION TYPE\n{line}")
    print(f"{'type':<16}{'n':>4}{'score':>9}{'pass':>9}{'recall':>9}{'correct':>9}{'errors':>8}")
    for t, s in summary["by_question_type"].items():
        print(f"{t:<16}{s['n']:>4}{_f(s['score']):>9}{_f(s['pass_rate']):>9}{_f(s['recall']):>9}"
              f"{_f(s['correctness']):>9}{s['errors']:>8}")

    if summary["failure_breakdown"]:
        print(f"\nFAILURE BREAKDOWN\n{line}")
        for cat, cnt in sorted(summary["failure_breakdown"].items(), key=lambda x: -x[1]):
            print(f"{cat:<24}{cnt}")

    cmp_ = summary.get("baseline_comparison")
    if cmp_:
        print(f"\nVS BASELINE\n{line}")
        for row in cmp_["rows"]:
            flag = "  <-- REGRESSION" if row["regression"] else ""
            print(f"{row['metric']:<36}{row['baseline']:>9} -> {row['current']:>9} ({row['delta']:+.2f}){flag}")

    print("\n" + "=" * 64)
    print(f"OVERALL SCORE {_f(ov['overall_score'])} | pass rate {_f(ov['pass_rate'])} | "
          f"grade {ov['grade']} | {ov['status']}")
    gate = summary.get("gate", {})
    print(f"QUALITY GATE: {'PASSED' if gate.get('passed') else 'FAILED'}")
    for reason in gate.get("reasons", []):
        print(f"  - {reason}")
    print(f"Results: {run_dir}")
    print("=" * 64 + "\n")