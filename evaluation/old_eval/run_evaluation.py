"""CLI for the RAG evaluation harness.

Examples
--------
  python run_evaluation.py                                  # full run, judge on
  python run_evaluation.py --retrieval-only --concurrency 4 # fast, no LLM judge
  python run_evaluation.py --types multi_metric comparative # only the hard categories
  python run_evaluation.py --baseline evaluation/results/latest_summary.json --fail-under 75   # CI gate
  python run_evaluation.py --run-name nightly --resume      # continue an interrupted run

Exit code is 1 when the quality gate fails (use in CI), 0 otherwise.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Evaluation may wait out Groq rate limits; serving keeps the short defaults. Must be set before app imports.
os.environ.setdefault("LLM_MAX_WAIT_S", "120")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")

from evaluation.evaluator import (
    DEFAULT_DATASET,
    DEFAULT_RESULTS_DIR,
    EvalConfig,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the financial RAG pipeline")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--run-name", help="Stable run id (default: UTC timestamp). Needed for --resume")
    p.add_argument("--resume", action="store_true", help="Skip questions already completed in --run-name")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel pipeline calls (keep 1 on low-TPM Groq tiers)")
    p.add_argument("--judge-concurrency", type=int, default=1)
    p.add_argument("--retry-passes", type=int, default=1, help="Re-run errored questions N times after a cooldown")
    p.add_argument("--timeout", type=float, default=None, help="Per-question timeout in seconds")
    p.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    p.add_argument("--types", nargs="+", help="Only these question_type values")
    p.add_argument("--retrieval-only", action="store_true", help="Skip the LLM judge")
    p.add_argument("--retrieval-k", default="ctx", help="Recall@k used in the score: an integer or 'ctx'")
    p.add_argument("--baseline", type=Path, help="Previous summary.json (or run dir) for regression checks")
    p.add_argument("--tolerance", type=float, default=2.0, help="Allowed drop vs baseline, in percentage points")
    p.add_argument("--fail-under", type=float, help="Gate: minimum overall score (percent)")
    p.add_argument("--max-error-rate", type=float, default=0.05, help="Gate: max share of errored questions")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = EvalConfig(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        run_name=args.run_name,
        resume=args.resume,
        concurrency=args.concurrency,
        judge_concurrency=args.judge_concurrency,
        retry_passes=args.retry_passes,
        limit=args.limit,
        question_types=args.types,
        use_judge=not args.retrieval_only,
        retrieval_k=args.retrieval_k,
        baseline_path=args.baseline,
        regression_tolerance=args.tolerance,
        fail_under=args.fail_under,
        max_error_rate=args.max_error_rate,
    )
    if args.timeout:
        cfg.timeout_s = args.timeout

    print("\nStarting RAG evaluation...\n")
    outcome = run_evaluation(cfg)
    return 0 if outcome["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())