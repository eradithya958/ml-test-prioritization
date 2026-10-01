"""
scripts/generate_synthetic_data.py
====================================
Generate a realistic synthetic test-execution dataset for pipeline validation.

Produces the SAME schema as scripts/collect_data.py so all downstream
code (feature engineering, training, evaluation) works identically.

Design decisions (documented for honest evaluation):
  - 600 tests, each "covers" 1-5 of 60 source files
  - 500 commits; each changes 1-8 source files
  - Failure probability = base_rate(test) + coverage_boost(changed ∩ covered)
  - Recent failures raise probability (simulates flaky/correlated tests)
  - Baseline failure rate ≈ 4-6% (realistic class imbalance)

Usage:
    python scripts/generate_synthetic_data.py --out-dir data/raw --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate(
    out_dir: Path,
    n_tests: int = 600,
    n_commits: int = 500,
    n_source_files: int = 60,
    history_window: int = 10,
    base_failure_rate: float = 0.04,
    coverage_boost: float = 0.25,
    recency_boost: float = 0.15,
    n_flaky_tests: int = 20,
    flaky_rate: float = 0.20,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Decide which tests are flaky (ground truth)
    # ------------------------------------------------------------------
    # Flaky tests: fail at a high random rate regardless of code changes
    # Ground truth label stored in is_flaky_ground_truth column

    # ------------------------------------------------------------------
    # 1. Source file names (mimic a real Python project layout)
    # ------------------------------------------------------------------
    modules = [
        "client",
        "auth",
        "models",
        "utils",
        "config",
        "exceptions",
        "cookies",
        "headers",
        "request",
        "response",
        "streams",
        "transports",
        "http2",
        "http11",
        "async_client",
        "sync_client",
        "decoders",
        "encoders",
        "status_codes",
        "timeout",
        "redirects",
        "middleware",
    ]
    source_files: list[str] = []
    for i in range(n_source_files):
        mod = modules[i % len(modules)]
        pkg = rng.choice(["httpx", "httpx/_internal", "httpx/transports"])
        source_files.append(f"{pkg}/{mod}_{i // len(modules)}.py")

    # ------------------------------------------------------------------
    # 2. Tests and their file coverage (which files they exercise)
    # ------------------------------------------------------------------
    test_ids: list[str] = [
        f"tests/test_{modules[i % len(modules)]}_{i:04d}.py::test_case" for i in range(n_tests)
    ]
    # Each test covers 1–5 source files
    test_coverage: dict[str, set[str]] = {}
    for tid in test_ids:
        n_covered = rng.randint(1, 5)
        test_coverage[tid] = set(rng.sample(source_files, n_covered))

    # Each test has an intrinsic base failure rate (some tests are inherently flakier)
    test_base_rates: dict[str, float] = {
        tid: np_rng.beta(0.5, 12.0)  # most tests: very low base rate; a few: higher
        for tid in test_ids
    }

    # ------------------------------------------------------------------
    # 3. Commit timeline
    # ------------------------------------------------------------------
    start_ts = datetime(2021, 1, 1, tzinfo=UTC)
    commit_shas: list[str] = [f"{rng.randint(0, 0xFFFFFFFF):08x}" * 5 for _ in range(n_commits)]
    commit_timestamps = [start_ts + timedelta(days=i * 2.5) for i in range(n_commits)]
    authors = [f"dev_{rng.randint(0, 8)}" for _ in range(n_commits)]

    # Each commit changes 1–8 source files
    commit_changed: list[list[str]] = [
        rng.sample(source_files, rng.randint(1, 8)) for _ in range(n_commits)
    ]
    commit_stats: list[dict] = [
        {
            "lines_added": rng.randint(1, 200),
            "lines_deleted": rng.randint(0, 100),
            "test_files_changed": rng.randint(0, 3),
        }
        for _ in range(n_commits)
    ]

    # ------------------------------------------------------------------
    # 4. Simulate test outcomes commit by commit
    # ------------------------------------------------------------------
    # Maintain rolling outcome history per test
    history: dict[str, list[int]] = {tid: [] for tid in test_ids}

    execution_rows: list[dict] = []
    commit_rows: list[dict] = []

    for c_idx, sha in enumerate(commit_shas):
        ts = commit_timestamps[c_idx]
        author = authors[c_idx]
        changed = commit_changed[c_idx]
        stats = commit_stats[c_idx]

        commit_rows.append(
            {
                "commit_sha": sha,
                "commit_ts": ts.isoformat(),
                "author": author,
                "files_changed": len(changed),
                "lines_added": stats["lines_added"],
                "lines_deleted": stats["lines_deleted"],
                "test_files_changed": stats["test_files_changed"],
                "src_files_changed": len(changed),
                "changed_file_list": json.dumps(changed),
            }
        )

        changed_set = set(changed)

        flaky_test_ids = set(test_ids[:n_flaky_tests])

        for tid in test_ids:
            is_flaky_gt = tid in flaky_test_ids
            hist = history[tid]

            # Compute failure probability
            p_fail = test_base_rates[tid]

            if is_flaky_gt:
                # Flaky test: high random failure rate, NOT correlated with changes
                p_fail = flaky_rate
            else:
                # Normal test: coverage boost + recency boost
                covered = test_coverage[tid]
                overlap = len(covered & changed_set)
                if overlap > 0:
                    p_fail += coverage_boost * (overlap / len(covered))

                if hist and hist[-1] == 1:
                    p_fail += recency_boost

            p_fail = min(p_fail, 0.95)
            outcome = int(np_rng.random() < p_fail)
            duration = max(0.01, np_rng.exponential(scale=2.5))

            # Build prev_N columns from history
            prev_vals = {}
            for lag in range(1, history_window + 1):
                idx = len(hist) - lag
                prev_vals[f"prev_{lag}"] = hist[idx] if idx >= 0 else float("nan")

            execution_rows.append(
                {
                    "commit_sha": sha,
                    "commit_ts": ts.isoformat(),
                    "author": author,
                    "files_changed": len(changed),
                    "lines_added": stats["lines_added"],
                    "lines_deleted": stats["lines_deleted"],
                    "test_files_changed": stats["test_files_changed"],
                    "src_files_changed": len(changed),
                    "changed_file_list": json.dumps(changed),
                    "test_id": tid,
                    "outcome": outcome,
                    "duration_s": round(duration, 3),
                    "error_msg": "AssertionError: synthetic fault" if outcome else None,
                    "workflow_run_id": c_idx * 1000,
                    "workflow_conclusion": "failure" if outcome else "success",
                    "is_flaky_ground_truth": is_flaky_gt,
                    **prev_vals,
                }
            )

            history[tid].append(outcome)
            if len(history[tid]) > history_window * 2:
                history[tid] = history[tid][-history_window * 2 :]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    exec_df = pd.DataFrame(execution_rows)
    commit_df = pd.DataFrame(commit_rows)

    exec_df["commit_ts"] = pd.to_datetime(exec_df["commit_ts"], utc=True)
    commit_df["commit_ts"] = pd.to_datetime(commit_df["commit_ts"], utc=True)

    total = len(exec_df)
    failures = int(exec_df["outcome"].sum())
    fail_rate = failures / total

    exec_df.to_parquet(out_dir / "test_executions.parquet", index=False)
    commit_df.to_parquet(out_dir / "commits.parquet", index=False)

    summary = {
        "repo": "SYNTHETIC",
        "since": str(exec_df["commit_ts"].min().date()),
        "until": str(exec_df["commit_ts"].max().date()),
        "total_rows": total,
        "failure_rows": failures,
        "pass_rows": total - failures,
        "failure_rate": round(fail_rate, 4),
        "unique_tests": exec_df["test_id"].nunique(),
        "unique_commits": exec_df["commit_sha"].nunique(),
        "commits_with_ci_results": n_commits,
        "commits_total": n_commits,
        "ci_coverage_pct": 100.0,
        "date_min": str(exec_df["commit_ts"].min()),
        "date_max": str(exec_df["commit_ts"].max()),
        "history_window": history_window,
        "class_imbalance_ratio": round((total - failures) / failures, 1)
        if failures
        else float("inf"),
        "known_limitations": [
            "SYNTHETIC DATA — not from a real repository",
            "Failure patterns are simulated; results will differ from real CI data",
            "Use only for pipeline validation; report real results when GitHub data is available",
        ],
    }

    import json as _json

    with open(out_dir / "dataset_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("SYNTHETIC DATASET GENERATED")
    log.info("  Rows     : %d", total)
    log.info("  Failures : %d (%.2f%%)", failures, fail_rate * 100)
    log.info("  Tests    : %d | Commits: %d", exec_df["test_id"].nunique(), n_commits)
    log.info("  Flaky (ground truth): %d injected", n_flaky_tests)
    log.info("  Saved → %s", out_dir)
    log.info("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic test execution dataset")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--n-tests", type=int, default=600)
    p.add_argument("--n-commits", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--failure-rate", type=float, default=0.04)
    p.add_argument(
        "--n-flaky",
        type=int,
        default=20,
        help="Number of flaky tests to inject (with ground-truth labels)",
    )
    p.add_argument(
        "--flaky-rate", type=float, default=0.20, help="Failure rate for injected flaky tests"
    )
    args = p.parse_args()
    generate(
        out_dir=Path(args.out_dir),
        n_tests=args.n_tests,
        n_commits=args.n_commits,
        base_failure_rate=args.failure_rate,
        n_flaky_tests=args.n_flaky,
        flaky_rate=args.flaky_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
