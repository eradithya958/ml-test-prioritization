"""
scripts/evaluate_baselines.py
==============================
Evaluate all baseline orderers on the collected dataset and produce:
  - results/tables/baselines_results.csv   — per-strategy aggregate metrics
  - results/tables/per_commit_results.csv  — per-commit breakdown (for plots)
  - Console summary table

Usage:
    python scripts/evaluate_baselines.py \
        --data-dir data/raw \
        --results-dir results

Requires: data/raw/test_executions.parquet (from Phase 1 collect_data.py)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import CommitGroup, DatasetLoader
from src.evaluation.metrics import (
    aggregate_metrics,
    apfd,
    pct_tests_to_catch_all,
    runtime_saved_pct,
    time_to_first_failure,
)
from src.models.base import OrderContext
from src.models.baselines import (
    CoverageBasedOrderer,
    RandomOrderer,
    RecentlyFailedFirstOrderer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate_commit(
    commit: CommitGroup,
    orderers: list,
) -> list[dict]:
    """
    Evaluate all orderers on a single commit.

    Returns a list of metric dicts, one per orderer.
    """
    test_ids = list(commit.labels.keys())
    failing = commit.failing_tests
    durations = dict(
        zip(commit.tests["test_id"], commit.tests["duration_s"].fillna(0.0), strict=False)
    )

    rows = []
    for orderer in orderers:
        ctx = OrderContext(
            commit_sha=commit.commit_sha,
            changed_files=commit.changed_files,
            test_history=commit.tests,
        )
        ordered = orderer.order(test_ids, ctx)

        # Sanity check: orderer must return the same set of tests
        assert set(ordered) == set(test_ids), (
            f"{orderer.name}: returned wrong test set for commit {commit.commit_sha[:8]}"
        )

        apfd_val = apfd(ordered, failing)
        ttff_val = time_to_first_failure(ordered, failing, durations)
        pct_val = pct_tests_to_catch_all(ordered, failing)
        saved_val = runtime_saved_pct(ordered, failing, durations)

        rows.append(
            {
                "strategy": orderer.name,
                "commit_sha": commit.commit_sha,
                "commit_ts": str(commit.commit_ts),
                "n_tests": len(test_ids),
                "n_failures": len(failing),
                "apfd": apfd_val,
                "ttff_s": ttff_val,
                "pct_to_catch_all": pct_val,
                "runtime_saved_pct": saved_val,
            }
        )
    return rows


def run(data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    loader = DatasetLoader(data_dir)
    df = loader.load()
    _, test_df = loader.train_test_split(df)

    log.info("Evaluating on test split: %d commits", test_df["commit_sha"].nunique())

    # ------------------------------------------------------------------
    # Define orderers
    # ------------------------------------------------------------------
    orderers = [
        RandomOrderer(seed=42),
        RecentlyFailedFirstOrderer(),
        CoverageBasedOrderer(),
    ]

    # ------------------------------------------------------------------
    # Evaluate commit by commit
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    commits_evaluated = 0

    for commit in loader.iter_commits(test_df, failures_only=True):
        rows = evaluate_commit(commit, orderers)
        all_rows.extend(rows)
        commits_evaluated += 1

    if not all_rows:
        log.error("No commits with failures found in the test split.")
        sys.exit(1)

    log.info("Evaluated %d commits with failures", commits_evaluated)

    # ------------------------------------------------------------------
    # Per-commit results CSV
    # ------------------------------------------------------------------
    per_commit_path = tables_dir / "per_commit_results.csv"
    per_commit_df = pd.DataFrame(all_rows)
    per_commit_df.to_csv(per_commit_path, index=False)
    log.info("Saved per-commit results → %s", per_commit_path)

    # ------------------------------------------------------------------
    # Aggregate by strategy
    # ------------------------------------------------------------------
    strategies = [o.name for o in orderers]
    agg_rows = []

    for strategy in strategies:
        strategy_rows = [r for r in all_rows if r["strategy"] == strategy]
        metrics_list = [
            {
                "apfd": r["apfd"],
                "ttff_s": r["ttff_s"] if not math.isinf(r["ttff_s"]) else float("nan"),
                "pct_to_catch_all": r["pct_to_catch_all"],
                "runtime_saved_pct": r["runtime_saved_pct"],
            }
            for r in strategy_rows
        ]
        agg = aggregate_metrics(metrics_list)
        agg["strategy"] = strategy
        agg["commits_evaluated"] = len(strategy_rows)
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    # Put strategy first
    cols = ["strategy", "commits_evaluated"] + [
        c for c in agg_df.columns if c not in ("strategy", "commits_evaluated")
    ]
    agg_df = agg_df[cols]
    agg_path = tables_dir / "baselines_results.csv"
    agg_df.to_csv(agg_path, index=False, float_format="%.4f")
    log.info("Saved aggregate results → %s", agg_path)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    SEP = "=" * 90
    print("\n" + SEP)
    print("BASELINE EVALUATION RESULTS  (test split, chronological 80/20)")
    print(SEP)
    print(
        f"{'Strategy':<25} {'APFD↑':>8} {'TTFF(s)↓':>10} {'%Catch↓':>9} {'Runtime saved↑':>15} {'N commits':>10}"
    )
    print("-" * 90)

    for row in agg_rows:
        name = row["strategy"]
        apfd_m = row.get("apfd_mean", float("nan"))
        ttff_m = row.get("ttff_s_mean", float("nan"))
        pct_m = row.get("pct_to_catch_all_mean", float("nan"))
        saved_m = row.get("runtime_saved_pct_mean", float("nan"))
        n = int(row["commits_evaluated"])

        def fmt(v: float, pct: bool = False, suffix: str = "") -> str:
            if math.isnan(v) or math.isinf(v):
                return "N/A"
            return f"{v * 100:.1f}%" if pct else f"{v:.3f}{suffix}"

        print(
            f"{name:<25} {fmt(apfd_m):>8} {fmt(ttff_m, suffix='s'):>10} "
            f"{fmt(pct_m, pct=True):>9} {fmt(saved_m, pct=True):>15} {n:>10}"
        )

    print(SEP)
    print("\nNotes:")
    print("  APFD ↑   higher is better. Random baseline ≈ 0.5 expected.")
    print("  TTFF ↓   lower is better. Time until first failing test is seen.")
    print("  %Catch ↓ lower is better. Fraction of suite needed to catch all failures.")
    print("  Runtime saved ↑ higher is better. % of suite runtime you can skip.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline test prioritization strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data/raw", help="Directory with test_executions.parquet"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Output directory for tables and plots"
    )
    args = parser.parse_args()
    run(Path(args.data_dir), Path(args.results_dir))


if __name__ == "__main__":
    main()
