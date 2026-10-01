"""
scripts/run_experiment.py
==========================
End-to-end experiment runner: trains models and evaluates ALL strategies
(3 baselines + RF + XGBoost) on the same test split.

Output:
  results/tables/full_results.csv    — aggregate metrics per strategy
  results/tables/per_commit_full.csv — per-commit breakdown
  Console table ready to screenshot for your portfolio

Usage:
    # With synthetic data (no GitHub token needed):
    python scripts/generate_synthetic_data.py
    python scripts/run_experiment.py

    # With real GitHub Actions data:
    python scripts/collect_data.py --token $GITHUB_TOKEN
    python scripts/run_experiment.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

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
from src.models.baselines import CoverageBasedOrderer, RandomOrderer, RecentlyFailedFirstOrderer
from src.models.train import train as train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_commit_all(commit: CommitGroup, orderers: list) -> list[dict]:
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
                "ttff_s": ttff_val if not math.isinf(ttff_val) else float("nan"),
                "pct_to_catch_all": pct_val,
                "runtime_saved_pct": saved_val,
            }
        )
    return rows


def run(
    data_dir: Path,
    results_dir: Path,
    experiment_name: str,
    skip_training: bool = False,
) -> None:

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    models_dir = Path("experiments/models")

    # ------------------------------------------------------------------
    # Step 1: Train ML models (or load if skip_training)
    # ------------------------------------------------------------------
    loader = DatasetLoader(data_dir)
    df = loader.load()
    train_df, test_df = loader.train_test_split(df)

    if skip_training and (models_dir / "random_forest.pkl").exists():
        log.info("Loading pre-trained models from %s", models_dir)
        import pickle

        with open(models_dir / "random_forest.pkl", "rb") as f:
            rf = pickle.load(f)
        with open(models_dir / "xgboost.pkl", "rb") as f:
            xgb = pickle.load(f)
        with open(models_dir / "feature_engineer.pkl", "rb") as f:
            fe = pickle.load(f)
    else:
        rf, xgb, fe = train_models(
            data_dir=data_dir,
            models_dir=models_dir,
            log_dir=Path("experiments/logs"),
            experiment_name=experiment_name,
        )

    # ------------------------------------------------------------------
    # Step 2: Get PR-AUC for ML models (classification metric)
    # ------------------------------------------------------------------
    from sklearn.metrics import average_precision_score

    X_test = fe.transform(test_df)
    y_test = test_df["outcome"].values.astype(int)

    # Fill NaN (new tests unseen in training)
    X_test = X_test.fillna(0.0)

    rf_pr_auc = float(average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]))
    xgb_pr_auc = float(average_precision_score(y_test, xgb.predict_proba(X_test)[:, 1]))

    # ------------------------------------------------------------------
    # Step 3: Evaluate all strategies commit-by-commit on the test split
    # ------------------------------------------------------------------
    orderers = [
        RandomOrderer(seed=42),
        RecentlyFailedFirstOrderer(),
        CoverageBasedOrderer(),
        rf,
        xgb,
    ]

    all_rows: list[dict] = []
    commits_evaluated = 0

    log.info("Evaluating %d strategies on test split...", len(orderers))
    for commit in loader.iter_commits(test_df, failures_only=True):
        rows = evaluate_commit_all(commit, orderers)
        all_rows.extend(rows)
        commits_evaluated += 1

    log.info("Evaluated %d commits with failures", commits_evaluated)

    if not all_rows:
        log.error("No commits with failures in test split.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Save per-commit results
    # ------------------------------------------------------------------
    per_commit_df = pd.DataFrame(all_rows)
    per_commit_path = tables_dir / "per_commit_full.csv"
    per_commit_df.to_csv(per_commit_path, index=False)

    # ------------------------------------------------------------------
    # Step 5: Aggregate and build summary table
    # ------------------------------------------------------------------
    pr_auc_map = {
        "random": float("nan"),
        "recently_failed_first": float("nan"),
        "coverage_based": float("nan"),
        "random_forest": rf_pr_auc,
        "xgboost": xgb_pr_auc,
    }

    strategy_names = [o.name for o in orderers]
    agg_rows = []

    for strategy in strategy_names:
        strategy_data = [r for r in all_rows if r["strategy"] == strategy]
        metrics_list = [
            {
                "apfd": r["apfd"],
                "ttff_s": r["ttff_s"],
                "pct_to_catch_all": r["pct_to_catch_all"],
                "runtime_saved_pct": r["runtime_saved_pct"],
            }
            for r in strategy_data
        ]
        agg = aggregate_metrics(metrics_list)
        agg["strategy"] = strategy
        agg["commits_evaluated"] = len(strategy_data)
        agg["pr_auc"] = pr_auc_map.get(strategy, float("nan"))
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = tables_dir / "full_results.csv"
    agg_df.to_csv(agg_path, index=False, float_format="%.4f")

    # ------------------------------------------------------------------
    # Step 6: Print results table
    # ------------------------------------------------------------------
    _print_results_table(agg_rows, commits_evaluated)

    log.info("Results saved → %s", tables_dir)


def _print_results_table(agg_rows: list[dict], commits_evaluated: int) -> None:
    sep = "=" * 95

    def fmt(v: float, pct: bool = False, suffix: str = "") -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "  N/A "
        if pct:
            return f"{v * 100:6.1f}%"
        return f"{v:.4f}{suffix}"

    print("\n" + sep)
    print(" FULL EXPERIMENT RESULTS  —  test split (chronological 80/20)")
    print(f" Commits evaluated: {commits_evaluated}")
    print(sep)
    print(
        f"{'Strategy':<25} {'APFD ↑':>8} {'TTFF(s) ↓':>10} {'%Catch ↓':>9} "
        f"{'Runtime saved ↑':>16} {'PR-AUC ↑':>10}"
    )
    print("-" * 95)

    for row in agg_rows:
        name = row["strategy"]
        apfd_m = row.get("apfd_mean", float("nan"))
        ttff_m = row.get("ttff_s_mean", float("nan"))
        pct_m = row.get("pct_to_catch_all_mean", float("nan"))
        saved_m = row.get("runtime_saved_pct_mean", float("nan"))
        pr_auc = row.get("pr_auc", float("nan"))

        # Highlight ML models
        marker = " ◀" if name in ("random_forest", "xgboost") else "  "
        print(
            f"{name:<25}{marker}"
            f"{fmt(apfd_m):>8} "
            f"{fmt(ttff_m, suffix='s'):>10} "
            f"{fmt(pct_m, pct=True):>9} "
            f"{fmt(saved_m, pct=True):>16} "
            f"{fmt(pr_auc):>10}"
        )

    print(sep)
    print("\n Legend:")
    print("  APFD ↑   Average % Faults Detected.  Random ≈ 0.50.  Higher is better.")
    print("  TTFF ↓   Time to first failure (seconds of cumulative test runtime).")
    print("  %Catch ↓ Fraction of suite needed to catch ALL failing tests.")
    print("  Runtime saved ↑ % of suite runtime skippable after all faults detected.")
    print("  PR-AUC ↑ Precision-Recall AUC (ML models only). Baselines: N/A.\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end experiment: train + evaluate all strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--experiment-name", default="full_experiment")
    p.add_argument(
        "--skip-training", action="store_true", help="Load pre-trained models instead of retraining"
    )
    args = p.parse_args()
    run(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        experiment_name=args.experiment_name,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
