"""
scripts/detect_flaky.py
=========================
Analyze the collected dataset for flaky tests and update the quarantine registry.

Steps:
  1. Load test_executions.parquet
  2. Run FlakyDetector (flip-without-change + failure-without-change)
  3. Print detection report (with precision/recall if ground truth available)
  4. Write flaky tests to data/flaky_registry.json
  5. Print conftest.py integration snippet

Usage:
    python scripts/detect_flaky.py --threshold 0.10 --min-runs 10
    python scripts/detect_flaky.py --dry-run      # show results, don't update registry
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flaky.detector import FlakyDetector
from src.flaky.registry import FlakyRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(
    data_dir: Path,
    registry_path: Path,
    threshold: float,
    min_runs: int,
    dry_run: bool,
    top_n: int,
) -> None:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    exec_path = data_dir / "test_executions.parquet"
    if not exec_path.exists():
        log.error(
            "Dataset not found: %s  — run collect_data.py or generate_synthetic_data.py first",
            exec_path,
        )
        sys.exit(1)

    df = pd.read_parquet(exec_path)
    log.info(
        "Loaded %d rows | %d tests | %d commits",
        len(df),
        df["test_id"].nunique(),
        df["commit_sha"].nunique(),
    )

    has_ground_truth = "is_flaky_ground_truth" in df.columns
    if has_ground_truth:
        n_gt_flaky = int(df.groupby("test_id")["is_flaky_ground_truth"].any().sum())
        log.info("Ground truth available: %d flaky tests labeled", n_gt_flaky)

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------
    detector = FlakyDetector(threshold=threshold, min_runs=min_runs)
    report = detector.analyze(df)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    sep = "=" * 75
    print("\n" + sep)
    print(" FLAKY TEST DETECTION REPORT")
    print(sep)
    print(f"  Tests analyzed       : {report.total_tests_analyzed}")
    print(f"  Flaky detected       : {report.n_flaky_detected}")
    print(f"  Threshold            : {report.threshold}")
    print(f"  Min runs required    : {min_runs}")

    if report.precision is not None:
        print("\n  [Ground Truth Evaluation]")
        print(f"  Precision            : {report.precision:.3f}")
        print(f"  Recall               : {report.recall:.3f}")
        print(f"  F1-score             : {report.f1:.3f}")

    print("\n  [Build Impact]")
    print(f"  False-failure builds : {report.false_failure_builds_total}")
    print(f"  After quarantine     : {report.false_failure_builds_after_quarantine}")
    builds_saved = report.false_failure_builds_total - report.false_failure_builds_after_quarantine
    print(f"  Builds saved         : {builds_saved}")
    if report.false_failure_builds_total > 0:
        reduction_pct = builds_saved / report.false_failure_builds_total * 100
        print(f"  Reduction            : {reduction_pct:.1f}%")

    print(f"\n  Top {top_n} flakiest tests:")
    print(f"  {'Test ID':<55} {'Score':>6} {'FlipRate':>9} {'NoChgRate':>10} {'GT':>4}")
    print("  " + "-" * 88)

    flaky_tests = [m for m in report.per_test if m.is_flaky]
    for m in flaky_tests[:top_n]:
        gt_label = "✓" if m.ground_truth_flaky else "✗" if has_ground_truth else "?"
        tid_short = m.test_id[-55:] if len(m.test_id) > 55 else m.test_id
        print(
            f"  {tid_short:<55} {m.flaky_score:>6.3f} {m.flip_rate:>9.3f} "
            f"{m.fail_rate_no_change:>10.3f} {gt_label:>4}"
        )

    if has_ground_truth and report.precision is not None:
        fn_tests = [m for m in report.per_test if not m.is_flaky and m.ground_truth_flaky]
        if fn_tests:
            print(f"\n  False negatives (missed flaky tests): {len(fn_tests)}")
            for m in fn_tests[:5]:
                print(f"    {m.test_id}  score={m.flaky_score:.3f}")

    print(sep)

    # ------------------------------------------------------------------
    # Update registry (unless dry-run)
    # ------------------------------------------------------------------
    if dry_run:
        log.info("DRY RUN — registry not updated")
    else:
        registry = FlakyRegistry(registry_path)
        n_added = registry.add_from_report(report)
        print(f"\n  Registry updated: {n_added} tests added → {registry_path}")

        # Print conftest integration snippet
        snippet_path = data_dir.parent / "docs" / "conftest_flaky_snippet.py"
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet = registry.generate_conftest_snippet()
        snippet_path.write_text(snippet)
        print(f"  conftest.py snippet → {snippet_path}")
        print("\n  Add this to your conftest.py to activate quarantine:")
        print("  " + "-" * 50)
        print(snippet)

    # ------------------------------------------------------------------
    # Save report JSON
    # ------------------------------------------------------------------
    report_path = data_dir.parent / "results" / "tables" / "flaky_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report.summary_dict(), f, indent=2)
    log.info("Report saved → %s", report_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Detect and quarantine flaky tests from historical CI data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--registry", default="data/flaky_registry.json")
    p.add_argument("--threshold", type=float, default=0.10, help="Flakiness score threshold [0-1]")
    p.add_argument(
        "--min-runs", type=int, default=10, help="Minimum runs before a test can be quarantined"
    )
    p.add_argument("--top-n", type=int, default=15, help="Number of top flaky tests to display")
    p.add_argument("--dry-run", action="store_true", help="Analyze but do not update the registry")
    args = p.parse_args()
    run(
        data_dir=Path(args.data_dir),
        registry_path=Path(args.registry),
        threshold=args.threshold,
        min_runs=args.min_runs,
        dry_run=args.dry_run,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
