"""
Phase 1 — Dataset Inspection Script
=====================================
Run this after collect_data.py to get a full statistical profile
of the collected dataset.

Usage:
    python scripts/inspect_dataset.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def report(data_dir: Path) -> None:
    exec_path = data_dir / "test_executions.parquet"
    commits_path = data_dir / "commits.parquet"
    summary_path = data_dir / "dataset_summary.json"

    if not exec_path.exists():
        print(f"ERROR: {exec_path} not found. Run collect_data.py first.")
        return

    df = pd.read_parquet(exec_path)
    commits = pd.read_parquet(commits_path) if commits_path.exists() else None

    with open(summary_path) as f:
        summary = json.load(f)

    sep = "=" * 70

    print(sep)
    print("DATASET OVERVIEW")
    print(sep)
    print(f"  Rows (test executions) : {len(df):,}")
    print(f"  Unique tests           : {df['test_id'].nunique():,}")
    print(f"  Unique commits         : {df['commit_sha'].nunique():,}")
    print(f"  Date range             : {df['commit_ts'].min()} → {df['commit_ts'].max()}")
    print(f"  Failure rate           : {summary['failure_rate'] * 100:.2f}%")
    print(f"  Class imbalance (N:P)  : {summary['class_imbalance_ratio']:.1f} : 1")

    print()
    print(sep)
    print("OUTCOME DISTRIBUTION")
    print(sep)
    vc = df["outcome"].value_counts().sort_index()
    for outcome, count in vc.items():
        label = {0: "PASS", 1: "FAIL"}.get(outcome, str(outcome))
        print(f"  {label}: {count:>10,}  ({count / len(df) * 100:.1f}%)")

    print()
    print(sep)
    print("TOP 10 MOST FAILING TESTS")
    print(sep)
    fail_rates = (
        df.groupby("test_id")["outcome"]
        .agg(["sum", "count"])
        .assign(fail_rate=lambda x: x["sum"] / x["count"])
        .sort_values("fail_rate", ascending=False)
        .head(10)
    )
    print(fail_rates.to_string())

    print()
    print(sep)
    print("TEST DURATION STATS (seconds)")
    print(sep)
    print(df["duration_s"].describe().round(3).to_string())

    print()
    print(sep)
    print("COMMIT-LEVEL CHANGE SIZE STATS")
    print(sep)
    if commits is not None:
        for col in ["files_changed", "lines_added", "lines_deleted"]:
            print(f"\n  {col}:")
            print(f"    {commits[col].describe().round(1).to_string()}")

    print()
    print(sep)
    print("TRAIN / TEST SPLIT (80/20 chronological)")
    print(sep)
    df_sorted = df.sort_values("commit_ts")
    cutoff = int(len(df_sorted) * 0.8)
    train = df_sorted.iloc[:cutoff]
    test = df_sorted.iloc[cutoff:]
    print(f"  Train rows : {len(train):,}  (up to {train['commit_ts'].max()})")
    print(f"  Test rows  : {len(test):,}  (from {test['commit_ts'].min()})")
    print(f"  Train failure rate : {train['outcome'].mean() * 100:.2f}%")
    print(f"  Test failure rate  : {test['outcome'].mean() * 100:.2f}%")

    print()
    print(sep)
    print("KNOWN LIMITATIONS")
    print(sep)
    for i, lim in enumerate(summary.get("known_limitations", []), 1):
        print(f"  {i}. {lim}")

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect collected test dataset")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    report(Path(args.data_dir))


if __name__ == "__main__":
    main()
