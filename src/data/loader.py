"""
src/data/loader.py
==================
Load, validate, and split the collected test-execution dataset.

Provides:
  - DatasetLoader: loads parquet, validates schema, returns train/test DataFrames
  - CommitGroup: named tuple for per-commit evaluation context
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Columns guaranteed to be present after Phase 1 collection
REQUIRED_COLUMNS = {
    "commit_sha",
    "commit_ts",
    "author",
    "files_changed",
    "lines_added",
    "lines_deleted",
    "test_files_changed",
    "src_files_changed",
    "changed_file_list",
    "test_id",
    "outcome",
    "duration_s",
}

HISTORY_COLUMN_PATTERN = "prev_"


@dataclass
class CommitGroup:
    """All information available at evaluation time for a single commit."""

    commit_sha: str
    commit_ts: pd.Timestamp
    changed_files: list[str]
    # Per-test rows for this commit (includes prev_N columns, duration_s)
    tests: pd.DataFrame
    # Ground-truth labels: test_id → outcome (0/1)
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def failing_tests(self) -> set[str]:
        return {tid for tid, out in self.labels.items() if out == 1}

    @property
    def has_failures(self) -> bool:
        return bool(self.failing_tests)


class DatasetLoader:
    """Load and validate the Phase 1 dataset."""

    def __init__(self, data_dir: str | Path = "data/raw") -> None:
        self.data_dir = Path(data_dir)
        self._exec_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load test_executions.parquet and validate schema."""
        path = self.data_dir / "test_executions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/collect_data.py first.")
        df = pd.read_parquet(path)
        self._validate(df)
        df["commit_ts"] = pd.to_datetime(df["commit_ts"], utc=True)
        df = df.sort_values(["commit_ts", "test_id"]).reset_index(drop=True)
        self._exec_df = df
        log.info(
            "Loaded %d rows | %d tests | %d commits | failure rate %.2f%%",
            len(df),
            df["test_id"].nunique(),
            df["commit_sha"].nunique(),
            df["outcome"].mean() * 100,
        )
        return df

    def train_test_split(
        self,
        df: pd.DataFrame | None = None,
        train_ratio: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological 80/20 split.

        IMPORTANT: We split on COMMIT timestamps, not rows, to avoid a single
        commit straddling both sets.
        """
        df = df if df is not None else self._exec_df
        if df is None:
            raise RuntimeError("Call .load() first.")

        # Get unique commits in chronological order
        commit_ts = (
            df[["commit_sha", "commit_ts"]].drop_duplicates("commit_sha").sort_values("commit_ts")
        )
        cutoff_idx = int(len(commit_ts) * train_ratio)
        cutoff_ts = commit_ts.iloc[cutoff_idx]["commit_ts"]

        train = df[df["commit_ts"] < cutoff_ts]
        test = df[df["commit_ts"] >= cutoff_ts]

        log.info(
            "Split: train=%d rows (%d commits) | test=%d rows (%d commits) | cutoff=%s",
            len(train),
            train["commit_sha"].nunique(),
            len(test),
            test["commit_sha"].nunique(),
            cutoff_ts.date(),
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def iter_commits(
        self,
        df: pd.DataFrame,
        failures_only: bool = True,
    ) -> Generator[CommitGroup, None, None]:
        """
        Iterate over commits in chronological order, yielding CommitGroup objects.

        Args:
            df: DataFrame slice (e.g., test split)
            failures_only: if True, skip commits with no failing tests
        """
        for sha, group in df.groupby("commit_sha", sort=False):
            group = group.sort_values("commit_ts")
            labels = dict(zip(group["test_id"], group["outcome"], strict=False))

            if failures_only and not any(v == 1 for v in labels.values()):
                continue

            # Parse changed files (JSON list stored as string)
            try:
                changed_files = json.loads(group.iloc[0]["changed_file_list"])
            except (json.JSONDecodeError, KeyError):
                changed_files = []

            yield CommitGroup(
                commit_sha=str(sha),
                commit_ts=group["commit_ts"].iloc[0],
                changed_files=changed_files,
                tests=group.reset_index(drop=True),
                labels=labels,
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if df.empty:
            raise ValueError("Dataset is empty.")
        if not df["outcome"].isin({0, 1}).all():
            raise ValueError(
                "Column 'outcome' must contain only 0 or 1 (skipped tests should be removed)."
            )
        log.debug("Schema validation passed.")
