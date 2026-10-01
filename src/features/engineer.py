"""
src/features/engineer.py
=========================
Feature engineering for ML test prioritization.

Design principle — NO data leakage:
  All lookups (test failure rates, file-test co-failure rates, author rates)
  are fitted ONLY on the training split, then applied to both train and test.

Features produced (per test × commit row):
  History features (from prev_N columns):
    hist_fail_rate      — fraction of last N runs that failed
    hist_last_failed    — 1 if the most recent run failed, else 0
    hist_weighted_score — recency-weighted failure score Σ prev_i / i
    hist_consecutive_passes — # consecutive passes before this run

  Test-level aggregate (fitted on train):
    test_global_fail_rate   — P(fail) for this test across all training commits
    test_is_high_risk       — 1 if test_global_fail_rate > median

  Commit-level change features:
    files_changed, lines_added, lines_deleted, change_size
    test_files_changed, src_files_changed

  Co-failure features (fitted on train):
    cochange_score  — mean failure rate of this test on commits that touched
                      at least one file also changed in THIS commit

  Author features (fitted on train):
    author_fail_rate — mean failure rate for this author's commits

  Test characteristics:
    duration_s        — raw test duration
    duration_log1p    — log1p(duration_s), reduces right-skew
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # History
    "hist_fail_rate",
    "hist_last_failed",
    "hist_weighted_score",
    "hist_consecutive_passes",
    # Test-level aggregates
    "test_global_fail_rate",
    "test_is_high_risk",
    # Commit change features
    "files_changed",
    "lines_added",
    "lines_deleted",
    "change_size",
    "test_files_changed",
    "src_files_changed",
    # Co-failure
    "cochange_score",
    # Author
    "author_fail_rate",
    # Duration
    "duration_s",
    "duration_log1p",
]


class FeatureEngineer:
    """
    Fit on training data → transform any split without leaking future info.

    Usage:
        fe = FeatureEngineer(history_window=10)
        X_train = fe.fit_transform(train_df)   # fits lookups, returns features
        X_test  = fe.transform(test_df)        # applies fitted lookups
    """

    def __init__(self, history_window: int = 10) -> None:
        self.history_window = history_window
        self._fitted = False

        # Lookup tables computed from training data
        self._test_fail_rate: dict[str, float] = {}
        self._author_fail_rate: dict[str, float] = {}
        self._global_fail_rate: float = 0.0
        self._high_risk_threshold: float = 0.0

        # file → {test_id: fail_rate on commits where file changed}
        self._file_test_fail_rate: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> FeatureEngineer:
        """Compute all lookup tables from the training split."""
        log.info("Fitting feature engineer on %d rows...", len(train_df))

        # 1. Per-test global failure rate
        test_stats = train_df.groupby("test_id")["outcome"].agg(["mean", "count"])
        self._test_fail_rate = test_stats["mean"].to_dict()
        self._global_fail_rate = float(train_df["outcome"].mean())

        # 2. High-risk threshold (median failure rate)
        rates = np.array(list(self._test_fail_rate.values()))
        self._high_risk_threshold = float(np.median(rates)) if len(rates) else 0.0

        # 3. Author failure rate
        author_stats = train_df.groupby("author")["outcome"].mean()
        self._author_fail_rate = author_stats.to_dict()

        # 4. File → test co-failure rate
        #    For each commit, parse changed_file_list, then for each (file, test)
        #    pair record whether the test failed.
        self._file_test_fail_rate = self._compute_cochange_rates(train_df)

        self._fitted = True
        log.info(
            "Fitted: %d test rates, %d author rates, %d file->test mappings",
            len(self._test_fail_rate),
            len(self._author_fail_rate),
            len(self._file_test_fail_rate),
        )
        return self

    def _compute_cochange_rates(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        For each source file, compute the per-test failure rate on commits
        where that file was changed.
        Returns: {file_path: {test_id: mean_failure_rate}}
        """
        # Parse changed_file_list once per commit
        commit_files: dict[str, list[str]] = {}
        for _, row in (
            df[["commit_sha", "changed_file_list"]].drop_duplicates("commit_sha").iterrows()
        ):
            try:
                commit_files[row["commit_sha"]] = json.loads(row["changed_file_list"])
            except (json.JSONDecodeError, TypeError):
                commit_files[row["commit_sha"]] = []

        # Build (file, test_id, outcome) triples
        records = []
        for _, row in df[["commit_sha", "test_id", "outcome"]].iterrows():
            sha = row["commit_sha"]
            for fpath in commit_files.get(sha, []):
                records.append((fpath, row["test_id"], row["outcome"]))

        if not records:
            return {}

        co_df = pd.DataFrame(records, columns=["file", "test_id", "outcome"])
        grouped = co_df.groupby(["file", "test_id"])["outcome"].mean()

        result: dict[str, dict[str, float]] = {}
        for (file, test_id), rate in grouped.items():
            result.setdefault(file, {})[test_id] = float(rate)
        return result

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted lookups to produce a feature matrix."""
        if not self._fitted:
            raise RuntimeError("Call .fit() or .fit_transform() first.")

        prev_cols = sorted(
            [c for c in df.columns if c.startswith("prev_")],
            key=lambda c: int(c.split("_")[1]),
        )

        rows = []
        for _, row in df.iterrows():
            rows.append(self._row_features(row, prev_cols))

        feature_df = pd.DataFrame(rows, index=df.index)
        return feature_df[FEATURE_COLUMNS].astype(float)

    def _row_features(self, row: pd.Series, prev_cols: list[str]) -> dict:
        # ---- History features ----
        prev_vals = []
        for col in prev_cols:
            v = row.get(col)
            if pd.notna(v):
                prev_vals.append(float(v))

        if prev_vals:
            hist_fail_rate = sum(prev_vals) / len(prev_vals)
            hist_last_failed = prev_vals[0]
            weighted = sum(v / (i + 1) for i, v in enumerate(prev_vals))
            hist_weighted_score = weighted

            # Consecutive passes: scan from most recent until a failure
            consec = 0
            for v in prev_vals:
                if v == 0.0:
                    consec += 1
                else:
                    break
            hist_consecutive_passes = float(consec)
        else:
            hist_fail_rate = self._global_fail_rate
            hist_last_failed = 0.0
            hist_weighted_score = 0.0
            hist_consecutive_passes = float(self.history_window)

        # ---- Test-level aggregates ----
        tid = row["test_id"]
        test_global_fail_rate = self._test_fail_rate.get(tid, self._global_fail_rate)
        test_is_high_risk = float(test_global_fail_rate > self._high_risk_threshold)

        # ---- Commit change features ----
        files_changed = float(row.get("files_changed", 0) or 0)
        lines_added = float(row.get("lines_added", 0) or 0)
        lines_deleted = float(row.get("lines_deleted", 0) or 0)
        change_size = lines_added + lines_deleted
        test_files_changed = float(row.get("test_files_changed", 0) or 0)
        src_files_changed = float(row.get("src_files_changed", 0) or 0)

        # ---- Co-failure score ----
        try:
            changed_files = json.loads(row.get("changed_file_list") or "[]")
        except (json.JSONDecodeError, TypeError):
            changed_files = []

        cochange_scores = []
        for fpath in changed_files:
            file_map = self._file_test_fail_rate.get(fpath, {})
            if tid in file_map:
                cochange_scores.append(file_map[tid])
        cochange_score = float(max(cochange_scores)) if cochange_scores else 0.0

        # ---- Author feature ----
        author = str(row.get("author", ""))
        author_fail_rate = self._author_fail_rate.get(author, self._global_fail_rate)

        # ---- Duration ----
        duration_s = float(row.get("duration_s", 0.0) or 0.0)
        duration_log1p = math.log1p(duration_s)

        return {
            "hist_fail_rate": hist_fail_rate,
            "hist_last_failed": hist_last_failed,
            "hist_weighted_score": hist_weighted_score,
            "hist_consecutive_passes": hist_consecutive_passes,
            "test_global_fail_rate": test_global_fail_rate,
            "test_is_high_risk": test_is_high_risk,
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "change_size": change_size,
            "test_files_changed": test_files_changed,
            "src_files_changed": src_files_changed,
            "cochange_score": cochange_score,
            "author_fail_rate": author_fail_rate,
            "duration_s": duration_s,
            "duration_log1p": duration_log1p,
        }

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------------
    # Single-commit inference (used by MLOrderer at evaluation time)
    # ------------------------------------------------------------------

    def transform_commit(
        self,
        commit_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a single commit's rows for real-time inference.
        Identical to transform() but accepts a commit-level DataFrame slice.
        """
        return self.transform(commit_df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("FeatureEngineer saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> FeatureEngineer:
        import pickle

        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        log.info("FeatureEngineer loaded from %s", path)
        return obj
