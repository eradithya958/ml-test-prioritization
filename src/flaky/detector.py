"""
src/flaky/detector.py
======================
Flaky test detection via historical pass/fail pattern analysis.

Two complementary detection signals (combined into a single flakiness score):

  1. Flip-Without-Change Rate (flip_rate)
     A test is flaky if it FAILS at commit T then PASSES at commit T+1
     without any relevant source file changes between the two commits.
     flip_rate = n_flips / n_failures  (higher = more flaky)

  2. Failure-Rate-Without-Code-Change (fail_rate_no_change)
     A test fails on commits where ZERO source files were changed.
     If a test fails when nothing relevant changed, it's not reacting to
     real code faults — it's flaky.
     fail_rate_no_change = n_fails_on_unchanged / n_runs_on_unchanged

  Combined score:
     flaky_score = 0.5 * flip_rate + 0.5 * fail_rate_no_change

  A test is quarantined when flaky_score >= threshold (default: 0.10)
  AND it has been run at least min_runs times (default: 10).

References:
  Luo et al., "An empirical analysis of flaky tests", FSE 2014.
  Eck et al., "Understanding flaky tests: The developer's perspective", FSE 2019.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FlakyTestMetrics:
    """Per-test flakiness metrics."""

    test_id: str
    n_runs: int
    n_failures: int
    failure_rate: float
    flip_rate: float  # flips-without-change / failures
    fail_rate_no_change: float  # failure rate on unchanged commits
    n_runs_no_change: int  # number of runs on unchanged commits
    n_flips: int  # raw flip count
    flaky_score: float  # combined score [0, 1]
    is_flaky: bool  # True if above threshold
    ground_truth_flaky: bool = False  # set if synthetic ground-truth is available


@dataclass
class DetectionReport:
    """Summary of a flaky detection run."""

    total_tests_analyzed: int
    n_flaky_detected: int
    n_not_flaky: int
    threshold: float
    # Precision / recall (only computable when ground truth is available)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    # Reduction in false-failure builds
    false_failure_builds_total: int = 0
    false_failure_builds_after_quarantine: int = 0
    per_test: list[FlakyTestMetrics] = field(default_factory=list)

    def summary_dict(self) -> dict:
        d = {
            "total_tests_analyzed": self.total_tests_analyzed,
            "n_flaky_detected": self.n_flaky_detected,
            "n_not_flaky": self.n_not_flaky,
            "threshold": self.threshold,
            "false_failure_builds_total": self.false_failure_builds_total,
            "false_failure_builds_after_quarantine": self.false_failure_builds_after_quarantine,
            "builds_saved": self.false_failure_builds_total
            - self.false_failure_builds_after_quarantine,
        }
        if self.precision is not None:
            d.update({"precision": self.precision, "recall": self.recall, "f1": self.f1})
        return d


class FlakyDetector:
    """
    Analyze historical test execution data to identify flaky tests.

    Usage:
        detector = FlakyDetector(threshold=0.10, min_runs=10)
        report   = detector.analyze(df)
    """

    def __init__(
        self,
        threshold: float = 0.10,
        min_runs: int = 10,
        flip_weight: float = 0.50,
        no_change_weight: float = 0.50,
    ) -> None:
        self.threshold = threshold
        self.min_runs = min_runs
        self.flip_weight = flip_weight
        self.no_change_weight = no_change_weight

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> DetectionReport:
        """
        Analyze all tests in df and return a DetectionReport.

        Args:
            df: test_executions DataFrame (from Phase 1 collection or synthetic).
                Must have columns: test_id, commit_sha, commit_ts, outcome,
                src_files_changed, changed_file_list.
                Optionally: is_flaky_ground_truth (bool) for synthetic data.
        """
        df = df.copy()
        if "commit_ts" in df.columns:
            df["commit_ts"] = pd.to_datetime(df["commit_ts"], utc=True)
        else:
            df["commit_ts"] = pd.date_range("2021-01-01", periods=len(df), freq="h")

        has_ground_truth = "is_flaky_ground_truth" in df.columns
        log.info(
            "Analyzing %d test executions | %d unique tests | ground truth: %s",
            len(df),
            df["test_id"].nunique(),
            has_ground_truth,
        )

        per_test_metrics: list[FlakyTestMetrics] = []

        for test_id, group in df.groupby("test_id"):
            metrics = self._analyze_test(str(test_id), group)
            if has_ground_truth:
                metrics.ground_truth_flaky = bool(group["is_flaky_ground_truth"].any())
            per_test_metrics.append(metrics)

        # Sort by flaky_score descending for easy inspection
        per_test_metrics.sort(key=lambda m: -m.flaky_score)

        detected = [m for m in per_test_metrics if m.is_flaky]

        # Compute false-failure build reduction
        false_builds_total, false_builds_after = self._compute_build_reduction(
            df, {m.test_id for m in detected}
        )

        # Precision / recall if ground truth available
        precision = recall = f1 = None
        if has_ground_truth:
            precision, recall, f1 = self._compute_pr(per_test_metrics)

        report = DetectionReport(
            total_tests_analyzed=len(per_test_metrics),
            n_flaky_detected=len(detected),
            n_not_flaky=len(per_test_metrics) - len(detected),
            threshold=self.threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            false_failure_builds_total=false_builds_total,
            false_failure_builds_after_quarantine=false_builds_after,
            per_test=per_test_metrics,
        )

        log.info(
            "Detected %d flaky tests out of %d (threshold=%.2f)",
            len(detected),
            len(per_test_metrics),
            self.threshold,
        )
        if precision is not None:
            log.info("Precision=%.3f  Recall=%.3f  F1=%.3f", precision, recall, f1)
        log.info(
            "False-failure builds: %d total → %d after quarantine (%d saved)",
            false_builds_total,
            false_builds_after,
            false_builds_total - false_builds_after,
        )

        return report

    # ------------------------------------------------------------------
    # Per-test analysis
    # ------------------------------------------------------------------

    def _analyze_test(self, test_id: str, group: pd.DataFrame) -> FlakyTestMetrics:
        group = group.sort_values("commit_ts").reset_index(drop=True)
        n_runs = len(group)
        n_failures = int((group["outcome"] == 1).sum())
        failure_rate = n_failures / n_runs if n_runs else 0.0

        # ------ Signal 1: flip-without-change ------
        n_flips = self._count_flips_without_change(group)
        flip_rate = n_flips / n_failures if n_failures > 0 else 0.0
        flip_rate = min(flip_rate, 1.0)

        # ------ Signal 2: failure on unchanged commits ------
        no_change_mask = group["src_files_changed"] == 0
        no_change_group = group[no_change_mask]
        n_runs_nc = len(no_change_group)
        if n_runs_nc >= 3:
            fail_rate_nc = float((no_change_group["outcome"] == 1).mean())
        else:
            # Not enough data — use a moderate prior
            fail_rate_nc = failure_rate * 0.5

        # ------ Combined score ------
        flaky_score = (
            self.flip_weight * flip_rate + self.no_change_weight * min(fail_rate_nc * 5.0, 1.0)
            # scale fail_rate_nc: a 20% base failure rate → score 1.0
        )
        flaky_score = min(flaky_score, 1.0)
        is_flaky = (flaky_score >= self.threshold) and (n_runs >= self.min_runs)

        return FlakyTestMetrics(
            test_id=test_id,
            n_runs=n_runs,
            n_failures=n_failures,
            failure_rate=round(failure_rate, 4),
            flip_rate=round(flip_rate, 4),
            fail_rate_no_change=round(fail_rate_nc, 4),
            n_runs_no_change=n_runs_nc,
            n_flips=n_flips,
            flaky_score=round(flaky_score, 4),
            is_flaky=is_flaky,
        )

    def _count_flips_without_change(self, group: pd.DataFrame) -> int:
        """
        Count (fail → pass) transitions where src_files_changed == 0
        in the *next* commit (meaning nothing relevant changed).
        """
        n_flips = 0
        outcomes = group["outcome"].tolist()
        src_changes = group["src_files_changed"].tolist()

        for i in range(len(outcomes) - 1):
            if outcomes[i] == 1 and outcomes[i + 1] == 0:
                # The next commit had no source changes — spontaneous flip
                if src_changes[i + 1] == 0:
                    n_flips += 1
        return n_flips

    # ------------------------------------------------------------------
    # Build-level false-failure reduction
    # ------------------------------------------------------------------

    def _compute_build_reduction(
        self,
        df: pd.DataFrame,
        quarantined_tests: set[str],
    ) -> tuple[int, int]:
        """
        Count how many builds (commits) had ALL their failures come from
        flaky tests (false-failure builds) — these would be unblocked by quarantine.

        Returns:
            (n_false_builds_before, n_false_builds_after_quarantine)
        """
        # Group by commit: find commits where at least one test failed
        failing_commits = df[df["outcome"] == 1].groupby("commit_sha")["test_id"].apply(set)

        n_false_builds = 0
        for _sha, failing_tests in failing_commits.items():
            # If ALL failures are from quarantined (flaky) tests → false-failure build
            if failing_tests.issubset(quarantined_tests):
                n_false_builds += 1

        total_failing_builds = len(failing_commits)
        builds_after = total_failing_builds - n_false_builds
        return total_failing_builds, builds_after

    # ------------------------------------------------------------------
    # Precision / recall
    # ------------------------------------------------------------------

    def _compute_pr(self, metrics: list[FlakyTestMetrics]) -> tuple[float, float, float]:
        tp = sum(1 for m in metrics if m.is_flaky and m.ground_truth_flaky)
        fp = sum(1 for m in metrics if m.is_flaky and not m.ground_truth_flaky)
        fn = sum(1 for m in metrics if not m.is_flaky and m.ground_truth_flaky)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 3), round(recall, 3), round(f1, 3)
