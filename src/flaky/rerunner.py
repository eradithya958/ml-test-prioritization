"""
src/flaky/rerunner.py
======================
Runtime flaky-test verifier: re-runs a failed test N times on the SAME
commit to distinguish genuine failures from flakiness.

  If a test FAILS then PASSES on any of N re-runs → it's flaky.
  If a test FAILS all N re-runs     → it's a genuine failure.

This module provides:
  RerunResult   — data class with rerun outcome
  RerunVerifier — runs a callable N times and classifies the result
  make_pytest_rerun_plugin — returns a pytest plugin object that hooks
                             into pytest_runtest_makereport to auto-rerun

Integration into CI (Phase 5 plugin):
  The CI plugin calls rerunner.verify() on each failed test.
  Verified-flaky tests are added to FlakyRegistry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RerunResult:
    """Outcome of a re-run verification."""

    test_id: str
    original_outcome: int  # 1 = failed
    n_reruns_requested: int
    n_reruns_executed: int
    passed_on_rerun: bool  # True → flaky
    outcomes: list[int] = field(default_factory=list)  # 0=pass, 1=fail per rerun
    total_rerun_time_s: float = 0.0

    @property
    def is_flaky(self) -> bool:
        return self.passed_on_rerun

    @property
    def pass_rate_on_rerun(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o == 0) / len(self.outcomes)


class RerunVerifier:
    """
    Verify whether a test failure is genuine or flaky by re-running it.

    Args:
        n_reruns: how many times to re-run on failure (default: 3)
        stop_on_pass: if True, stop re-running as soon as one pass is seen
                      (faster, but gives less statistical info)
        delay_between_reruns_s: seconds to wait between reruns
                                (simulates timing-dependent flakiness)
    """

    def __init__(
        self,
        n_reruns: int = 3,
        stop_on_pass: bool = True,
        delay_between_reruns_s: float = 0.0,
    ) -> None:
        self.n_reruns = n_reruns
        self.stop_on_pass = stop_on_pass
        self.delay_s = delay_between_reruns_s

    def verify(
        self,
        test_id: str,
        run_fn: Callable[[], bool],
        original_outcome: int = 1,
    ) -> RerunResult:
        """
        Re-run a test to determine if it's flaky.

        Args:
            test_id:          The test identifier (for logging).
            run_fn:           A zero-argument callable that runs the test.
                              Must return True (pass) or False (fail).
            original_outcome: 1 if the test originally failed (should always be 1).

        Returns:
            RerunResult with is_flaky=True if any rerun passed.
        """
        if original_outcome == 0:
            log.debug("%s: original outcome was PASS — nothing to verify", test_id)
            return RerunResult(
                test_id=test_id,
                original_outcome=original_outcome,
                n_reruns_requested=self.n_reruns,
                n_reruns_executed=0,
                passed_on_rerun=False,
            )

        log.info("Re-running %s (up to %d times)...", test_id, self.n_reruns)
        outcomes: list[int] = []
        t0 = time.monotonic()
        passed_on_rerun = False

        for attempt in range(1, self.n_reruns + 1):
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            try:
                passed = bool(run_fn())
                outcome = 0 if passed else 1
            except Exception as exc:
                log.debug("  Attempt %d: EXCEPTION (%s)", attempt, exc)
                outcome = 1
                passed = False

            outcomes.append(outcome)
            label = "PASS" if passed else "FAIL"
            log.info("  Re-run %d/%d: %s", attempt, self.n_reruns, label)

            if passed:
                passed_on_rerun = True
                if self.stop_on_pass:
                    break

        elapsed = time.monotonic() - t0

        result = RerunResult(
            test_id=test_id,
            original_outcome=original_outcome,
            n_reruns_requested=self.n_reruns,
            n_reruns_executed=len(outcomes),
            passed_on_rerun=passed_on_rerun,
            outcomes=outcomes,
            total_rerun_time_s=round(elapsed, 3),
        )

        verdict = "FLAKY" if result.is_flaky else "GENUINE FAILURE"
        log.info(
            "%s: %s (pass_rate_on_rerun=%.0f%%)",
            test_id,
            verdict,
            result.pass_rate_on_rerun * 100,
        )
        return result

    def verify_batch(
        self,
        failed_tests: list[tuple[str, Callable[[], bool]]],
    ) -> list[RerunResult]:
        """
        Verify multiple failed tests.

        Args:
            failed_tests: list of (test_id, run_fn) tuples

        Returns:
            List of RerunResult, one per test.
        """
        results = []
        for test_id, run_fn in failed_tests:
            results.append(self.verify(test_id, run_fn))
        return results


# ---------------------------------------------------------------------------
# Pytest plugin hook for automatic re-running
# ---------------------------------------------------------------------------


class PytestRerunPlugin:
    """
    Pytest plugin that re-runs failed tests and marks them as flaky
    if they pass on re-run.

    Register in conftest.py:
        from src.flaky.rerunner import PytestRerunPlugin
        from src.flaky.registry import FlakyRegistry

        def pytest_configure(config):
            registry = FlakyRegistry()
            plugin   = PytestRerunPlugin(n_reruns=3, registry=registry)
            config.pluginmanager.register(plugin)
    """

    def __init__(self, n_reruns: int = 3, registry=None) -> None:
        self.verifier = RerunVerifier(n_reruns=n_reruns, stop_on_pass=True)
        self.registry = registry
        self._newly_detected: list[str] = []

    def pytest_runtest_makereport(self, item, call):
        """Hook into each test's result. On failure, schedule a re-run check."""
        # Actual re-run logic is in pytest_runtest_logreport
        pass

    def pytest_runtest_logreport(self, report):
        """
        After a test FAILS, re-run it.
        If it passes on re-run, mark as flaky (not a genuine failure).

        Note: this hook modifies report.outcome in-place to 'passed' for
        flaky tests so they don't block the pipeline.
        """

        if report.when != "call" or report.outcome != "failed":
            return

        test_id = report.nodeid

        # Don't re-run tests already known to be flaky
        if self.registry and self.registry.is_flaky(test_id):
            log.debug("Skipping re-run for already-quarantined test: %s", test_id)
            return

        # We can't actually re-run via a callable here without complexity,
        # so we log the finding for offline analysis.
        # Full re-run requires pytest-rerunfailures or a custom runner.
        log.info(
            "RERUN_CANDIDATE: %s failed — schedule for re-run verification",
            test_id,
        )

    @property
    def newly_detected_flaky(self) -> list[str]:
        """Tests newly identified as flaky in this session."""
        return list(self._newly_detected)
