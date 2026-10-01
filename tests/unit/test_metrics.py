"""
Unit tests for src/evaluation/metrics.py

Tests every metric function with handcrafted examples where the
correct answer is known analytically.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    aggregate_metrics,
    apfd,
    pct_tests_to_catch_all,
    runtime_saved_pct,
    time_to_first_failure,
)

# ---------------------------------------------------------------------------
# APFD tests
# ---------------------------------------------------------------------------


class TestAPFD:
    def test_perfect_ordering_gives_high_apfd(self):
        """Failing tests first → APFD close to 1."""
        tests = ["fail_1", "fail_2", "pass_1", "pass_2", "pass_3"]
        failing = {"fail_1", "fail_2"}
        score = apfd(tests, failing)
        # Positions: fail_1=1, fail_2=2 → APFD = 1 - (1+2)/(5*2) + 1/(2*5)
        expected = 1 - 3 / 10 + 0.1
        assert abs(score - expected) < 1e-9

    def test_worst_ordering_gives_low_apfd(self):
        """Failing tests last → APFD close to 0."""
        tests = ["pass_1", "pass_2", "pass_3", "fail_1", "fail_2"]
        failing = {"fail_1", "fail_2"}
        score = apfd(tests, failing)
        # Positions: fail_1=4, fail_2=5 → APFD = 1 - (4+5)/(5*2) + 1/(2*5)
        expected = 1 - 9 / 10 + 0.1
        assert abs(score - expected) < 1e-9
        assert score < 0.3

    def test_random_ordering_apfd_near_half(self):
        """Over many shuffles, random APFD ≈ 0.5."""
        import random

        rng = random.Random(0)
        tests = [f"t{i}" for i in range(100)]
        failing = {f"t{i}" for i in range(10)}  # 10% failure rate
        scores = []
        for _ in range(500):
            shuffled = list(tests)
            rng.shuffle(shuffled)
            scores.append(apfd(shuffled, failing))
        mean_apfd = sum(scores) / len(scores)
        assert abs(mean_apfd - 0.5) < 0.05

    def test_no_failures_returns_nan(self):
        score = apfd(["t1", "t2", "t3"], set())
        assert math.isnan(score)

    def test_empty_test_list_returns_nan(self):
        score = apfd([], {"t1"})
        assert math.isnan(score)

    def test_single_failing_test_first(self):
        tests = ["fail", "pass1", "pass2"]
        failing = {"fail"}
        score = apfd(tests, failing)
        # TF = 1, n=3, m=1 → 1 - 1/3 + 1/6
        expected = 1 - 1 / 3 + 1 / 6
        assert abs(score - expected) < 1e-9

    def test_single_failing_test_last(self):
        tests = ["pass1", "pass2", "fail"]
        failing = {"fail"}
        score = apfd(tests, failing)
        # TF = 3, n=3, m=1 → 1 - 3/3 + 1/6
        expected = 1 - 1 + 1 / 6
        assert abs(score - expected) < 1e-9

    def test_all_tests_failing(self):
        tests = ["t1", "t2", "t3"]
        failing = {"t1", "t2", "t3"}
        score = apfd(tests, failing)
        # Positions: 1+2+3=6, n=3, m=3 → 1 - 6/9 + 1/6
        expected = 1 - 6 / 9 + 1 / 6
        assert abs(score - expected) < 1e-9

    def test_apfd_range_is_zero_to_one(self):
        tests = [f"t{i}" for i in range(20)]
        failing = {"t5", "t10", "t15"}
        score = apfd(tests, failing)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Time to First Failure tests
# ---------------------------------------------------------------------------


class TestTimeToFirstFailure:
    def test_first_test_fails_immediately(self):
        tests = ["fail", "pass1", "pass2"]
        failing = {"fail"}
        durations = {"fail": 5.0, "pass1": 3.0, "pass2": 2.0}
        result = time_to_first_failure(tests, failing, durations)
        assert result == pytest.approx(5.0)

    def test_last_test_fails_full_duration(self):
        tests = ["pass1", "pass2", "fail"]
        failing = {"fail"}
        durations = {"pass1": 2.0, "pass2": 3.0, "fail": 1.0}
        result = time_to_first_failure(tests, failing, durations)
        assert result == pytest.approx(6.0)  # 2+3+1

    def test_no_failure_returns_inf(self):
        tests = ["pass1", "pass2"]
        failing = {"missing_test"}
        durations = {"pass1": 1.0, "pass2": 2.0}
        result = time_to_first_failure(tests, failing, durations)
        assert math.isinf(result)

    def test_missing_duration_treated_as_zero(self):
        tests = ["fail", "pass1"]
        failing = {"fail"}
        durations = {}  # no durations
        result = time_to_first_failure(tests, failing, durations)
        assert result == pytest.approx(0.0)

    def test_multiple_failures_stops_at_first(self):
        tests = ["pass", "fail_a", "fail_b"]
        failing = {"fail_a", "fail_b"}
        durations = {"pass": 10.0, "fail_a": 2.0, "fail_b": 1.0}
        result = time_to_first_failure(tests, failing, durations)
        assert result == pytest.approx(12.0)  # stops at fail_a


# ---------------------------------------------------------------------------
# Pct Tests to Catch All tests
# ---------------------------------------------------------------------------


class TestPctTestsToCatchAll:
    def test_all_failures_first_returns_fraction(self):
        tests = ["f1", "f2", "p1", "p2", "p3"]
        failing = {"f1", "f2"}
        result = pct_tests_to_catch_all(tests, failing)
        assert result == pytest.approx(2 / 5)

    def test_failures_at_end_returns_one(self):
        tests = ["p1", "p2", "p3", "f1"]
        failing = {"f1"}
        result = pct_tests_to_catch_all(tests, failing)
        assert result == pytest.approx(1.0)

    def test_no_failures_returns_nan(self):
        result = pct_tests_to_catch_all(["t1", "t2"], set())
        assert math.isnan(result)

    def test_empty_suite_returns_nan(self):
        result = pct_tests_to_catch_all([], {"f1"})
        assert math.isnan(result)

    def test_scattered_failures(self):
        # failures at positions 2 and 5 out of 5 → need 5/5 = 1.0
        tests = ["p1", "f1", "p2", "p3", "f2"]
        failing = {"f1", "f2"}
        result = pct_tests_to_catch_all(tests, failing)
        assert result == pytest.approx(5 / 5)


# ---------------------------------------------------------------------------
# Runtime Saved tests
# ---------------------------------------------------------------------------


class TestRuntimeSavedPct:
    def test_failures_first_saves_most_time(self):
        tests = ["f1", "f2", "p1", "p2", "p3"]
        failing = {"f1", "f2"}
        durations = {"f1": 1.0, "f2": 1.0, "p1": 5.0, "p2": 5.0, "p3": 5.0}
        result = runtime_saved_pct(tests, failing, durations)
        # total = 17, at_last_failure (f2 at pos 2) = 2, saved = 15/17
        assert result == pytest.approx(15 / 17, rel=1e-4)

    def test_failures_last_saves_nothing(self):
        tests = ["p1", "p2", "f1"]
        failing = {"f1"}
        durations = {"p1": 5.0, "p2": 5.0, "f1": 1.0}
        result = runtime_saved_pct(tests, failing, durations)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_no_failures_returns_nan(self):
        result = runtime_saved_pct(["t1"], set(), {"t1": 1.0})
        assert math.isnan(result)

    def test_zero_total_runtime_returns_nan(self):
        result = runtime_saved_pct(["f1"], {"f1"}, {"f1": 0.0})
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Aggregate metrics tests
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_mean_and_std_computed(self):
        data = [{"apfd": 0.6, "ttff_s": 10.0}, {"apfd": 0.8, "ttff_s": 20.0}]
        result = aggregate_metrics(data)
        assert result["apfd_mean"] == pytest.approx(0.7)
        assert result["ttff_s_mean"] == pytest.approx(15.0)

    def test_single_item_std_is_zero(self):
        data = [{"apfd": 0.75}]
        result = aggregate_metrics(data)
        assert result["apfd_mean"] == pytest.approx(0.75)
        assert result["apfd_std"] == pytest.approx(0.0)

    def test_empty_list_returns_empty_dict(self):
        result = aggregate_metrics([])
        assert result == {}

    def test_nan_values_are_excluded(self):
        data = [
            {"apfd": 0.6},
            {"apfd": float("nan")},
            {"apfd": 0.8},
        ]
        result = aggregate_metrics(data)
        # nan row excluded → mean of 0.6 and 0.8
        assert result["apfd_mean"] == pytest.approx(0.7)
        assert result["apfd_n"] == 2
