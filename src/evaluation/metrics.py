"""
src/evaluation/metrics.py
==========================
Evaluation metrics for test case prioritization.

Metrics implemented
-------------------
- APFD  (Average Percentage of Faults Detected)
- time_to_first_failure   — cumulative test duration until first failing test
- pct_tests_to_catch_all  — fraction of suite run before all failures seen
- runtime_saved_pct       — runtime saved vs. running the full suite
- apfd_c                  — APFD weighted by fault severity (future extension stub)

All functions are pure Python / numpy — no side effects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def apfd(
    ordered_test_ids: Sequence[str],
    failing_test_ids: set[str],
) -> float:
    """
    Compute APFD (Average Percentage of Faults Detected).

    Formula:
        APFD = 1 - (TF_1 + TF_2 + ... + TF_m) / (n * m) + 1 / (2 * n)

    Where:
        n = total number of tests
        m = number of faults (unique failing tests)
        TF_i = 1-indexed position of the i-th failing test in the ordered list

    Returns:
        float in [0, 1]. Higher is better. Returns float("nan") if m == 0.

    Reference:
        Elbaum et al., "Test Case Prioritization: A Family of Empirical Studies",
        IEEE TSE 2002.
    """
    n = len(ordered_test_ids)
    m = len(failing_test_ids)

    if m == 0 or n == 0:
        return float("nan")

    # Find the 1-indexed position of each failing test in the ordered list
    positions = []
    for idx, test_id in enumerate(ordered_test_ids, start=1):
        if test_id in failing_test_ids:
            positions.append(idx)

    if not positions:
        # No failing test was included in the ordered list — worst case
        return 0.0

    tf_sum = sum(positions)
    return 1.0 - tf_sum / (n * m) + 1.0 / (2.0 * n)


def time_to_first_failure(
    ordered_test_ids: Sequence[str],
    failing_test_ids: set[str],
    durations: dict[str, float],
) -> float:
    """
    Cumulative test duration (seconds) until the first failing test is encountered.

    Returns:
        float seconds. Returns float("inf") if no failing test is found.
    """
    cumulative = 0.0
    for test_id in ordered_test_ids:
        cumulative += durations.get(test_id, 0.0)
        if test_id in failing_test_ids:
            return cumulative
    return float("inf")


def pct_tests_to_catch_all(
    ordered_test_ids: Sequence[str],
    failing_test_ids: set[str],
) -> float:
    """
    Fraction of the test suite (0–1) that must be executed before
    all failing tests have been seen.

    Returns:
        float in (0, 1]. Returns float("nan") if m == 0.
    """
    n = len(ordered_test_ids)
    m = len(failing_test_ids)

    if m == 0 or n == 0:
        return float("nan")

    remaining = set(failing_test_ids)
    for idx, test_id in enumerate(ordered_test_ids, start=1):
        remaining.discard(test_id)
        if not remaining:
            return idx / n

    # Some failing tests weren't in the ordered list at all
    return 1.0


def runtime_saved_pct(
    ordered_test_ids: Sequence[str],
    failing_test_ids: set[str],
    durations: dict[str, float],
) -> float:
    """
    Fraction of total suite runtime saved by stopping as soon as all
    failing tests have been detected.

    runtime_saved = (total_runtime - runtime_at_last_failure) / total_runtime

    Returns:
        float in [0, 1]. Returns float("nan") if total runtime is 0 or m == 0.
        Multiply by 100 to get a percentage for display.
    """
    m = len(failing_test_ids)
    if m == 0:
        return float("nan")

    total_runtime = sum(durations.get(t, 0.0) for t in ordered_test_ids)
    if total_runtime <= 0:
        return float("nan")

    remaining = set(failing_test_ids)
    runtime_at_last_failure = 0.0
    cumulative = 0.0

    for test_id in ordered_test_ids:
        cumulative += durations.get(test_id, 0.0)
        remaining.discard(test_id)
        if not remaining:
            runtime_at_last_failure = cumulative
            break
    else:
        runtime_at_last_failure = total_runtime  # no saving possible

    saved = (total_runtime - runtime_at_last_failure) / total_runtime
    return max(0.0, saved)


def aggregate_metrics(
    per_commit_results: list[dict[str, float]],
) -> dict[str, float]:
    """
    Aggregate per-commit metric dicts into mean ± std summary.

    Input:  list of dicts like {"apfd": 0.72, "ttff": 12.3, ...}
    Output: dict with keys "{metric}_mean" and "{metric}_std"
    """
    import statistics

    if not per_commit_results:
        return {}

    keys = [k for k in per_commit_results[0].keys() if not math.isnan(per_commit_results[0][k])]
    result: dict[str, float] = {}

    for key in keys:
        values = [r[key] for r in per_commit_results if not math.isnan(r.get(key, float("nan")))]
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            result[f"{key}_n"] = len(values)

    return result
