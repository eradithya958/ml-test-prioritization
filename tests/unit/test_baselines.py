"""
Unit tests for src/models/baselines.py

Every test is self-contained — no disk I/O, no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.base import OrderContext
from src.models.baselines import (
    CoverageBasedOrderer,
    RandomOrderer,
    RecentlyFailedFirstOrderer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(
    changed_files: list[str] | None = None,
    history_records: list[dict] | None = None,
) -> OrderContext:
    """Build a minimal OrderContext for testing."""
    history_records = history_records or []
    history = pd.DataFrame(history_records)
    return OrderContext(
        commit_sha="abc123",
        changed_files=changed_files or [],
        test_history=history,
    )


TEST_IDS = ["test_a", "test_b", "test_c", "test_d", "test_e"]


# ---------------------------------------------------------------------------
# Contract tests (all orderers must satisfy these)
# ---------------------------------------------------------------------------


class TestOrdererContract:
    """
    All orderers must return the same set of test IDs as the input.
    Order may differ; set must not.
    """

    @pytest.mark.parametrize(
        "orderer",
        [
            RandomOrderer(),
            RecentlyFailedFirstOrderer(),
            CoverageBasedOrderer(),
        ],
    )
    def test_returns_same_set_of_tests(self, orderer):
        ctx = make_context()
        result = orderer.order(TEST_IDS, ctx)
        assert set(result) == set(TEST_IDS)

    @pytest.mark.parametrize(
        "orderer",
        [
            RandomOrderer(),
            RecentlyFailedFirstOrderer(),
            CoverageBasedOrderer(),
        ],
    )
    def test_returns_correct_count(self, orderer):
        ctx = make_context()
        result = orderer.order(TEST_IDS, ctx)
        assert len(result) == len(TEST_IDS)

    @pytest.mark.parametrize(
        "orderer",
        [
            RandomOrderer(),
            RecentlyFailedFirstOrderer(),
            CoverageBasedOrderer(),
        ],
    )
    def test_handles_empty_test_list(self, orderer):
        ctx = make_context()
        result = orderer.order([], ctx)
        assert result == []

    @pytest.mark.parametrize(
        "orderer",
        [
            RandomOrderer(),
            RecentlyFailedFirstOrderer(),
            CoverageBasedOrderer(),
        ],
    )
    def test_handles_single_test(self, orderer):
        ctx = make_context()
        result = orderer.order(["only_test"], ctx)
        assert result == ["only_test"]


# ---------------------------------------------------------------------------
# RandomOrderer
# ---------------------------------------------------------------------------


class TestRandomOrderer:
    def test_different_seeds_give_different_orders(self):
        ctx = make_context()
        r1 = RandomOrderer(seed=1).order(TEST_IDS, ctx)
        r2 = RandomOrderer(seed=2).order(TEST_IDS, ctx)
        assert r1 != r2

    def test_same_seed_gives_same_order(self):
        ctx = make_context()
        r1 = RandomOrderer(seed=99).order(TEST_IDS, ctx)
        r2 = RandomOrderer(seed=99).order(TEST_IDS, ctx)
        assert r1 == r2

    def test_name_is_random(self):
        assert RandomOrderer().name == "random"


# ---------------------------------------------------------------------------
# RecentlyFailedFirstOrderer
# ---------------------------------------------------------------------------


class TestRecentlyFailedFirstOrderer:
    def _history(self, records: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(records)

    def test_recently_failed_test_comes_first(self):
        history = self._history(
            [
                {"test_id": "test_a", "prev_1": 1.0, "prev_2": 0.0, "duration_s": 1.0},
                {"test_id": "test_b", "prev_1": 0.0, "prev_2": 0.0, "duration_s": 1.0},
                {"test_id": "test_c", "prev_1": 0.0, "prev_2": 0.0, "duration_s": 1.0},
            ]
        )
        ctx = make_context(history_records=history.to_dict("records"))
        orderer = RecentlyFailedFirstOrderer()
        result = orderer.order(["test_a", "test_b", "test_c"], ctx)
        assert result[0] == "test_a"

    def test_test_with_no_history_goes_last(self):
        history = self._history(
            [
                {"test_id": "test_a", "prev_1": 1.0, "duration_s": 1.0},
            ]
        )
        ctx = make_context(history_records=history.to_dict("records"))
        orderer = RecentlyFailedFirstOrderer()
        # test_b has no history → should rank below test_a
        result = orderer.order(["test_a", "test_b"], ctx)
        assert result[0] == "test_a"
        assert result[1] == "test_b"

    def test_more_recent_failure_beats_older_failure(self):
        """prev_1 (weight 1.0) vs prev_2 (weight 0.5) — most recent counts more."""
        history = self._history(
            [
                # test_a failed 2 runs ago (prev_2=1)
                {"test_id": "test_a", "prev_1": 0.0, "prev_2": 1.0, "duration_s": 1.0},
                # test_b failed last run (prev_1=1)
                {"test_id": "test_b", "prev_1": 1.0, "prev_2": 0.0, "duration_s": 1.0},
            ]
        )
        ctx = make_context(history_records=history.to_dict("records"))
        orderer = RecentlyFailedFirstOrderer()
        result = orderer.order(["test_a", "test_b"], ctx)
        assert result[0] == "test_b"

    def test_empty_history_does_not_crash(self):
        ctx = make_context(history_records=[])
        orderer = RecentlyFailedFirstOrderer()
        result = orderer.order(["t1", "t2", "t3"], ctx)
        assert set(result) == {"t1", "t2", "t3"}

    def test_name_is_recently_failed_first(self):
        assert RecentlyFailedFirstOrderer().name == "recently_failed_first"


# ---------------------------------------------------------------------------
# CoverageBasedOrderer
# ---------------------------------------------------------------------------


class TestCoverageBasedOrderer:
    def test_test_matching_changed_file_comes_first(self):
        # Changed file: httpx/_client.py → token "client"
        # test_client should rank above test_auth
        ctx = make_context(changed_files=["httpx/_client.py"])
        orderer = CoverageBasedOrderer()
        result = orderer.order(["test_auth", "test_client"], ctx)
        assert result[0] == "test_client"

    def test_no_changed_files_falls_back_to_recency(self):
        history = [
            {"test_id": "test_a", "prev_1": 1.0, "duration_s": 1.0},
            {"test_id": "test_b", "prev_1": 0.0, "duration_s": 1.0},
        ]
        # No changed files → coverage score = 0 for all → recency tiebreak
        ctx = make_context(changed_files=[], history_records=history)
        orderer = CoverageBasedOrderer()
        result = orderer.order(["test_a", "test_b"], ctx)
        assert result[0] == "test_a"

    def test_multiple_matching_files_score_higher(self):
        # test_auth matches "auth" from two changed files
        ctx = make_context(changed_files=["httpx/_auth.py", "httpx/auth.py"])
        orderer = CoverageBasedOrderer()
        result = orderer.order(["test_client", "test_auth"], ctx)
        assert result[0] == "test_auth"

    def test_token_extraction_handles_path_components(self):
        tokens = CoverageBasedOrderer._extract_tokens(["src/httpx/_client.py"])
        assert "httpx" in tokens
        assert "client" in tokens

    def test_token_extraction_strips_leading_underscore(self):
        tokens = CoverageBasedOrderer._extract_tokens(["src/_internal.py"])
        assert "internal" in tokens

    def test_token_extraction_empty_files(self):
        tokens = CoverageBasedOrderer._extract_tokens([])
        assert tokens == set()

    def test_name_is_coverage_based(self):
        assert CoverageBasedOrderer().name == "coverage_based"

    def test_unchanged_files_all_get_same_score(self):
        """With no file overlap, order should not raise and must return same set."""
        ctx = make_context(changed_files=["totally_unrelated/xyz.py"])
        orderer = CoverageBasedOrderer()
        result = orderer.order(["test_a", "test_b", "test_c"], ctx)
        assert set(result) == {"test_a", "test_b", "test_c"}
