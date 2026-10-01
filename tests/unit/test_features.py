"""
Unit tests for src/features/engineer.py

Tests feature engineering logic without any disk I/O.
All test DataFrames are constructed inline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.engineer import FEATURE_COLUMNS, FeatureEngineer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "commit_ts" in df.columns:
        df["commit_ts"] = pd.to_datetime(df["commit_ts"], utc=True)
    return df


def _minimal_row(
    test_id="tests::test_foo",
    outcome=0,
    commit_sha="aaa",
    commit_ts="2023-01-01",
    author="dev_0",
    files_changed=2,
    lines_added=10,
    lines_deleted=5,
    test_files_changed=0,
    src_files_changed=2,
    changed_file_list=None,
    duration_s=1.0,
    prev_1=None,
    prev_2=None,
    **kwargs,
) -> dict:
    row = dict(
        test_id=test_id,
        outcome=outcome,
        commit_sha=commit_sha,
        commit_ts=commit_ts,
        author=author,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        test_files_changed=test_files_changed,
        src_files_changed=src_files_changed,
        changed_file_list=json.dumps(changed_file_list or ["src/client.py"]),
        duration_s=duration_s,
    )
    if prev_1 is not None:
        row["prev_1"] = prev_1
    if prev_2 is not None:
        row["prev_2"] = prev_2
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# FEATURE_COLUMNS contract
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    def test_all_expected_features_present(self):
        expected = {
            "hist_fail_rate",
            "hist_last_failed",
            "hist_weighted_score",
            "hist_consecutive_passes",
            "test_global_fail_rate",
            "test_is_high_risk",
            "files_changed",
            "lines_added",
            "lines_deleted",
            "change_size",
            "test_files_changed",
            "src_files_changed",
            "cochange_score",
            "author_fail_rate",
            "duration_s",
            "duration_log1p",
        }
        assert expected.issubset(set(FEATURE_COLUMNS))

    def test_feature_columns_has_no_duplicates(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# FeatureEngineer — fit / transform
# ---------------------------------------------------------------------------


class TestFeatureEngineerFitTransform:
    def _fitted_fe(self, records=None) -> FeatureEngineer:
        records = records or [
            _minimal_row("t1", outcome=1, prev_1=1.0, prev_2=0.0),
            _minimal_row("t2", outcome=0, prev_1=0.0, prev_2=0.0),
            _minimal_row("t1", outcome=0, prev_1=0.0, commit_sha="bbb", commit_ts="2023-01-02"),
        ]
        df = _make_df(records)
        fe = FeatureEngineer(history_window=5)
        fe.fit(df)
        return fe

    def test_transform_returns_correct_columns(self):
        fe = self._fitted_fe()
        df = _make_df([_minimal_row("t1", prev_1=0.0)])
        result = fe.transform(df)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_transform_returns_float_dtype(self):
        fe = self._fitted_fe()
        df = _make_df([_minimal_row("t1", prev_1=0.0)])
        result = fe.transform(df)
        assert all(pd.api.types.is_float_dtype(dt) for dt in result.dtypes)

    def test_transform_row_count_matches_input(self):
        fe = self._fitted_fe()
        records = [_minimal_row("t1"), _minimal_row("t2"), _minimal_row("t1", commit_sha="bbb")]
        df = _make_df(records)
        result = fe.transform(df)
        assert len(result) == len(records)

    def test_fit_transform_is_equivalent_to_fit_then_transform(self):
        records = [_minimal_row("t1", outcome=1), _minimal_row("t2", outcome=0)]
        df = _make_df(records)
        fe1 = FeatureEngineer()
        ft1 = fe1.fit_transform(df)
        fe2 = FeatureEngineer()
        fe2.fit(df)
        ft2 = fe2.transform(df)
        pd.testing.assert_frame_equal(ft1, ft2)

    def test_transform_without_fit_raises(self):
        fe = FeatureEngineer()
        df = _make_df([_minimal_row("t1")])
        with pytest.raises(RuntimeError, match="fit"):
            fe.transform(df)


# ---------------------------------------------------------------------------
# History features
# ---------------------------------------------------------------------------


class TestHistoryFeatures:
    def _get_features(self, prev_vals: dict, outcome: int = 0) -> pd.Series:
        row = _minimal_row("t1", outcome=outcome, **prev_vals)
        train_df = _make_df([row])
        fe = FeatureEngineer(history_window=5)
        return fe.fit_transform(train_df).iloc[0]

    def test_hist_fail_rate_all_failures(self):
        feats = self._get_features({"prev_1": 1.0, "prev_2": 1.0, "prev_3": 1.0})
        assert feats["hist_fail_rate"] == pytest.approx(1.0)

    def test_hist_fail_rate_no_failures(self):
        feats = self._get_features({"prev_1": 0.0, "prev_2": 0.0})
        assert feats["hist_fail_rate"] == pytest.approx(0.0)

    def test_hist_last_failed_when_prev1_is_one(self):
        feats = self._get_features({"prev_1": 1.0, "prev_2": 0.0})
        assert feats["hist_last_failed"] == pytest.approx(1.0)

    def test_hist_last_failed_when_prev1_is_zero(self):
        feats = self._get_features({"prev_1": 0.0})
        assert feats["hist_last_failed"] == pytest.approx(0.0)

    def test_hist_consecutive_passes_after_long_pass_streak(self):
        feats = self._get_features({"prev_1": 0.0, "prev_2": 0.0, "prev_3": 0.0})
        assert feats["hist_consecutive_passes"] == pytest.approx(3.0)

    def test_hist_consecutive_passes_resets_on_failure(self):
        feats = self._get_features({"prev_1": 0.0, "prev_2": 1.0, "prev_3": 0.0})
        assert feats["hist_consecutive_passes"] == pytest.approx(1.0)

    def test_no_history_uses_global_rate_as_fallback(self):
        """Test with no prev columns at all — should use global mean."""
        row = _minimal_row("t1", outcome=1)  # no prev_* keys
        train_df = _make_df([row])
        fe = FeatureEngineer()
        result = fe.fit_transform(train_df)
        # Global rate is 1.0 (only 1 failure in training)
        # hist_fail_rate should be filled with that global rate
        assert not math.isnan(result.iloc[0]["hist_fail_rate"])

    def test_weighted_score_weights_recent_more(self):
        """
        A test whose most recent run failed (prev_1=1) should have a higher
        hist_weighted_score than one where only an older run failed (prev_2=1),
        because the weighting is 1/i.
        Both rows are in the same DataFrame so the same global rate applies.
        """
        records = [
            # row_a: prev_1=1, prev_2=0 → score = 1/1 + 0/2 = 1.0
            _minimal_row("t_a", outcome=0, prev_1=1.0, prev_2=0.0, commit_sha="c1"),
            # row_b: prev_1=0, prev_2=1 → score = 0/1 + 1/2 = 0.5
            _minimal_row("t_b", outcome=0, prev_1=0.0, prev_2=1.0, commit_sha="c2"),
        ]
        df = _make_df(records)
        fe = FeatureEngineer(history_window=5)
        result = fe.fit_transform(df)
        score_a = result[df["test_id"] == "t_a"]["hist_weighted_score"].iloc[0]
        score_b = result[df["test_id"] == "t_b"]["hist_weighted_score"].iloc[0]
        assert score_a > score_b


# ---------------------------------------------------------------------------
# Change features
# ---------------------------------------------------------------------------


class TestChangeFeatures:
    def _get(self, **kw) -> pd.Series:
        row = _minimal_row("t1", **kw)
        fe = FeatureEngineer()
        return fe.fit_transform(_make_df([row])).iloc[0]

    def test_change_size_is_sum_of_adds_and_deletes(self):
        feats = self._get(lines_added=30, lines_deleted=10)
        assert feats["change_size"] == pytest.approx(40.0)

    def test_duration_log1p_of_zero_is_zero(self):
        feats = self._get(duration_s=0.0)
        assert feats["duration_log1p"] == pytest.approx(0.0)

    def test_duration_log1p_greater_than_raw_for_large_values(self):
        feats = self._get(duration_s=100.0)
        assert feats["duration_log1p"] < feats["duration_s"]

    def test_files_changed_is_preserved(self):
        feats = self._get(files_changed=7)
        assert feats["files_changed"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Leakage protection — test_global_fail_rate
# ---------------------------------------------------------------------------


class TestLeakageProtection:
    def test_test_global_fail_rate_from_train_only(self):
        """
        test_global_fail_rate should reflect training data only.
        If a test never appeared in training, it should get the global mean.
        """
        train_records = [
            _minimal_row("known_test", outcome=1, commit_sha="c1", commit_ts="2023-01-01"),
            _minimal_row("known_test", outcome=1, commit_sha="c2", commit_ts="2023-01-02"),
        ]
        test_records = [
            _minimal_row("new_test", outcome=0, commit_sha="c3", commit_ts="2023-06-01"),
        ]
        train_df = _make_df(train_records)
        test_df = _make_df(test_records)

        fe = FeatureEngineer()
        fe.fit(train_df)
        result = fe.transform(test_df)

        # "new_test" was never in training → gets global mean (1.0 from train)
        global_rate = fe._global_fail_rate
        assert result.iloc[0]["test_global_fail_rate"] == pytest.approx(global_rate)

    def test_author_fail_rate_from_train_only(self):
        train_records = [
            _minimal_row("t1", outcome=1, author="dev_a", commit_sha="c1"),
            _minimal_row("t2", outcome=0, author="dev_b", commit_sha="c2"),
        ]
        test_records = [
            _minimal_row("t3", outcome=0, author="dev_a", commit_sha="c3"),
            _minimal_row("t4", outcome=0, author="unknown_dev", commit_sha="c4"),
        ]
        fe = FeatureEngineer()
        fe.fit(_make_df(train_records))
        result = fe.transform(_make_df(test_records))

        # dev_a had 100% failure rate in training
        assert result.iloc[0]["author_fail_rate"] == pytest.approx(1.0)
        # unknown_dev → global rate
        assert result.iloc[1]["author_fail_rate"] == pytest.approx(fe._global_fail_rate)


# ---------------------------------------------------------------------------
# Co-failure (cochange_score) feature
# ---------------------------------------------------------------------------


class TestCochangeScore:
    def test_high_cochange_score_for_test_that_always_fails_with_file(self):
        """
        If test_A always fails when file 'src/client.py' changes,
        its cochange_score should be 1.0 when src/client.py is in changed_file_list.
        """
        train_records = [
            _minimal_row("test_A", outcome=1, changed_file_list=["src/client.py"], commit_sha="c1"),
            _minimal_row("test_A", outcome=1, changed_file_list=["src/client.py"], commit_sha="c2"),
            _minimal_row("test_A", outcome=1, changed_file_list=["src/client.py"], commit_sha="c3"),
        ]
        test_records = [
            _minimal_row("test_A", outcome=0, changed_file_list=["src/client.py"], commit_sha="c4"),
        ]
        fe = FeatureEngineer()
        fe.fit(_make_df(train_records))
        result = fe.transform(_make_df(test_records))
        assert result.iloc[0]["cochange_score"] == pytest.approx(1.0)

    def test_zero_cochange_score_for_unrelated_file(self):
        train_records = [
            _minimal_row("test_A", outcome=0, changed_file_list=["src/auth.py"], commit_sha="c1"),
        ]
        test_records = [
            _minimal_row(
                "test_A", outcome=0, changed_file_list=["src/totally_new.py"], commit_sha="c2"
            ),
        ]
        fe = FeatureEngineer()
        fe.fit(_make_df(train_records))
        result = fe.transform(_make_df(test_records))
        assert result.iloc[0]["cochange_score"] == pytest.approx(0.0)
