"""
Unit tests for scripts/collect_data.py

Tests the pure-Python functions that don't require network access.
Run with: pytest tests/unit/test_data_collection.py -v
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.collect_data import (
    build_history_window,
    extract_commit_features,
    extract_junit_from_zip,
    parse_junit_xml,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

PASSING_JUNIT = b"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="test_suite" tests="3" failures="0" errors="0" time="1.234">
  <testcase classname="tests.test_client" name="test_get" time="0.500"/>
  <testcase classname="tests.test_client" name="test_post" time="0.300"/>
  <testcase classname="tests.test_auth"   name="test_bearer" time="0.434"/>
</testsuite>
"""

MIXED_JUNIT = b"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="test_suite" tests="4" failures="1" errors="1" time="2.0">
  <testcase classname="tests.test_client" name="test_get"    time="0.5"/>
  <testcase classname="tests.test_client" name="test_post"   time="0.3">
    <failure message="AssertionError: expected 200 got 500">long traceback</failure>
  </testcase>
  <testcase classname="tests.test_stream" name="test_stream" time="0.8">
    <error message="ConnectionError">connection reset</error>
  </testcase>
  <testcase classname="tests.test_async"  name="test_await"  time="0.0">
    <skipped/>
  </testcase>
</testsuite>
"""

MALFORMED_XML = b"this is not xml <broken"


# -------------------------------------------------------------------------
# parse_junit_xml tests
# -------------------------------------------------------------------------


class TestParseJunitXml:
    def test_all_passing_tests_outcome_zero(self):
        results = parse_junit_xml(PASSING_JUNIT)
        assert len(results) == 3
        assert all(r["outcome"] == 0 for r in results)

    def test_test_ids_are_formatted_correctly(self):
        results = parse_junit_xml(PASSING_JUNIT)
        ids = {r["test_id"] for r in results}
        assert "tests.test_client::test_get" in ids
        assert "tests.test_client::test_post" in ids
        assert "tests.test_auth::test_bearer" in ids

    def test_durations_are_parsed(self):
        results = parse_junit_xml(PASSING_JUNIT)
        durations = {r["test_id"]: r["duration_s"] for r in results}
        assert abs(durations["tests.test_client::test_get"] - 0.500) < 1e-6

    def test_failure_outcome_is_one(self):
        results = parse_junit_xml(MIXED_JUNIT)
        by_id = {r["test_id"]: r for r in results}
        assert by_id["tests.test_client::test_post"]["outcome"] == 1

    def test_error_outcome_is_one(self):
        results = parse_junit_xml(MIXED_JUNIT)
        by_id = {r["test_id"]: r for r in results}
        assert by_id["tests.test_stream::test_stream"]["outcome"] == 1

    def test_skipped_outcome_is_two(self):
        results = parse_junit_xml(MIXED_JUNIT)
        by_id = {r["test_id"]: r for r in results}
        assert by_id["tests.test_async::test_await"]["outcome"] == 2

    def test_failure_message_is_captured(self):
        results = parse_junit_xml(MIXED_JUNIT)
        by_id = {r["test_id"]: r for r in results}
        assert "AssertionError" in by_id["tests.test_client::test_post"]["error_msg"]

    def test_passing_test_has_no_error_msg(self):
        results = parse_junit_xml(PASSING_JUNIT)
        assert all(r["error_msg"] is None for r in results)

    def test_malformed_xml_returns_empty_list(self):
        results = parse_junit_xml(MALFORMED_XML)
        assert results == []

    def test_empty_bytes_returns_empty_list(self):
        results = parse_junit_xml(b"")
        assert results == []

    def test_testsuites_root_element(self):
        xml = b"""<?xml version="1.0"?>
        <testsuites>
          <testsuite name="s1">
            <testcase classname="A" name="b" time="0.1"/>
          </testsuite>
        </testsuites>"""
        results = parse_junit_xml(xml)
        assert len(results) == 1
        assert results[0]["outcome"] == 0


# -------------------------------------------------------------------------
# extract_junit_from_zip tests
# -------------------------------------------------------------------------


class TestExtractJunitFromZip:
    def _make_zip(self, files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_extracts_xml_from_zip(self):
        zip_bytes = self._make_zip({"results/junit.xml": PASSING_JUNIT})
        results = extract_junit_from_zip(zip_bytes)
        assert len(results) == 3

    def test_ignores_non_xml_files(self):
        zip_bytes = self._make_zip(
            {
                "output.log": b"some log output",
                "junit.xml": PASSING_JUNIT,
            }
        )
        results = extract_junit_from_zip(zip_bytes)
        assert len(results) == 3

    def test_multiple_xml_files_merged(self):
        zip_bytes = self._make_zip(
            {
                "unit.xml": PASSING_JUNIT,
                "integration.xml": MIXED_JUNIT,
            }
        )
        results = extract_junit_from_zip(zip_bytes)
        # 3 from PASSING + 4 from MIXED
        assert len(results) == 7

    def test_bad_zip_returns_empty_list(self):
        results = extract_junit_from_zip(b"not a zip file")
        assert results == []

    def test_empty_zip_returns_empty_list(self):
        zip_bytes = self._make_zip({})
        results = extract_junit_from_zip(zip_bytes)
        assert results == []


# -------------------------------------------------------------------------
# extract_commit_features tests
# -------------------------------------------------------------------------


class TestExtractCommitFeatures:
    def _commit_detail(self, files: list[dict], additions: int = 10, deletions: int = 5):
        return {
            "files": files,
            "stats": {"additions": additions, "deletions": deletions},
        }

    def test_basic_counts(self):
        detail = self._commit_detail(
            [{"filename": "src/client.py"}, {"filename": "tests/test_client.py"}]
        )
        feats = extract_commit_features(detail)
        assert feats["files_changed"] == 2
        assert feats["lines_added"] == 10
        assert feats["lines_deleted"] == 5
        assert feats["test_files_changed"] == 1
        assert feats["src_files_changed"] == 1

    def test_only_test_files(self):
        detail = self._commit_detail(
            [
                {"filename": "tests/test_a.py"},
                {"filename": "tests/test_b.py"},
            ]
        )
        feats = extract_commit_features(detail)
        assert feats["test_files_changed"] == 2
        assert feats["src_files_changed"] == 0

    def test_no_files_returns_zeros(self):
        detail = {"files": [], "stats": {"additions": 0, "deletions": 0}}
        feats = extract_commit_features(detail)
        assert feats["files_changed"] == 0

    def test_changed_file_list_is_json_serialisable(self):
        detail = self._commit_detail([{"filename": "src/a.py"}])
        feats = extract_commit_features(detail)
        parsed = json.loads(feats["changed_file_list"])
        assert parsed == ["src/a.py"]

    def test_missing_stats_defaults_to_zero(self):
        detail = {"files": [{"filename": "a.py"}]}
        feats = extract_commit_features(detail)
        assert feats["lines_added"] == 0
        assert feats["lines_deleted"] == 0


# -------------------------------------------------------------------------
# build_history_window tests
# -------------------------------------------------------------------------


class TestBuildHistoryWindow:
    def _make_df(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df["commit_ts"] = pd.to_datetime(df["commit_ts"])
        return df

    def test_prev_1_is_previous_outcome(self):
        df = self._make_df(
            [
                {"test_id": "A", "commit_ts": "2023-01-01", "outcome": 0},
                {"test_id": "A", "commit_ts": "2023-01-02", "outcome": 1},
                {"test_id": "A", "commit_ts": "2023-01-03", "outcome": 0},
            ]
        )
        result = build_history_window(df, history_window=1)
        result = result.sort_values("commit_ts").reset_index(drop=True)
        assert pd.isna(result.loc[0, "prev_1"])  # no prior history
        assert result.loc[1, "prev_1"] == 0.0
        assert result.loc[2, "prev_1"] == 1.0

    def test_multiple_tests_do_not_bleed_into_each_other(self):
        df = self._make_df(
            [
                {"test_id": "A", "commit_ts": "2023-01-01", "outcome": 1},
                {"test_id": "B", "commit_ts": "2023-01-02", "outcome": 0},
                {"test_id": "A", "commit_ts": "2023-01-03", "outcome": 0},
            ]
        )
        result = build_history_window(df, history_window=1)
        # Test B's prev_1 should be NaN (it has no prior run of test B)
        b_row = result[result["test_id"] == "B"].iloc[0]
        assert pd.isna(b_row["prev_1"])

    def test_history_window_column_count(self):
        df = self._make_df(
            [
                {"test_id": "A", "commit_ts": "2023-01-01", "outcome": 0},
            ]
        )
        result = build_history_window(df, history_window=5)
        for i in range(1, 6):
            assert f"prev_{i}" in result.columns

    def test_long_history_first_rows_are_nan(self):
        records = [
            {"test_id": "A", "commit_ts": f"2023-01-{i:02d}", "outcome": i % 2} for i in range(1, 8)
        ]
        df = self._make_df(records)
        result = build_history_window(df, history_window=3)
        result = result.sort_values("commit_ts").reset_index(drop=True)
        # First row: all prev_* should be NaN
        assert pd.isna(result.loc[0, "prev_1"])
        assert pd.isna(result.loc[0, "prev_2"])
        assert pd.isna(result.loc[0, "prev_3"])
        # 4th row: all three prev columns should be filled
        assert not pd.isna(result.loc[3, "prev_1"])
        assert not pd.isna(result.loc[3, "prev_2"])
        assert not pd.isna(result.loc[3, "prev_3"])
