"""
tests/unit/test_flaky.py
========================
Unit tests for flaky test detection, registry management, and rerun verification.
"""

from pathlib import Path

import pandas as pd

from src.flaky.detector import DetectionReport, FlakyDetector, FlakyTestMetrics
from src.flaky.registry import FlakyRegistry
from src.flaky.rerunner import RerunVerifier


class TestFlakyRegistry:
    def test_add_and_is_flaky(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)

        assert len(reg) == 0
        assert not reg.is_flaky("tests/test_foo.py::test_one")

        reg.add(
            "tests/test_foo.py::test_one",
            reason="flip_without_change",
            metrics={"flaky_score": 0.5},
        )
        assert len(reg) == 1
        assert reg.is_flaky("tests/test_foo.py::test_one")
        assert "tests/test_foo.py::test_one" in reg

        rec = reg.get_record("tests/test_foo.py::test_one")
        assert rec is not None
        assert rec["reason"] == "flip_without_change"
        assert rec["flaky_score"] == 0.5

    def test_remove(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)
        reg.add("test_a", reason="manual")
        assert reg.is_flaky("test_a")

        removed = reg.remove("test_a")
        assert removed is True
        assert not reg.is_flaky("test_a")
        assert len(reg) == 0

        # Removing again returns False
        assert reg.remove("test_a") is False

    def test_persistence(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg1 = FlakyRegistry(reg_file)
        reg1.add("test_a", reason="test_reason")

        # Create new instance pointing to same file
        reg2 = FlakyRegistry(reg_file)
        assert reg2.is_flaky("test_a")
        assert len(reg2) == 1

    def test_update_from_report(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)

        metrics = [
            FlakyTestMetrics(
                test_id="test_1",
                n_runs=20,
                n_failures=5,
                failure_rate=0.25,
                flip_rate=0.4,
                fail_rate_no_change=0.2,
                n_runs_no_change=10,
                n_flips=2,
                flaky_score=0.3,
                is_flaky=True,
            ),
            FlakyTestMetrics(
                test_id="test_2",
                n_runs=20,
                n_failures=1,
                failure_rate=0.05,
                flip_rate=0.0,
                fail_rate_no_change=0.0,
                n_runs_no_change=10,
                n_flips=0,
                flaky_score=0.0,
                is_flaky=False,
            ),
        ]
        report = DetectionReport(
            total_tests_analyzed=2,
            n_flaky_detected=1,
            n_not_flaky=1,
            threshold=0.10,
            per_test=metrics,
        )

        added = reg.update_from_report(report)
        assert added == 1
        assert reg.is_flaky("test_1")
        assert not reg.is_flaky("test_2")

    def test_generate_conftest_snippet(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)
        snippet = reg.generate_conftest_snippet()
        assert "pytest_collection_modifyitems" in snippet
        assert "FlakyRegistry" in snippet


class TestFlakyDetector:
    def test_analyze_identifies_flaky_pattern(self):
        # Construct DataFrame with clear flip without change
        records = [
            # Commit 1: src changed, test passes
            {
                "commit_sha": "c1",
                "test_id": "test_flaky",
                "outcome": 0,
                "src_files_changed": 1,
                "is_flaky_ground_truth": True,
            },
            # Commit 2: 0 src changed, test fails
            {
                "commit_sha": "c2",
                "test_id": "test_flaky",
                "outcome": 1,
                "src_files_changed": 0,
                "is_flaky_ground_truth": True,
            },
            # Commit 3: 0 src changed, test passes (flip!)
            {
                "commit_sha": "c3",
                "test_id": "test_flaky",
                "outcome": 0,
                "src_files_changed": 0,
                "is_flaky_ground_truth": True,
            },
            # Commit 4: 0 src changed, test fails
            {
                "commit_sha": "c4",
                "test_id": "test_flaky",
                "outcome": 1,
                "src_files_changed": 0,
                "is_flaky_ground_truth": True,
            },
            # Commit 5: 0 src changed, test passes (flip!)
            {
                "commit_sha": "c5",
                "test_id": "test_flaky",
                "outcome": 0,
                "src_files_changed": 0,
                "is_flaky_ground_truth": True,
            },
        ]
        df = pd.DataFrame(records)
        detector = FlakyDetector(threshold=0.10, min_runs=3)
        report = detector.analyze(df)

        assert report.total_tests_analyzed == 1
        assert report.n_flaky_detected == 1
        m = report.per_test[0]
        assert m.test_id == "test_flaky"
        assert m.is_flaky is True
        assert m.flaky_score > 0.10

    def test_stable_test_not_flagged(self):
        records = [
            {"commit_sha": f"c{i}", "test_id": "test_stable", "outcome": 0, "src_files_changed": 2}
            for i in range(10)
        ]
        df = pd.DataFrame(records)
        detector = FlakyDetector(threshold=0.10, min_runs=5)
        report = detector.analyze(df)

        assert report.n_flaky_detected == 0
        assert report.per_test[0].flaky_score == 0.0


class TestRerunVerifier:
    def test_flaky_test_passes_on_second_run(self):
        verifier = RerunVerifier(n_reruns=3, stop_on_pass=True)

        call_count = 0

        def flaky_run():
            nonlocal call_count
            call_count += 1
            # First rerun fails, second passes
            return call_count == 2

        result = verifier.verify("test_flaky", run_fn=flaky_run, original_outcome=1)

        assert result.is_flaky is True
        assert result.passed_on_rerun is True
        assert result.n_reruns_executed == 2
        assert result.outcomes == [1, 0]

    def test_genuine_failure_fails_all_reruns(self):
        verifier = RerunVerifier(n_reruns=3, stop_on_pass=True)

        def always_fails():
            return False

        result = verifier.verify("test_broken", run_fn=always_fails, original_outcome=1)

        assert result.is_flaky is False
        assert result.passed_on_rerun is False
        assert result.n_reruns_executed == 3
        assert result.outcomes == [1, 1, 1]

    def test_original_pass_not_rerun(self):
        verifier = RerunVerifier(n_reruns=3)
        result = verifier.verify("test_passed", run_fn=lambda: True, original_outcome=0)
        assert result.n_reruns_executed == 0
        assert result.is_flaky is False
