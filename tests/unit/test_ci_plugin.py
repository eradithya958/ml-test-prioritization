"""
tests/unit/test_ci_plugin.py
============================
Unit tests for pytest CI plugin (prioritization and flaky quarantine).
"""

from pathlib import Path
from unittest.mock import MagicMock

from src.ci_plugin.plugin import (
    pytest_collection_modifyitems,
    pytest_configure,
)
from src.flaky.registry import FlakyRegistry


class DummyItem:
    def __init__(self, nodeid: str):
        self.nodeid = nodeid
        self.markers = []

    def add_marker(self, marker, append=False):
        if append:
            self.markers.append(marker)
        else:
            self.markers.insert(0, marker)


class DummyConfig:
    def __init__(self, options: dict):
        self._opts = options

    def getoption(self, name: str, default=None):
        return self._opts.get(name, default)

    def addinivalue_line(self, name: str, line: str):
        pass


class TestCIPlugin:
    def test_pytest_configure_registers_marker(self):
        config = MagicMock()
        pytest_configure(config)
        config.addinivalue_line.assert_called_with(
            "markers", "flaky: marks test as quarantined flaky test"
        )

    def test_quarantine_marking(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)
        reg.add("test_a", reason="flip")

        items = [DummyItem("test_a"), DummyItem("test_b")]
        config = DummyConfig(
            {
                "--quarantine": "mark",
                "--flaky-registry": str(reg_file),
                "--prioritize": "none",
            }
        )

        pytest_collection_modifyitems(None, config, items)

        assert len(items[0].markers) == 1
        assert items[0].markers[0].name == "flaky"
        assert len(items[1].markers) == 0

    def test_quarantine_skipping(self, tmp_path: Path):
        reg_file = tmp_path / "flaky_reg.json"
        reg = FlakyRegistry(reg_file)
        reg.add("test_a", reason="flip")

        items = [DummyItem("test_a"), DummyItem("test_b")]
        config = DummyConfig(
            {
                "--quarantine": "skip",
                "--flaky-registry": str(reg_file),
                "--prioritize": "none",
            }
        )

        pytest_collection_modifyitems(None, config, items)

        assert len(items[0].markers) == 1
        assert items[0].markers[0].name == "skip"
        assert len(items[1].markers) == 0

    def test_prioritization_random(self):
        items = [DummyItem(f"test_{i}") for i in range(20)]
        original_order = [item.nodeid for item in items]
        config = DummyConfig(
            {
                "--quarantine": "none",
                "--prioritize": "random",
                "--changed-files": "",
            }
        )

        pytest_collection_modifyitems(None, config, items)
        new_order = [item.nodeid for item in items]

        assert set(new_order) == set(original_order)
        assert len(new_order) == len(original_order)

    def test_prioritization_coverage(self):
        items = [
            DummyItem("tests/test_auth.py::test_login"),
            DummyItem("tests/test_client.py::test_get"),
        ]
        config = DummyConfig(
            {
                "--quarantine": "none",
                "--prioritize": "coverage",
                "--changed-files": "src/auth/handler.py",
            }
        )

        pytest_collection_modifyitems(None, config, items)
        new_order = [item.nodeid for item in items]

        # test_auth should be ranked first because of name/token matching
        assert new_order[0] == "tests/test_auth.py::test_login"
