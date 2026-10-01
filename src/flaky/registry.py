"""
src/flaky/registry.py
======================
Persistent quarantine registry for known-flaky tests.

Stores a JSON file at data/flaky_registry.json with one entry per
quarantined test. Provides:

  - FlakyRegistry.add()    — quarantine a test
  - FlakyRegistry.remove() — unquarantine a test (after fix)
  - FlakyRegistry.is_flaky() — fast O(1) lookup
  - FlakyRegistry.generate_conftest_snippet() — copy into conftest.py

Registry file format:
{
  "tests/test_client_0001.py::test_case": {
    "detected_at": "2024-01-15T10:30:00+00:00",
    "reason": "flip_without_change",
    "flaky_score": 0.42,
    "flip_rate": 0.38,
    "fail_rate_no_change": 0.18,
    "quarantine_version": 1
  }
}
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/flaky_registry.json")


class FlakyRegistry:
    """Thread-safe (single-process) read/write registry for flaky tests."""

    def __init__(self, path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        test_id: str,
        reason: str,
        metrics: dict | None = None,
    ) -> None:
        """Quarantine a test. Idempotent — re-adding updates the record."""
        self._data[test_id] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            **(metrics or {}),
        }
        self._save()
        log.info("Quarantined: %s  (reason=%s)", test_id, reason)

    def remove(self, test_id: str) -> bool:
        """
        Remove a test from quarantine (e.g., after the root cause is fixed).

        Returns True if the test was in the registry, False otherwise.
        """
        if test_id in self._data:
            del self._data[test_id]
            self._save()
            log.info("Unquarantined: %s", test_id)
            return True
        return False

    def is_flaky(self, test_id: str) -> bool:
        """Return True if the test_id is in the quarantine registry."""
        # Support both exact match and pytest nodeid format
        return test_id in self._data or any(
            test_id.endswith(k) or k.endswith(test_id) for k in self._data
        )

    def all_flaky(self) -> list[str]:
        """Return sorted list of all quarantined test IDs."""
        return sorted(self._data.keys())

    def get_record(self, test_id: str) -> dict | None:
        """Return the full quarantine record for a test, or None."""
        return self._data.get(test_id)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, test_id: str) -> bool:
        return self.is_flaky(test_id)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def add_from_report(self, report, threshold: float | None = None) -> int:
        """
        Add all flaky tests from a DetectionReport to the registry.

        Args:
            report: DetectionReport from FlakyDetector.analyze()
            threshold: override threshold (default: use report.threshold)

        Returns:
            Number of tests newly quarantined.
        """
        thresh = threshold if threshold is not None else report.threshold
        added = 0
        for m in report.per_test:
            if m.flaky_score >= thresh:
                self.add(
                    test_id=m.test_id,
                    reason="historical_flip_analysis",
                    metrics={
                        "flaky_score": m.flaky_score,
                        "flip_rate": m.flip_rate,
                        "fail_rate_no_change": m.fail_rate_no_change,
                        "n_runs": m.n_runs,
                        "n_flips": m.n_flips,
                    },
                )
                added += 1
        return added

    # Alias for update_from_report
    update_from_report = add_from_report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"FlakyRegistry — {len(self._data)} quarantined tests ({self.path})"]
        for tid, record in sorted(self._data.items()):
            score = record.get("flaky_score", "?")
            reason = record.get("reason", "?")
            lines.append(f"  [{score:.3f}] {tid}  ({reason})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # conftest.py integration snippet
    # ------------------------------------------------------------------

    def generate_conftest_snippet(self) -> str:
        """
        Return a conftest.py snippet that automatically tags all quarantined
        tests with @pytest.mark.flaky based on this registry.

        Usage: append this to your project's conftest.py.
        """
        registry_path_str = str(self.path)
        return f'''
# ── Flaky test quarantine (auto-generated by FlakyRegistry) ──────────
import pytest
from pathlib import Path
from src.flaky.registry import FlakyRegistry

_FLAKY_REGISTRY = FlakyRegistry(Path("{registry_path_str}"))

def pytest_collection_modifyitems(items, config):
    """Tag known-flaky tests so they can be excluded from blocking CI."""
    flaky_marker = pytest.mark.flaky
    for item in items:
        if _FLAKY_REGISTRY.is_flaky(item.nodeid):
            item.add_marker(flaky_marker, append=False)

# Run without flaky tests:  pytest -m "not flaky"
# Run only flaky tests:     pytest -m "flaky"
# ─────────────────────────────────────────────────────────────────────
'''

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
                log.debug("Loaded %d quarantined tests from %s", len(self._data), self.path)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not load registry %s: %s — starting empty", self.path, exc)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        tmp.replace(self.path)  # atomic rename
