"""
src/ci_plugin/plugin.py
========================
pytest plugin for ML-based test case prioritization and flaky test quarantine.

Hooks:
  - pytest_addoption: register CLI flags (--prioritize, --model-path, --flaky-registry, etc.)
  - pytest_configure: register custom markers (e.g. flaky)
  - pytest_collection_modifyitems:
      1. Quarantine known-flaky tests (mark or skip)
      2. Reorder test items by failure probability (ML model or baseline)

Usage:
  # Prioritize using trained Random Forest model:
  pytest --prioritize=rf

  # Prioritize using XGBoost with explicit model path:
  pytest --prioritize=xgb --model-path=experiments/models/xgboost.pkl

  # Run only non-flaky tests (standard pytest marker):
  pytest -m "not flaky"

  # Automatically skip quarantined flaky tests in CI:
  pytest --quarantine=skip
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.flaky.registry import FlakyRegistry
from src.models.base import OrderContext
from src.models.baselines import CoverageBasedOrderer, RandomOrderer, RecentlyFailedFirstOrderer
from src.models.ml_models import RandomForestOrderer, XGBoostOrderer

log = logging.getLogger(__name__)


def _get_git_changed_files() -> list[str]:
    """Auto-detect changed files in current git workspace."""
    try:
        # Check uncommitted + staged changes first
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            # If no uncommitted changes, check last commit
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        return [f.strip() for f in out.splitlines() if f.strip()]
    except Exception:
        return []


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ml-prioritization", "ML-based Test Prioritization & Flaky Quarantine")
    group.addoption(
        "--prioritize",
        action="store",
        default="none",
        choices=["none", "rf", "xgb", "recent", "coverage", "random"],
        help="Prioritization strategy to order tests: 'rf', 'xgb', 'recent', 'coverage', 'random', or 'none'.",
    )
    group.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to trained model pickle file (.pkl). Defaults to experiments/models/{strategy}.pkl",
    )
    group.addoption(
        "--flaky-registry",
        action="store",
        default="data/flaky_registry.json",
        help="Path to flaky quarantine registry JSON file.",
    )
    group.addoption(
        "--quarantine",
        action="store",
        default="mark",
        choices=["mark", "skip", "none"],
        help="Action for quarantined flaky tests: 'mark' (add @pytest.mark.flaky), 'skip', or 'none'.",
    )
    group.addoption(
        "--changed-files",
        action="store",
        default=None,
        help="Comma-separated list of modified files for prioritization context. If omitted, auto-detected from git.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "flaky: marks test as quarantined flaky test",
    )


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not items:
        return

    # ------------------------------------------------------------------
    # 1. Flaky Test Quarantine
    # ------------------------------------------------------------------
    quarantine_action = config.getoption("--quarantine", "mark")
    registry_path = Path(config.getoption("--flaky-registry", "data/flaky_registry.json"))

    if quarantine_action != "none" and registry_path.exists():
        try:
            registry = FlakyRegistry(registry_path)
            flaky_marker = pytest.mark.flaky
            skip_marker = pytest.mark.skip(reason="Quarantined flaky test")

            quarantined_count = 0
            for item in items:
                # Check nodeid, file location, and function name
                if registry.is_flaky(item.nodeid):
                    if quarantine_action == "skip":
                        item.add_marker(skip_marker, append=False)
                    else:
                        item.add_marker(flaky_marker, append=False)
                    quarantined_count += 1

            if quarantined_count > 0:
                print(
                    f"\n[ML-TP] Quarantined {quarantined_count} flaky test(s) (action={quarantine_action})"
                )
        except Exception as exc:
            log.warning("[ML-TP] Failed to load flaky registry from %s: %s", registry_path, exc)

    # ------------------------------------------------------------------
    # 2. Test Prioritization / Reordering
    # ------------------------------------------------------------------
    strategy = config.getoption("--prioritize", "none")
    if strategy == "none":
        return

    # Determine changed files
    changed_arg = config.getoption("--changed-files", None)
    if changed_arg:
        changed_files = [f.strip() for f in changed_arg.split(",") if f.strip()]
    else:
        changed_files = _get_git_changed_files()

    orderer: Any = None
    if strategy == "random":
        orderer = RandomOrderer()
    elif strategy == "recent":
        orderer = RecentlyFailedFirstOrderer()
    elif strategy == "coverage":
        orderer = CoverageBasedOrderer()
    elif strategy in ("rf", "xgb"):
        model_path_str = config.getoption("--model-path", None)
        if model_path_str:
            model_path = Path(model_path_str)
        else:
            default_name = "random_forest.pkl" if strategy == "rf" else "xgboost.pkl"
            model_path = Path("experiments/models") / default_name

        if model_path.exists():
            try:
                orderer = (
                    RandomForestOrderer.load(model_path)
                    if strategy == "rf"
                    else XGBoostOrderer.load(model_path)
                )
            except Exception as exc:
                print(
                    f"\n[ML-TP] Warning: Could not load model from {model_path} ({exc}). Using recently-failed fallback."
                )
                orderer = RecentlyFailedFirstOrderer()
        else:
            print(
                f"\n[ML-TP] Warning: Model file {model_path} not found. Using recently-failed fallback."
            )
            orderer = RecentlyFailedFirstOrderer()

    if orderer is not None:
        nodeids = [item.nodeid for item in items]
        ctx = OrderContext(
            commit_sha="current",
            changed_files=changed_files,
        )
        ordered_ids = orderer.order(nodeids, ctx)

        # Reorder pytest items to match ordered_ids
        item_map = {item.nodeid: item for item in items}
        ordered_items = []
        for tid in ordered_ids:
            if tid in item_map:
                ordered_items.append(item_map.pop(tid))
        # Append any remaining items that weren't in ordered_ids
        ordered_items.extend(item_map.values())

        items[:] = ordered_items
        print(
            f"\n[ML-TP] Prioritized {len(items)} tests using strategy '{strategy}' (changed_files={len(changed_files)})"
        )
