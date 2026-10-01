"""
src/ci_plugin/__init__.py
==========================
CI Integration package for ML Test Prioritization and Flaky Quarantine.
"""

from src.ci_plugin.plugin import (
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_configure,
)

__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_collection_modifyitems",
]
