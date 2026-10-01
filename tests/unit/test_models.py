"""
tests/unit/test_models.py
=========================
Unit tests for RandomForestOrderer, XGBoostOrderer, and model serialization.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer
from src.models.base import OrderContext
from src.models.ml_models import RandomForestOrderer, XGBoostOrderer


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "test_id": [f"test_{i % 10}" for i in range(n_samples)],
            "commit_sha": [f"sha_{i // 10}" for i in range(n_samples)],
            "commit_ts": pd.date_range("2021-01-01", periods=n_samples, freq="D"),
            "author": [f"author_{i % 3}" for i in range(n_samples)],
            "outcome": np.random.binomial(1, 0.1, size=n_samples),
            "duration_s": np.random.exponential(1.0, size=n_samples),
            "files_changed": np.random.randint(1, 5, size=n_samples),
            "lines_added": np.random.randint(0, 100, size=n_samples),
            "lines_deleted": np.random.randint(0, 50, size=n_samples),
            "test_files_changed": np.random.randint(0, 2, size=n_samples),
            "src_files_changed": np.random.randint(1, 4, size=n_samples),
            "changed_file_list": '["src/foo.py"]',
            "prev_1": np.random.binomial(1, 0.1, size=n_samples),
            "prev_2": np.random.binomial(1, 0.1, size=n_samples),
            "prev_3": np.random.binomial(1, 0.1, size=n_samples),
            "prev_4": np.random.binomial(1, 0.1, size=n_samples),
            "prev_5": np.random.binomial(1, 0.1, size=n_samples),
        }
    )
    return df


class TestMLModels:
    def test_random_forest_fit_predict_order(self, dummy_train_data: pd.DataFrame, tmp_path: Path):
        fe = FeatureEngineer()
        X = fe.fit_transform(dummy_train_data)
        y = dummy_train_data["outcome"].values

        rf = RandomForestOrderer(feature_engineer=fe, n_estimators=10)
        rf.fit(X, y)

        test_ids = [f"test_{i}" for i in range(10)]
        ctx = OrderContext(
            commit_sha="test_sha",
            changed_files=["src/foo.py"],
            test_history=dummy_train_data.head(10),
        )
        ordered = rf.order(test_ids, ctx)

        assert len(ordered) == len(test_ids)
        assert set(ordered) == set(test_ids)

        # Save and load
        save_path = tmp_path / "rf.pkl"
        rf.save(save_path)
        assert save_path.exists()

        loaded_rf = RandomForestOrderer.load(save_path)
        assert loaded_rf.name == "random_forest"

    def test_xgboost_fit_predict_order(self, dummy_train_data: pd.DataFrame, tmp_path: Path):
        fe = FeatureEngineer()
        X = fe.fit_transform(dummy_train_data)
        y = dummy_train_data["outcome"].values

        xgb = XGBoostOrderer(feature_engineer=fe, n_estimators=10, max_depth=3)
        xgb.fit(X, y)

        test_ids = [f"test_{i}" for i in range(10)]
        ctx = OrderContext(
            commit_sha="test_sha",
            changed_files=["src/foo.py"],
            test_history=dummy_train_data.head(10),
        )
        ordered = xgb.order(test_ids, ctx)

        assert len(ordered) == len(test_ids)
        assert set(ordered) == set(test_ids)

        # Save and load
        save_path = tmp_path / "xgb.pkl"
        xgb.save(save_path)
        assert save_path.exists()

        loaded_xgb = XGBoostOrderer.load(save_path)
        assert loaded_xgb.name == "xgboost"
