# tests/test_evaluator.py

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from flight_predictor.evaluator import ModelEvaluator


class DummyModel:
    """A fake model that returns predictions we control completely."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


@pytest.fixture
def perfect_model():
    # model that predicts exactly correct values
    return DummyModel(predictions=np.array([1000.0, 2000.0, 3000.0]))


@pytest.fixture
def imperfect_model():
    # model that predicts with known errors
    return DummyModel(predictions=np.array([1100.0, 1900.0, 3200.0]))


@pytest.fixture
def X():
    # evaluator only passes X to model.predict() — content doesn't matter
    return pd.DataFrame({"feature1": [1, 2, 3]})


@pytest.fixture
def y():
    return pd.Series([1000.0, 2000.0, 3000.0])


def test_evaluate_returns_dict(perfect_model, X, y):
    result = ModelEvaluator().evaluate(perfect_model, X, y)
    assert isinstance(result, dict)


def test_evaluate_has_correct_keys(perfect_model, X, y):
    result = ModelEvaluator().evaluate(perfect_model, X, y)
    assert "MAE" in result
    assert "RMSE" in result
    assert "R2" in result


def test_perfect_model_mae_is_zero(perfect_model, X, y):
    result = ModelEvaluator().evaluate(perfect_model, X, y)
    assert result["MAE"] == 0


def test_perfect_model_r2_is_one(perfect_model, X, y):
    result = ModelEvaluator().evaluate(perfect_model, X, y)
    assert result["R2"] == 1.0


def test_imperfect_model_mae_correct(imperfect_model, X, y):
    result = ModelEvaluator().evaluate(imperfect_model, X, y)
    # errors: 100, 100, 200 → mean = 133
    assert result["MAE"] == 133


def test_imperfect_model_r2_below_one(imperfect_model, X, y):
    result = ModelEvaluator().evaluate(imperfect_model, X, y)
    assert result["R2"] < 1.0


def test_report_returns_dataframe(perfect_model, X, y):
    report = ModelEvaluator().report(perfect_model, {
        "val": (X, y),
        "test": (X, y)
    })
    assert isinstance(report, pd.DataFrame)


def test_report_has_correct_shape(perfect_model, X, y):
    report = ModelEvaluator().report(perfect_model, {
        "val": (X, y),
        "test": (X, y)
    })
    # 2 splits × 3 metrics
    assert report.shape == (2, 3)


def test_report_index_matches_split_names(perfect_model, X, y):
    report = ModelEvaluator().report(perfect_model, {
        "val": (X, y),
        "test": (X, y)
    })
    assert "val" in report.index
    assert "test" in report.index
