import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from flight_predictor.trainer import ModelTrainer


# ── Fake config module 
# Simulates what configs.lightgbm would provide
# This lets us test ModelTrainer without needing real config files

class FakeModel:
    """Minimal model that records fit calls and returns fixed predictions."""

    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        # always predict 5000.0 — fixed value we control
        return np.full(len(X), 5000.0)


def fake_build_model(trial):
    """Fake build_model — ignores trial params, returns FakeModel."""
    return FakeModel()


def fake_fit_model(model, X_train, y_train, X_val, y_val):
    """Fake fit_model — calls fit and returns model."""
    model.fit(X_train, y_train)
    return model


# ── Fixtures 

@pytest.fixture
def small_data():
    """200 rows of random but realistic flight features."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        "stops_numeric":    np.random.randint(0, 3, n),
        "duration_minutes": np.random.randint(60, 300, n),
        "departure_hour":   np.random.randint(0, 23, n),
        "arrival_hour":     np.random.randint(0, 23, n),
        "month":            np.random.randint(2, 4, n),
        "day":              np.random.randint(0, 7, n),
        "is_weekend":       np.random.randint(0, 2, n),
        "is_business":      np.random.randint(0, 2, n),
        "airline_encoded":  np.random.uniform(3000, 52000, n),
        "from_encoded":     np.random.uniform(3000, 52000, n),
        "to_encoded":       np.random.uniform(3000, 52000, n),
    })

    y = pd.Series(np.random.uniform(1000, 120000, n))

    X_train, X_val = X.iloc[:160], X.iloc[160:]
    y_train, y_val = y.iloc[:160], y.iloc[160:]

    return X_train, X_val, y_train, y_val


@pytest.fixture
def fake_config():
    """A mock config module with build_model and fit_model."""
    config = MagicMock()
    config.build_model.side_effect = fake_build_model
    config.fit_model.side_effect = fake_fit_model
    return config


@pytest.fixture
def trainer(fake_config):
    """ModelTrainer with lightgbm name but fake config injected."""
    with patch("importlib.import_module", return_value=fake_config):
        t = ModelTrainer(model_name="lightgbm", models_dir="/tmp/test_models")
    return t


# ── __init__ tests 

def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelTrainer(model_name="random_forest")


def test_known_models_do_not_raise():
    for name in ["lightgbm", "xgboost", "catboost", "linear"]:
        with patch("importlib.import_module", return_value=MagicMock()):
            trainer = ModelTrainer(model_name=name)
            assert trainer.model_name == name


def test_initial_state_is_none(trainer):
    assert trainer.model is None
    assert trainer.best_params is None
    assert trainer.best_trial is None


# ── train() tests 

def test_model_exists_after_training(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.model is not None


def test_best_params_exist_after_training(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.best_params is not None
    assert isinstance(trainer.best_params, dict)


def test_best_trial_exists_after_training(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.best_trial is not None


def test_best_params_matches_best_trial_params(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.best_params == trainer.best_trial.params


# ── predict tests 

def test_model_can_predict_after_training(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    preds = trainer.model.predict(X_val)
    assert len(preds) == len(X_val)


def test_predictions_are_positive(trainer, small_data):
    X_train, X_val, y_train, y_val = small_data
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    preds = trainer.model.predict(X_val)
    assert (preds > 0).all()


# ── save() / load() tests 

def test_default_filename_uses_model_name(trainer, small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer.models_dir = tmp_path
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    trainer.save()
    assert (tmp_path / "lightgbm_model.joblib").exists()


def test_custom_filename_respected(trainer, small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer.models_dir = tmp_path
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    trainer.save(filename="my_custom_model.joblib")
    assert (tmp_path / "my_custom_model.joblib").exists()


def test_models_dir_created_if_missing(trainer, small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer.models_dir = tmp_path / "does" / "not" / "exist"
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    trainer.save()
    assert trainer.models_dir.exists()


def test_model_loaded_from_disk(trainer, small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer.models_dir = tmp_path
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)
    trainer.save()

    # load into a brand new trainer instance
    with patch("importlib.import_module", return_value=trainer.config):
        new_trainer = ModelTrainer(model_name="lightgbm", models_dir=str(tmp_path))
    new_trainer.load()
    assert new_trainer.model is not None


def test_loaded_model_predictions_match(trainer, small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer.models_dir = tmp_path
    with patch("mlflow.start_run"):
        trainer.train(X_train, y_train, X_val, y_val)

    original_preds = trainer.model.predict(X_val)
    trainer.save()

    with patch("importlib.import_module", return_value=trainer.config):
        new_trainer = ModelTrainer(model_name="lightgbm", models_dir=str(tmp_path))
    new_trainer.load()
    loaded_preds = new_trainer.model.predict(X_val)

    np.testing.assert_array_equal(original_preds, loaded_preds)