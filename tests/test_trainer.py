# tests/test_trainer.py

import pytest
import numpy as np
import pandas as pd
from flight_predictor.trainer import ModelTrainer


@pytest.fixture
def small_data():
    """Small but realistic data to train a quick LightGBM model."""
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

    # split into train and val
    X_train, X_val = X.iloc[:160], X.iloc[160:]
    y_train, y_val = y.iloc[:160], y.iloc[160:]

    return X_train, X_val, y_train, y_val


def test_model_exists_after_training(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.model is not None


def test_best_params_exist_after_training(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.best_params is not None
    assert isinstance(trainer.best_params, dict)


def test_best_params_has_correct_keys(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    expected_keys = [
        "learning_rate", "max_depth", "num_leaves",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
    ]
    for key in expected_keys:
        assert key in trainer.best_params


def test_model_can_predict_after_training(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    preds = trainer.model.predict(X_val)
    assert len(preds) == len(X_val)


def test_predictions_are_positive(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    preds = trainer.model.predict(X_val)
    assert (preds > 0).all()


def test_model_saved_to_disk(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.save()
    assert (tmp_path / "tuned_lgb_model.joblib").exists()


def test_model_loaded_from_disk(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data

    # train and save
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.save()

    # load into a brand new trainer instance
    new_trainer = ModelTrainer(models_dir=str(tmp_path))
    new_trainer.load()

    assert new_trainer.model is not None


def test_loaded_model_predictions_match(small_data, tmp_path):
    X_train, X_val, y_train, y_val = small_data

    # train, save, get original predictions
    trainer = ModelTrainer(models_dir=str(tmp_path))
    trainer.train(X_train, y_train, X_val, y_val)
    original_preds = trainer.model.predict(X_val)
    trainer.save()

    # load and predict again
    new_trainer = ModelTrainer(models_dir=str(tmp_path))
    new_trainer.load()
    loaded_preds = new_trainer.model.predict(X_val)

    np.testing.assert_array_equal(original_preds, loaded_preds)
