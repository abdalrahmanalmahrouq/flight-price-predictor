import importlib
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error


class ModelTrainer:
    CONFIGS = {
        "lightgbm": "configs.lightgbm",
        "xgboost": "configs.xgboost",
        "catboost": "configs.catboost",
        "linear": "configs.linear",
    }

    def __init__(self, model_name: str = "lightgbm", models_dir: str = "models"):
        if model_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(self.CONFIGS.keys())}"
            )
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.model = None
        self.best_params = None
        self.best_trial = None
        self.config = importlib.import_module(self.CONFIGS[model_name])

    def train(self, X_train, y_train, X_val, y_val) -> "ModelTrainer":
        best_trial = self._tune(X_train, y_train, X_val, y_val)
        self.best_params = best_trial.params
        self.best_trial = best_trial
        self.model = self._train_final(X_train, y_train, X_val, y_val)
        return self

    def save(self, filename: str = None):
        if filename is None:
            filename = f"{self.model_name}_model.joblib"

        path = self.models_dir / filename
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, filename: str = None):
        if filename is None:
            filename = f"{self.model_name}_model.joblib"
        path = self.models_dir / filename
        self.model = joblib.load(path)
        return self

    def _tune(self, X_train, y_train, X_val, y_val) -> dict:
        def objective(trial):

            model = self.config.build_model(trial)
            model = self.config.fit_model(model, X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
            rmse = round(np.sqrt(mean_squared_error(y_val, preds)))
            return rmse

        def mlflow_callback(study, trial):
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("val_RMSE", trial.value)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective, n_trials=50, show_progress_bar=True, callbacks=[mlflow_callback]
        )

        return study.best_trial

    def _train_final(self, X_train, y_train, X_val, y_val):

        class BestTrial:
            def __init__(self, params):
                self.params = params

            def suggest_float(self, name, *args, **kwargs):
                return self.params[name]

            def suggest_int(self, name, *args, **kwargs):
                return self.params[name]

        trial = BestTrial(self.best_params)
        model = self.config.build_model(trial)
        model = self.config.fit_model(model, X_train, y_train, X_val, y_val)
        return model
