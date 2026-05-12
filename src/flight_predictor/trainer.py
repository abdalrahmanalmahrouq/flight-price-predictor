import joblib
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow 

class ModelTrainer:

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.best_params = None
        self.best_trial = None
        
    def train(self, X_train, y_train, X_val, y_val) -> "ModelTrainer":
        best_trial = self._tune(X_train, y_train, X_val, y_val)
        self.best_params = best_trial.params
        self.best_trial = best_trial
        self.model = self._train_final(X_train, y_train, X_val, y_val)
        return self

    def save(self, filename: str = "tuned_lgb_model.joblib"):
        path = self.models_dir / filename
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, filename: str = "tuned_lgb_model.joblib"):
        path = self.models_dir / filename
        self.model = joblib.load(path)
        return self

    def _tune(self, X_train, y_train, X_val, y_val) -> dict:
        def objective(trial):

            params = {
                "objective": "regression",
                "verbosity": -1,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 6, 15), # changeed 
                "num_leaves": trial.suggest_int("num_leaves", 100, 400), # changed
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }

            model = lgb.LGBMRegressor(**params, n_estimators=3000) #changed
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=True),
                    lgb.log_evaluation(period=-1)
                ]
            )

            preds = model.predict(X_val)
            rmse = round(np.sqrt(mean_squared_error(y_val, preds)))
            return rmse

        def mlflow_callback(study, trial):
            with mlflow.start_run(
                run_name=f"trial-{trial.number}",
                nested=True       
            ):
                mlflow.log_params(trial.params)
                mlflow.log_metric("val_RMSE", trial.value)
        study = optuna.create_study(direction="minimize")

        study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[mlflow_callback])
        return study.best_trial

    def _train_final(self, X_train, y_train, X_val, y_val):
        model = lgb.LGBMRegressor(
            **self.best_params,
            n_estimators=3000,
            objective="regression"
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        return model
