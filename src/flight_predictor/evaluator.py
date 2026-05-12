import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:

    def evaluate(self, model, X, y) -> dict:
        preds = model.predict(X)
        return {
            "MAE": round(mean_absolute_error(y, preds)),
            "RMSE": round(np.sqrt(mean_squared_error(y, preds))),
            "R2": round(r2_score(y, preds), 4)
        }

    def report(self, model, splits: dict) -> pd.DataFrame:
        rows = {}
        for split_name, (X, y) in splits.items():
            rows[split_name] = self.evaluate(model, X, y)
        return pd.DataFrame(rows).T
