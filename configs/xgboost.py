import xgboost as xgb


def build_model(trial):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 10.0),
        "verbosity": 0,
        "objective": "reg:squarederror",
    }
    return xgb.XGBRegressor(**params, n_estimators=3000, early_stopping_rounds=50)

def fit_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose= False
    )
    return model

