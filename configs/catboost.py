import catboost as ctboost 

def build_model(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth":         trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 0.0, 10.0),
        "verbose": 0,
    }
    return ctboost.CatBoostRegressor(
        **params,
        iterations=3000,
        early_stopping_rounds=50,
        eval_metric="RMSE"
    )

def fit_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )
    return model