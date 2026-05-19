import catboost as ctboost

def build_model(trial, config):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", config["learning_rate"][0], config["learning_rate"][1], log=True),
        "depth":         trial.suggest_int("depth", config["depth"][0], config["depth"][1]),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", config["l2_leaf_reg"][0], config["l2_leaf_reg"][1]),
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
