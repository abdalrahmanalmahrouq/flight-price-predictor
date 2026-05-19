import lightgbm as lgb

def build_model(trial, config):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", config["learning_rate"][0], config["learning_rate"][1], log=True),
        "max_depth":        trial.suggest_int("max_depth", config["max_depth"][0], config["max_depth"][1]),
        "num_leaves":       trial.suggest_int("num_leaves", config["num_leaves"][0], config["num_leaves"][1]),
        "subsample":        trial.suggest_float("subsample", config["subsample"][0], config["subsample"][1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", config["colsample_bytree"][0], config["colsample_bytree"][1]),
        "reg_alpha":        trial.suggest_float("reg_alpha", config["reg_alpha"][0], config["reg_alpha"][1]),
        "reg_lambda":       trial.suggest_float("reg_lambda", config["reg_lambda"][0], config["reg_lambda"][1]),
        "verbosity":  -1,
        "objective":  "regression",
    }
    return lgb.LGBMRegressor(**params, n_estimators=3000)

def fit_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=50),
            lgb.log_evaluation(period=-1)
        ]
    )
    return model
