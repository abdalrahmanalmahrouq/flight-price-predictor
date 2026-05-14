import lightgbm as lgb


def build_model(trial):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 6, 15),
        "num_leaves":       trial.suggest_int("num_leaves", 100, 400),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 10.0),
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

