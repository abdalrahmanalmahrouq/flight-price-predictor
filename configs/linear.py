from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
    }
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(**params)),
    ])
def fit_model(model, X_train, y_train, X_val, y_val):

    model.fit(X_train, y_train)
    return model

