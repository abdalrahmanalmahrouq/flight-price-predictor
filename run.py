import sys
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost

from sklearn.model_selection import train_test_split
from flight_predictor.data_loader import DataLoader
from flight_predictor.preprocessor import Preprocessor
from flight_predictor.feature_engineer import FeatureEngineer
from flight_predictor.target_encoder import TargetEncoder
from flight_predictor.trainer import ModelTrainer
from flight_predictor.evaluator import ModelEvaluator
from flight_predictor.explainer import Explainer


MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"

LOG_MODEL = {
    "lightgbm": mlflow.lightgbm.log_model,
    "xgboost":  mlflow.xgboost.log_model,
    "catboost":  mlflow.catboost.log_model,
    "linear":    mlflow.sklearn.log_model,
}

def split(df):
    X = df.drop(columns=["price"])
    y = df["price"]


    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Then split remaining into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    mlflow.set_experiment("flight-price-predictor")

    with mlflow.start_run(run_name=f"{MODEL_NAME}-run"):
        print (f"Model: {MODEL_NAME}")

        print("Loading data...")
        loader = DataLoader()
        df_raw = loader.load()

        print("Preprocessing...")
        preprocessor = Preprocessor()
        df_clean = preprocessor.fit_transform(df_raw)

        print("Feature engineering...")
        engineer = FeatureEngineer()
        df_featured = engineer.fit_transform(df_clean)

        print("Splitting...")
        X_train, X_val, X_test, y_train, y_val, y_test = split(df_featured)

        print("Encoding...")
        encoder = TargetEncoder()
        X_train = encoder.fit_transform(X_train, y_train)
        X_val = encoder.transform(X_val)
        X_test = encoder.transform(X_test)

        print("Training...")
        trainer = ModelTrainer(model_name=MODEL_NAME)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save()


        print("Loging params...")
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_params(trainer.best_params)
        mlflow.log_param("best_trial_number", trainer.best_trial.number)
        mlflow.log_param("best_trial_RMSE",   round(trainer.best_trial.value, 2))
        mlflow.log_param("n_trials", 50)


        print("Evaluating...")
        evaluator = ModelEvaluator()
        val_metrics = evaluator.evaluate(trainer.model, X_val, y_val)
        test_metrics = evaluator.evaluate(trainer.model, X_test, y_test)
        print(evaluator.report(trainer.model, {
            "val":  (X_val,  y_val),
            "test": (X_test, y_test)
        }))

        print("Logging metrics...")
        mlflow.log_metric("val_MAE",   val_metrics["MAE"])
        mlflow.log_metric("val_RMSE",  val_metrics["RMSE"])
        mlflow.log_metric("val_R2",    val_metrics["R2"])

        mlflow.log_metric("test_MAE",  test_metrics["MAE"])
        mlflow.log_metric("test_RMSE", test_metrics["RMSE"])
        mlflow.log_metric("test_R2",   test_metrics["R2"])


        TREE_MODELS = ["lightgbm", "xgboost", "catboost"]
        if(MODEL_NAME in TREE_MODELS):
            print("Explaining...")
            try:
                explainer = Explainer()
                explainer.fit(trainer.model, X_test.iloc[:1000])
                explainer.summary_bar()
                explainer.summary_dot()
                explainer.waterfall()

                print("Logging artifacts...")
                mlflow.log_artifact("reports/figures/shap_summary_bar.png")
                mlflow.log_artifact("reports/figures/shap_summary_dot.png")
                mlflow.log_artifact("reports/figures/shap_waterfall.png")
            except Exception as e:
                print(f"SHAP failed for {MODEL_NAME}: {e} — skipping plots.")    
        
        mlflow.log_artifact(f"models/{MODEL_NAME}_model.joblib")
        print("Logging model...")
        LOG_MODEL[MODEL_NAME](trainer.model, name=f"{MODEL_NAME}_model")

        print("Done.")

    
