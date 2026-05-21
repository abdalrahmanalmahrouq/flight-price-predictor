import argparse
import sys
import yaml
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Evidently AI imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from flight_predictor.data_loader import DataLoader
from flight_predictor.evaluator import ModelEvaluator
from flight_predictor.explainer import Explainer
from flight_predictor.feature_engineer import FeatureEngineer
from flight_predictor.preprocessor import Preprocessor
from flight_predictor.target_encoder import TargetEncoder
from flight_predictor.trainer import ModelTrainer
from flight_predictor.logger import logger, setup_logger

setup_logger(level="DEBUG")

LOG_MODEL = {
    "lightgbm": mlflow.lightgbm.log_model,
    "xgboost": mlflow.xgboost.log_model,
    "catboost": mlflow.catboost.log_model,
    "linear": mlflow.sklearn.log_model,
}

def split(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_evidently_monitoring(train_df, test_df):
    logger.info("Generating Evidently AI Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    drift_report.run(reference_data=train_df, current_data=test_df)
    
    report_path = "evidently_drift_report.html"
    drift_report.save_html(report_path)
    mlflow.log_artifact(report_path)
    logger.info("Drift report logged to S3 successfully.")

if __name__ == "__main__":
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", help="Specific model to train")
    parser.add_argument("--full-training", action="store_true", help="Train all enabled models in config.yml")
    args = parser.parse_args()

    # 2. Load Configuration
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("flight-price-predictor")

    # 3. Data Pipeline (Runs once for all models)
    logger.info("Loading data...")
    loader = DataLoader()
    df_raw = loader.load()

    logger.info("Preprocessing...")
    preprocessor = Preprocessor()
    df_clean = preprocessor.fit_transform(df_raw)

    logger.info("Feature engineering...")
    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df_clean)

    logger.info("Splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split(df_featured)

    logger.info("Encoding...")
    encoder = TargetEncoder()
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)

    # 4. Determine which models to run
    models_to_run = []
    if args.full_training:
        models_to_run = [m for m, c in config["models"].items() if c.get("enabled")]
    else:
        if config["models"].get(args.model, {}).get("enabled"):
            models_to_run = [args.model]
        else:
            logger.error(f"Model {args.model} is not enabled in config.yml")
            sys.exit(1)

    # 5. Training Loop
    for model_name in models_to_run:
        # Let MLflow auto-generate the run name!
        with mlflow.start_run():
            logger.info("=========================================")
            logger.info("Training Model: {}", model_name)
            
            trainer = ModelTrainer(model_name=model_name, model_config=config["models"][model_name])
            trainer.train(X_train_enc, y_train, X_val_enc, y_val)
            trainer.save()

            logger.info("Logging params...")
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(trainer.best_params)
            mlflow.log_param("best_trial_number", trainer.best_trial.number)
            mlflow.log_param("best_trial_RMSE", round(trainer.best_trial.value, 2))
            mlflow.log_param("n_trials", 50)

            logger.info("Evaluating...")
            evaluator = ModelEvaluator()
            val_metrics = evaluator.evaluate(trainer.model, X_val_enc, y_val)
            test_metrics = evaluator.evaluate(trainer.model, X_test_enc, y_test)
            logger.debug(
                evaluator.report(
                    trainer.model, {"val": (X_val_enc, y_val), "test": (X_test_enc, y_test)}
                )
            )

            logger.info("Logging metrics...")
            mlflow.log_metric("val_MAE", val_metrics["MAE"])
            mlflow.log_metric("val_RMSE", val_metrics["RMSE"])
            mlflow.log_metric("val_R2", val_metrics["R2"])
            mlflow.log_metric("test_MAE", test_metrics["MAE"])
            mlflow.log_metric("test_RMSE", test_metrics["RMSE"])
            mlflow.log_metric("test_R2", test_metrics["R2"])

            TREE_MODELS = ["lightgbm", "xgboost", "catboost"]
            if model_name in TREE_MODELS:
                logger.info("Explaining...")
                try:
                    explainer = Explainer()
                    explainer.fit(trainer.model, X_test_enc.iloc[:1000])
                    explainer.summary_bar()
                    explainer.summary_dot()
                    explainer.waterfall()

                    logger.info("Logging artifacts...")
                    mlflow.log_artifact("reports/figures/shap_summary_bar.png")
                    mlflow.log_artifact("reports/figures/shap_summary_dot.png")
                    mlflow.log_artifact("reports/figures/shap_waterfall.png")
                except Exception as e:
                    logger.info("SHAP failed for {}: {} — skipping plots.", model_name, e)

            mlflow.log_artifact(f"models/{model_name}_model.joblib")
            logger.info("Logging model...")
            
            # Use registered_model_name directly to avoid UI mapping issues!
            LOG_MODEL[model_name](
                trainer.model, 
                name=f"{model_name}_model",
                registered_model_name=f"{model_name}_model"
            )

    # 6. Evidently AI Drift Monitoring
    # Combine X and y so Evidently can calculate Target Drift
    train_for_drift = X_train.copy()
    train_for_drift['price'] = y_train
    test_for_drift = X_test.copy()
    test_for_drift['price'] = y_test
    
    with mlflow.start_run(run_name="evidently-monitoring"):
        run_evidently_monitoring(train_for_drift, test_for_drift)

    logger.info("Done.")
