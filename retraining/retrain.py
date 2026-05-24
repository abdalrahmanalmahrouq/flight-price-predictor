import argparse
import sys
from pathlib import Path
import os
import mlflow
import pandas as pd
import mlflow.lightgbm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flight_predictor.logger import setup_logger, logger
from flight_predictor.data_loader import DataLoader
from flight_predictor.preprocessor import Preprocessor
from flight_predictor.feature_engineer import FeatureEngineer
from flight_predictor.target_encoder import TargetEncoder
from flight_predictor.trainer import ModelTrainer
from flight_predictor.evaluator import ModelEvaluator
from retraining.validate_data import validate_new_data
from retraining.compare_models import compare_and_promote, rollback

from sklearn.model_selection import train_test_split

setup_logger(level="INFO")

# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = "lightgbm_model"
MODEL_TYPE          = "lightgbm"


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def load_and_validate(data_path: str | None) -> pd.DataFrame:
    """
    Loads new data and validates it.
    If data_path is None → uses existing processed data.
    """
    if data_path:
        logger.info("Loading new data from {}", data_path)
        new_df = pd.read_csv(data_path)

        is_valid, errors = validate_new_data(new_df)
        if not is_valid:
            logger.error("Data validation failed — aborting retraining")
            for err in errors:
                logger.error("  {}", err)
            sys.exit(1)

        logger.info("New data validated — {} rows", len(new_df))
        return new_df
    else:
        logger.info("No new data provided — retraining on existing processed data")
        return None


def preprocess(new_df: pd.DataFrame | None) -> tuple:
    if new_df is None:
        # Use existing processed splits
        logger.info("Loading existing processed splits...")
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_val   = pd.read_csv("data/processed/X_val.csv")
        X_test  = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
        y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

        logger.info(
            "Splits loaded — train: {}, val: {}, test: {}",
            len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # New data provided — run full pipeline like run.py
    logger.info("Running full preprocessing pipeline on new data...")

    loader = DataLoader()
    existing_df = loader.load()

    # Combine existing + new
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    logger.info("Combined: {} rows", len(combined_df))

    # Match run.py exactly
    preprocessor = Preprocessor()
    df_clean = preprocessor.fit_transform(combined_df)

    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df_clean)

    # Split — match run.py split() function exactly
    X = df_featured.drop(columns=["price"])
    y = df_featured["price"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    # Target encode — match run.py exactly
    encoder = TargetEncoder()
    X_train = encoder.fit_transform(X_train, y_train)
    X_val   = encoder.transform(X_val)
    X_test  = encoder.transform(X_test)

    logger.info(
        "Preprocessing complete — train: {}, val: {}, test: {}",
        len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
def train_challenger(X_train, y_train, X_val, y_val) -> tuple[ModelTrainer, str]:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("flight-price-predictor-retraining")

    # Load config.yml — needed for Optuna ranges
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["models"][MODEL_TYPE]  

    logger.info("Starting challenger training...")

    with mlflow.start_run(run_name="challenger") as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: {}", run_id)

        trainer = ModelTrainer(
            model_name=MODEL_TYPE,
            model_config=model_config,   
        )
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save()

        mlflow.log_param("model_name", MODEL_TYPE)
        mlflow.log_params(trainer.best_params)

        mlflow.lightgbm.log_model(
            trainer.model,
            name=f"{MODEL_TYPE}_model",
            registered_model_name=f"{MODEL_TYPE}_model"
        )

        logger.info(
            "Challenger trained — best_iteration: {}",
            trainer.model.best_iteration_
        )

    return trainer, run_id

def evaluate_challenger(
    trainer, run_id,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
) -> dict:

    evaluator = ModelEvaluator()

    val_metrics  = evaluator.evaluate(trainer.model, X_val,  y_val)
    test_metrics = evaluator.evaluate(trainer.model, X_test, y_test)

    report = evaluator.report(
        trainer.model,
        {"val": (X_val, y_val), "test": (X_test, y_test)}
    )
    logger.info("Challenger evaluation:\n{}", report.to_string())

    # Log metrics to existing MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_MAE",   val_metrics["MAE"])
        mlflow.log_metric("val_RMSE",  val_metrics["RMSE"])
        mlflow.log_metric("val_R2",    val_metrics["R2"])
        mlflow.log_metric("test_MAE",  test_metrics["MAE"])
        mlflow.log_metric("test_RMSE", test_metrics["RMSE"])
        mlflow.log_metric("test_R2",   test_metrics["R2"])

    logger.info(
        "Challenger test — MAE: {:.0f}, RMSE: {:.0f}, R²: {:.4f}",
        test_metrics["MAE"], test_metrics["RMSE"], test_metrics["R2"]
    )
    return test_metrics

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flight Price Retraining Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to new data CSV. If not provided, retrains on existing data."
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Retrain on existing processed data without new data"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous champion model"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.02,
        help="Minimum improvement required to promote challenger (default: 2%%)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RETRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    # ── Rollback mode ─────────────────────────────────────────────────────────
    if args.rollback:
        logger.info("ROLLBACK MODE — restoring previous champion")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        success = rollback(MODEL_NAME)
        if success:
            logger.info("Rollback complete — restart API to load previous champion")
        else:
            logger.error("Rollback failed — check MLflow registry")
        return

    # ── Step 1 — Load and validate data ───────────────────────────────────────
    logger.info("Step 1 — Loading and validating data")
    data_path = args.data_path if not args.use_existing else None
    new_df = load_and_validate(data_path)

    # ── Step 2 — Preprocess ───────────────────────────────────────────────────
    logger.info("Step 2 — Preprocessing")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(new_df)

    # ── Step 3 — Train challenger ─────────────────────────────────────────────
    logger.info("Step 3 — Training challenger model")
    trainer, run_id = train_challenger(X_train, y_train, X_val, y_val)

    # ── Step 4 — Evaluate challenger ──────────────────────────────────────────
    logger.info("Step 4 — Evaluating challenger")
    test_metrics = evaluate_challenger(
        trainer, run_id,       # ← moved run_id here
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )

    # ── Step 5 — Compare and promote ─────────────────────────────────────────
    logger.info("Step 5 — Comparing challenger vs champion")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    promoted = compare_and_promote(
        model_name=MODEL_NAME,
        challenger_run_id=run_id,
        challenger_metrics=test_metrics,
        improvement_threshold=args.improvement_threshold,
        artifact_path=f"{MODEL_TYPE}_model", 
    )

    # ── Step 6 — Save challenger model to disk ────────────────────────────────
    trainer.save(filename=f"{MODEL_TYPE}_model.joblib")
    logger.info("Challenger model saved to disk")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if promoted:
        logger.info("PIPELINE COMPLETE — challenger promoted to champion")
        logger.info("Restart API to serve new champion:")
        logger.info("  docker compose restart api")
    else:
        logger.info("PIPELINE COMPLETE — champion unchanged")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
