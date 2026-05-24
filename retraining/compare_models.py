import pandas as pd
import mlflow
from mlflow import MlflowClient
from loguru import logger


def get_champion_metrics(model_name: str) -> dict | None:
    """
    Fetches champion model metrics from MLflow registry.
    Returns dict with MAE, RMSE, R2 or None if champion not found.
    """
    client = MlflowClient()

    try:
        # Get champion version
        champion = client.get_model_version_by_alias(model_name, "champion")
        run_id = champion.run_id

        # Get metrics from the run
        run = client.get_run(run_id)
        metrics = run.data.metrics

        result = {
            "version":  champion.version,
            "run_id":   run_id,
            "MAE":      metrics.get("test_MAE",  metrics.get("val_MAE",  None)),
            "RMSE":     metrics.get("test_RMSE", metrics.get("val_RMSE", None)),
            "R2":       metrics.get("test_R2",   metrics.get("val_R2",   None)),
        }

        logger.info(
            "Champion (v{}) — MAE: {:.0f}, RMSE: {:.0f}, R²: {:.4f}",
            result["version"], result["MAE"], result["RMSE"], result["R2"]
        )
        return result

    except Exception as e:
        logger.warning("Could not fetch champion metrics: {}", str(e))
        return None


def compare_and_promote(
    model_name:         str,
    challenger_run_id:  str,
    challenger_metrics: dict,
    improvement_threshold: float = 0.02,
    artifact_path:         str = "lightgbm_model",
) -> bool:
    """
    Compares challenger vs champion on MAE.
    Promotes challenger if it's better by at least improvement_threshold %.
    Returns True if challenger was promoted.

    improvement_threshold = 0.02 means challenger must be at least 2% better.
    Prevents promoting a model that's only marginally better (could be noise).
    """
    client = MlflowClient()

    challenger_mae = challenger_metrics["MAE"]
    challenger_rmse = challenger_metrics["RMSE"]
    challenger_r2 = challenger_metrics["R2"]

    logger.info(
        "Challenger — MAE: {:.0f}, RMSE: {:.0f}, R²: {:.4f}",
        challenger_mae, challenger_rmse, challenger_r2
    )

    # Get champion metrics
    champion = get_champion_metrics(model_name)

    if champion is None:
        # No champion exists yet — promote automatically
        logger.info("No existing champion — promoting challenger automatically")
        _promote(client, model_name, challenger_run_id, previous_version=None)
        return True

    champion_mae = champion["MAE"]

    # Calculate improvement
    improvement = (champion_mae - challenger_mae) / champion_mae
    logger.info(
        "MAE improvement: {:.2f}% (threshold: {:.2f}%)",
        improvement * 100, improvement_threshold * 100
    )

    if improvement >= improvement_threshold:
        logger.info(
            "Challenger WINS — {:.2f}% better than champion → promoting",
            improvement * 100
        )
        _promote(
            client, model_name,
            challenger_run_id,
            previous_version=champion["version"],
            artifact_path=artifact_path,
        )
        return True
    else:
        logger.info(
            "Champion WINS — challenger not better enough "
            "({:.2f}% improvement < {:.2f}% threshold) → keeping champion",
            improvement * 100, improvement_threshold * 100
        )
        return False


def _promote(
    client,
    model_name:       str,
    challenger_run_id: str,
    previous_version: str | None,
    artifact_path:     str = "lightgbm_model",
) -> None:
    """
    Registers challenger in MLflow and sets champion alias.
    Saves previous champion as previous_champion for rollback.
    """
    # Register challenger model
    model_uri = f"runs:/{challenger_run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri, model_name)
    new_version = registered.version

    logger.info(
        "Registered challenger as {} version {}",
        model_name, new_version
    )

    # Save old champion as previous_champion for rollback
    if previous_version:
        client.set_registered_model_alias(
            model_name, "previous_champion", previous_version
        )
        logger.info(
            "Previous champion (v{}) saved as 'previous_champion' alias",
            previous_version
        )

    # Promote challenger to champion
    client.set_registered_model_alias(
        model_name, "champion", new_version
    )
    logger.info(
        "Challenger (v{}) promoted to 'champion'",
        new_version
    )


def rollback(model_name: str) -> bool:
    """
    Rolls back to previous_champion if current champion is causing issues.
    Returns True if rollback succeeded.
    """
    client = MlflowClient()

    try:
        # Get previous champion
        prev = client.get_model_version_by_alias(model_name, "previous_champion")
        prev_version = prev.version

        # Get current champion version before overwriting
        current = client.get_model_version_by_alias(model_name, "champion")
        current_version = current.version

        # Restore previous champion
        client.set_registered_model_alias(
            model_name, "champion", prev_version
        )

        logger.info(
            "ROLLBACK complete — restored v{} as champion (was v{})",
            prev_version, current_version
        )
        return True

    except Exception as e:
        logger.error("Rollback FAILED — {}", str(e))
        return False
