"""
Evidently AI Monitoring — Phase 14
Generates HTML drift reports and runs automated test suites.

Usage:
    python monitoring/evidently_monitor.py              # real data vs training
    python monitoring/evidently_monitor.py --simulate   # inject drift first
"""
import joblib
import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from loguru import logger

from evidently.report import Report
from evidently.test_suite import TestSuite

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.test_preset import (
    DataDriftTestPreset,
    DataQualityTestPreset,
    RegressionTestPreset,
)
from evidently.pipeline.column_mapping import ColumnMapping

from flight_predictor.logger import setup_logger


setup_logger(level="INFO")

# ── Config ────────────────────────────────────────────────────────────────────

REFERENCE_PATH  = "data/processed/X_train.csv"
CURRENT_PATH    = "data/processed/X_test.csv"
DB_PATH         = "logs/predictions.db"
REPORTS_DIR     = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "is_business", "stops_numeric", "duration_minutes",
    "departure_hour", "arrival_hour", "month", "day",
    "is_weekend", "airline_encoded", "from_encoded", "to_encoded",
]

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_reference() -> pd.DataFrame:
    """Load training data as reference distribution."""
    df = pd.read_csv(REFERENCE_PATH)
    logger.info("Reference data loaded — {} rows", len(df))
    return df[FEATURE_COLS]


def load_current(simulate: bool = False) -> pd.DataFrame:
    """Load current data — test set or predictions from DB."""
    df = pd.read_csv(CURRENT_PATH)
    df = df[FEATURE_COLS]

    if simulate:
        logger.info("Injecting artificial drift for simulation...")
        df = df.copy()
        df["duration_minutes"] = (df["duration_minutes"] * 1.6).astype(int)
        n_flip = int(len(df) * 0.4)
        economy_idx = df[df["is_business"] == 0].sample(n_flip).index
        df.loc[economy_idx, "is_business"] = 1
        df["departure_hour"] = (df["departure_hour"] * 0.5).astype(int)
        logger.info("Drift injected — duration +60%, is_business flipped 40%")

    logger.info("Current data loaded — {} rows", len(df))
    return df


def load_predictions_from_db() -> pd.DataFrame | None:
    """Load logged predictions from SQLite for concept drift monitoring."""
    if not Path(DB_PATH).exists():
        logger.warning("No predictions DB found at {}", DB_PATH)
        return None

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        logger.warning("Predictions table is empty")
        return None

    logger.info("Loaded {} predictions from DB", len(df))
    return df


# ── Report 1 — Data Drift ─────────────────────────────────────────────────────

def run_data_drift_report(reference: pd.DataFrame,
                          current: pd.DataFrame) -> None:
    """
    Generates interactive HTML report showing feature distribution
    comparisons between reference (training) and current (production) data.
    Runs KS test on continuous features, PSI on categorical features.
    """
    logger.info("Running Data Drift Report...")

    report = Report(metrics=[
        DataDriftPreset(),           # KS + PSI per feature
    ])

    report.run(reference_data=reference, current_data=current)

    output_path = REPORTS_DIR / "data_drift.html"
    report.save_html(str(output_path))
    logger.info("Data Drift Report saved → {}", output_path)


# ── Report 2 — Data Quality ───────────────────────────────────────────────────

def run_data_quality_report(reference: pd.DataFrame,
                            current: pd.DataFrame) -> None:
    """
    Generates HTML report showing data health:
    missing values, duplicates, outliers, value distributions.
    """
    logger.info("Running Data Quality Report...")

    report = Report(metrics=[
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=current)

    output_path = REPORTS_DIR / "data_quality.html"
    report.save_html(str(output_path))
    logger.info("Data Quality Report saved → {}", output_path)


# ── Report 3 — Target/Prediction Drift ───────────────────────────────────────
def run_target_drift_report(reference: pd.DataFrame,
                            current: pd.DataFrame,
                            predictions_df: pd.DataFrame | None) -> None:

    logger.info("Running Target Drift Report...")

    if predictions_df is None or len(predictions_df) == 0:
        logger.warning("No predictions in DB — skipping target drift")
        return

    n_current = len(predictions_df)

    # Generate reference predictions
    model = joblib.load("models/lightgbm_model.joblib")

    # Subsample reference to match current size
    # This makes Wasserstein comparison fair
    reference_sample = reference.sample(
        n=min(n_current * 5, len(reference)),  # 5x current size max
        random_state=42
    )
    reference_preds = model.predict(reference_sample)

    reference_with_pred = reference_sample.copy()
    reference_with_pred["predicted_price"] = reference_preds

    # Current
    current_with_pred = current.copy().iloc[:n_current]
    current_with_pred["predicted_price"] = \
        predictions_df["predicted_price"].values[:n_current]

    column_mapping = ColumnMapping(prediction="predicted_price")

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference_with_pred,
        current_data=current_with_pred,
        column_mapping=column_mapping,
    )

    output_path = REPORTS_DIR / "target_drift.html"
    report.save_html(str(output_path))
    logger.info("Target Drift Report saved → {}", output_path)
# ── TestSuite — Automated Pass/Fail ──────────────────────────────────────────

def run_test_suite(reference: pd.DataFrame,
                   current: pd.DataFrame) -> bool:
    logger.info("Running Test Suite...")

    suite = TestSuite(tests=[
        DataDriftTestPreset(),
        DataQualityTestPreset(),
    ])

    suite.run(reference_data=reference, current_data=current)

    output_path = REPORTS_DIR / "test_suite.html"
    suite.save_html(str(output_path))

    results    = suite.as_dict()
    summary    = results["summary"]
    all_passed = summary["all_passed"]
    total      = summary["total_tests"]
    passed     = summary["success_tests"]
    failed     = summary["failed_tests"]

    logger.info("Test Suite — {}/{} tests passed", passed, total)

    if not all_passed:
        logger.warning(
            "DRIFT ALERT — {} test(s) failed out of {} — consider retraining",
            failed, total
        )
    else:
        logger.info("All tests passed — no significant drift detected")

    logger.info("Test Suite Report saved → {}", output_path)
    return all_passed
# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evidently AI Monitoring")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Inject artificial drift to test detection"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EVIDENTLY MONITORING REPORT")
    logger.info("=" * 60)

    # Load data
    reference    = load_reference()
    current      = load_current(simulate=args.simulate)
    predictions  = load_predictions_from_db()

    # Run all reports
    run_data_drift_report(reference, current)
    run_data_quality_report(reference, current)
    run_target_drift_report(reference, current, predictions)
    all_passed = run_test_suite(reference, current)

    # Final verdict
    logger.info("=" * 60)
    if all_passed:
        logger.info("MONITORING COMPLETE — system healthy")
    else:
        logger.warning("MONITORING COMPLETE — drift detected, action required")
    logger.info("Reports saved to: {}", REPORTS_DIR)
    logger.info("Open in browser:")
    for report_file in REPORTS_DIR.glob("*.html"):
        logger.info("  {}", report_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
