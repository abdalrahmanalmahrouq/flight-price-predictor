"""
Drift Detection Script — Phase 13
Run this periodically to check for data and concept drift.

Usage:
    python drift_detection.py              # uses test set as "production"
    python drift_detection.py --simulate   # injects artificial drift first
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger

from flight_predictor.logger import setup_logger

setup_logger(level="INFO")

# ── Config ────────────────────────────────────────────────────────────────────

REFERENCE_PATH  = "data/processed/X_train.csv"
PRODUCTION_PATH = "data/processed/X_test.csv"   # pretend this is production
DB_PATH         = "logs/predictions.db"

# Thresholds
KS_P_VALUE_THRESHOLD = 0.05   # below this → drift detected
PSI_THRESHOLD        = 0.20   # above this → drift detected
MAE_THRESHOLD        = 2000   # above training MAE of 1501 by 33%

CONTINUOUS_FEATURES  = [
    "duration_minutes", "departure_hour", "arrival_hour",
    "airline_encoded", "from_encoded", "to_encoded"
]
CATEGORICAL_FEATURES = ["is_business", "stops_numeric", "month", "is_weekend"]


# ── PSI Calculation ───────────────────────────────────────────────────────────

def calculate_psi(reference: pd.Series, current: pd.Series,
                  bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → no drift
    PSI 0.1-0.2 → moderate drift
    PSI > 0.2  → significant drift
    """
    # Get unique values for categorical features
    unique_vals = sorted(set(reference.unique()) | set(current.unique()))

    if len(unique_vals) <= bins:
        # Categorical — use value counts directly
        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)

        psi = 0.0
        for val in unique_vals:
            ref_pct = ref_counts.get(val, 1e-6)   # avoid log(0)
            cur_pct = cur_counts.get(val, 1e-6)
            psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        return psi
    else:
        # Continuous — bin first
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_binned = pd.cut(reference, bins=breakpoints, include_lowest=True)
        cur_binned = pd.cut(current,   bins=breakpoints, include_lowest=True)

        ref_pct = ref_binned.value_counts(normalize=True).sort_index()
        cur_pct = cur_binned.value_counts(normalize=True).sort_index()

        psi = 0.0
        for bin_label in ref_pct.index:
            r = ref_pct.get(bin_label, 1e-6)
            c = cur_pct.get(bin_label, 1e-6)
            psi += (c - r) * np.log(c / r)
        return abs(psi)


# ── Data Drift Detection ──────────────────────────────────────────────────────

def detect_data_drift(reference: pd.DataFrame,
                      current: pd.DataFrame) -> pd.DataFrame:
    """
    Run KS test on continuous features.
    Run PSI on categorical features.
    Return a summary DataFrame.
    """
    results = []

    # KS test — continuous features
    for feature in CONTINUOUS_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        stat, p_value = ks_2samp(reference[feature].dropna(),
                                  current[feature].dropna())
        drift = p_value < KS_P_VALUE_THRESHOLD

        results.append({
            "feature":        feature,
            "test":           "KS",
            "statistic":      round(stat, 4),
            "p_value":        round(p_value, 4),
            "threshold":      KS_P_VALUE_THRESHOLD,
            "drift_detected": drift,
        })

        if drift:
            logger.warning("DATA DRIFT detected in {} — KS={:.3f}, p={:.4f}",
                           feature, stat, p_value)
        else:
            logger.info("No drift in {} — KS={:.3f}, p={:.4f}",
                        feature, stat, p_value)

    # PSI — categorical features
    for feature in CATEGORICAL_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        psi = calculate_psi(reference[feature], current[feature])
        drift = psi > PSI_THRESHOLD

        results.append({
            "feature":        feature,
            "test":           "PSI",
            "statistic":      round(psi, 4),
            "p_value":        None,
            "threshold":      PSI_THRESHOLD,
            "drift_detected": drift,
        })

        if drift:
            logger.warning("DATA DRIFT detected in {} — PSI={:.4f}", feature, psi)
        else:
            logger.info("No drift in {} — PSI={:.4f}", feature, psi)

    return pd.DataFrame(results)


# ── Concept Drift Detection ───────────────────────────────────────────────────

def detect_concept_drift_from_db() -> dict | None:
    """
    Load predictions from SQLite and check prediction distribution.
    Without ground truth we watch the prediction distribution as a proxy.
    """
    if not Path(DB_PATH).exists():
        logger.warning("No predictions database found at {} — "
                       "start the API and make some requests first.", DB_PATH)
        return None

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        logger.warning("Predictions table is empty — no concept drift data.")
        return None

    logger.info("Loaded {} predictions from database", len(df))

    # Prediction distribution stats
    pred_mean   = df["predicted_price"].mean()
    pred_std    = df["predicted_price"].std()
    pred_median = df["predicted_price"].median()
    pred_p95    = df["predicted_price"].quantile(0.95)

    # Training prediction distribution (approximate from test labels)
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    train_mean = y_test.mean()

    # Proxy signal — how much has mean prediction shifted?
    mean_shift_pct = abs(pred_mean - train_mean) / train_mean * 100

    concept_drift = mean_shift_pct > 20   # 20% shift → investigate

    result = {
        "n_predictions":      len(df),
        "pred_mean":          round(pred_mean, 2),
        "pred_std":           round(pred_std, 2),
        "pred_median":        round(pred_median, 2),
        "pred_p95":           round(pred_p95, 2),
        "train_mean":         round(train_mean, 2),
        "mean_shift_pct":     round(mean_shift_pct, 2),
        "concept_drift":      concept_drift,
    }

    if concept_drift:
        logger.warning("CONCEPT DRIFT proxy signal — prediction mean shifted "
                       "{:.1f}% from training distribution", mean_shift_pct)
    else:
        logger.info("No concept drift proxy signal — "
                    "mean shift={:.1f}%", mean_shift_pct)

    return result


# ── Drift Simulation ──────────────────────────────────────────────────────────

def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Artificially inject drift into a DataFrame to test detection.
    This verifies your thresholds actually fire before real drift arrives.
    """
    drifted = df.copy()
    logger.info("Injecting artificial drift for testing...")

    # 1. Shift duration_minutes distribution significantly
    drifted["duration_minutes"] = (drifted["duration_minutes"] * 1.6).astype(int)
    logger.info("  → duration_minutes shifted up 60%")

    # 2. Flip is_business ratio (30% → 70% business)
    n_flip = int(len(drifted) * 0.4)
    economy_idx = drifted[drifted["is_business"] == 0].sample(n_flip).index
    drifted.loc[economy_idx, "is_business"] = 1
    logger.info("  → is_business flipped from 30% to ~70% business")

    # 3. Shift departure hours — more morning flights
    drifted["departure_hour"] = (drifted["departure_hour"] * 0.5).astype(int)
    logger.info("  → departure_hour shifted toward morning")

    return drifted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flight Price Drift Detection")
    parser.add_argument("--simulate", action="store_true",
                        help="Inject artificial drift to test detection")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DRIFT DETECTION REPORT")
    logger.info("=" * 60)

    # Load reference (training) data
    logger.info("Loading reference data from {}", REFERENCE_PATH)
    reference = pd.read_csv(REFERENCE_PATH)

    # Load current (production) data
    logger.info("Loading current data from {}", PRODUCTION_PATH)
    current = pd.read_csv(PRODUCTION_PATH)

    # Optionally inject drift
    if args.simulate:
        logger.info("SIMULATION MODE — injecting artificial drift")
        current = simulate_drift(current)

    logger.info("Reference size: {} rows", len(reference))
    logger.info("Current size:   {} rows", len(current))
    logger.info("-" * 60)

    # ── Data Drift ──
    logger.info("Running DATA DRIFT detection...")
    drift_results = detect_data_drift(reference, current)

    logger.info("-" * 60)
    logger.info("DATA DRIFT SUMMARY:")
    print("\n" + drift_results.to_string(index=False))

    n_drifted = drift_results["drift_detected"].sum()
    n_total   = len(drift_results)
    logger.info("\n{}/{} features show drift", n_drifted, n_total)

    # ── Concept Drift ──
    logger.info("-" * 60)
    logger.info("Running CONCEPT DRIFT proxy detection...")
    concept_result = detect_concept_drift_from_db()

    if concept_result:
        logger.info("CONCEPT DRIFT SUMMARY:")
        for key, val in concept_result.items():
            logger.info("  {}: {}", key, val)

    # ── Final Verdict ──
    logger.info("=" * 60)
    if n_drifted > 0:
        logger.warning("ACTION REQUIRED: {} features show data drift", n_drifted)
        logger.warning("Consider retraining with fresh data")
    else:
        logger.info("No significant data drift detected")

    if concept_result and concept_result["concept_drift"]:
        logger.warning("ACTION REQUIRED: Concept drift proxy signal triggered")
        logger.warning("Collect ground truth labels and verify MAE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
