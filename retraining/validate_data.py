import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = [
    "airline", "from", "to", "price", "stop",
    "dep_time", "arr_time", "time_taken", "date", "ch_code", "num_code"
]

VALID_AIRLINES = [
    "IndiGo", "Air India", "Vistara", "GO FIRST",
    "AirAsia", "SpiceJet", "StarAir", "Trujet"
]

VALID_CITIES = [
    "Delhi", "Mumbai", "Bangalore",
    "Chennai", "Kolkata", "Hyderabad"
]


def validate_new_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validates a new data DataFrame before retraining.
    Returns (is_valid, list_of_errors).
    Empty error list means data passed all checks.
    """
    errors = []

    logger.info("Validating new data — {} rows, {} columns",
                len(df), len(df.columns))

    # ── 1. Schema check ───────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        logger.error("Schema check FAILED — missing: {}", missing_cols)
    else:
        logger.info("Schema check PASSED — all required columns present")

    # ── 2. Minimum size check ─────────────────────────────────────────────────
    if len(df) < 1000:
        errors.append(
            f"Too few rows: {len(df)} (minimum 1,000 required for retraining)"
        )
        logger.error("Size check FAILED — only {} rows", len(df))
    else:
        logger.info("Size check PASSED — {} rows", len(df))

    # ── 3. Null check ─────────────────────────────────────────────────────────
    if "price" in df.columns:
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(f"Null values found: {cols_with_nulls.to_dict()}")
            logger.error("Null check FAILED — {}", cols_with_nulls.to_dict())
        else:
            logger.info("Null check PASSED — zero nulls")

    # ── 4. Price range check ──────────────────────────────────────────────────
    if "price" in df.columns:
        # Clean price first (remove commas if string)
        try:
            prices = df["price"].astype(str).str.replace(",", "").astype(float)
            price_min = prices.min()
            price_max = prices.max()

            if price_min < 500:
                errors.append(
                    f"Suspicious minimum price: ₹{price_min:.0f} (expected > ₹500)"
                )
                logger.error("Price check FAILED — min price: ₹{:.0f}", price_min)
            if price_max > 250000:
                errors.append(
                    f"Suspicious maximum price: ₹{price_max:.0f} (expected < ₹250,000)"
                )
                logger.error("Price check FAILED — max price: ₹{:.0f}", price_max)
            if price_min >= 500 and price_max <= 250000:
                logger.info(
                    "Price check PASSED — range ₹{:.0f} to ₹{:.0f}",
                    price_min, price_max
                )
        except Exception as e:
            errors.append(f"Price column could not be parsed: {str(e)}")
            logger.error("Price parsing FAILED — {}", str(e))

    # ── 5. Duplicate check ────────────────────────────────────────────────────
    n_duplicates = df.duplicated().sum()
    dup_pct = n_duplicates / len(df) * 100
    if dup_pct > 10:
        errors.append(
            f"Too many duplicates: {n_duplicates} rows ({dup_pct:.1f}%)"
        )
        logger.error("Duplicate check FAILED — {:.1f}% duplicates", dup_pct)
    else:
        logger.info(
            "Duplicate check PASSED — {:.1f}% duplicates", dup_pct
        )

    # ── 6. Airline values check ───────────────────────────────────────────────
    if "airline" in df.columns:
        unknown_airlines = set(df["airline"].unique()) - set(VALID_AIRLINES)
        if unknown_airlines:
            logger.warning(
                "Unknown airlines found: {} — will use global mean for encoding",
                unknown_airlines
            )
            # Warning only — not an error (new airlines can be added)

    # ── Final result ──────────────────────────────────────────────────────────
    is_valid = len(errors) == 0

    if is_valid:
        logger.info("Validation PASSED — data ready for retraining")
    else:
        logger.error(
            "Validation FAILED — {} error(s) found:", len(errors)
        )
        for err in errors:
            logger.error("  → {}", err)

    return is_valid, errors
