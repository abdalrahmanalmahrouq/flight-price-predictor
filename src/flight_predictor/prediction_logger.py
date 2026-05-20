import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger


DB_PATH = Path("logs/predictions.db")


def init_db() -> None:
    """Create predictions table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            is_business       INTEGER,
            stops_numeric     INTEGER,
            duration_minutes  INTEGER,
            departure_hour    INTEGER,
            arrival_hour      INTEGER,
            month             INTEGER,
            day               INTEGER,
            is_weekend        INTEGER,
            airline_encoded   REAL,
            from_encoded      REAL,
            to_encoded        REAL,
            predicted_price   REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Prediction database initialized at {}", DB_PATH)


def log_prediction(features: dict, predicted_price: float) -> None:
    """Log one prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, is_business, stops_numeric, duration_minutes,
            departure_hour, arrival_hour, month, day, is_weekend,
            airline_encoded, from_encoded, to_encoded, predicted_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        features["is_business"],
        features["stops_numeric"],
        features["duration_minutes"],
        features["departure_hour"],
        features["arrival_hour"],
        features["month"],
        features["day"],
        features["is_weekend"],
        features["airline_encoded"],
        features["from_encoded"],
        features["to_encoded"],
        predicted_price,
    ))
    conn.commit()
    conn.close()
