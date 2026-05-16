from contextlib import asynccontextmanager
import os

import joblib
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from api.schemas import (
    BatchFlightInput,
    BatchPredictionOutput,
    FlightInput,
    HealthResponse,
    PredictionOutput,
)

from flight_predictor.logger import logger, setup_logger

LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO")
setup_logger(level=LOG_LEVEL)


# Model container
ml_model: dict = {}

FEATURE_ORDER = [
    "is_business",
    "stops_numeric",
    "duration_minutes",
    "departure_hour",
    "arrival_hour",
    "month",
    "day",
    "is_weekend",
    "airline_encoded",
    "from_encoded",
    "to_encoded",
]

ENCODING_MAP = {
    "airline": {
        ("Air India", 0): 7301.0,
        ("Air India", 1): 47154.0,
        ("AirAsia", 0): 4117.0,
        ("GO FIRST", 0): 5656.0,
        ("Indigo", 0): 5318.0,
        ("SpiceJet", 0): 6195.0,
        ("StarAir", 0): 4493.0,
        ("Trujet", 0): 3316.0,
        ("Vistara", 0): 7815.0,
        ("Vistara", 1): 55497.0,
    },
    "from": {
        ("Bangalore", 0): 6582.0,  ("Bangalore", 1): 53792.0,
        ("Chennai", 0): 6600.0,    ("Chennai", 1): 54206.0,
        ("Delhi", 0): 6299.0,      ("Delhi", 1): 48798.0,
        ("Hyderabad", 0): 6228.0,  ("Hyderabad", 1): 50331.0,
        ("Kolkata", 0): 7464.0,    ("Kolkata", 1): 56546.0,
        ("Mumbai", 0): 6350.0,     ("Mumbai", 1): 52748.0,
    },
    "to": {
        ("Bangalore", 0): 6593.0,  ("Bangalore", 1): 53801.0,
        ("Chennai", 0): 6650.0,    ("Chennai", 1): 53623.0,
        ("Delhi", 0): 6263.0,      ("Delhi", 1): 48081.0,
        ("Hyderabad", 0): 6310.0,  ("Hyderabad", 1): 50561.0,
        ("Kolkata", 0): 7185.0,    ("Kolkata", 1): 56823.0,
        ("Mumbai", 0): 6475.0,     ("Mumbai", 1): 52878.0,
    },
}

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"      
)
MODEL_NAME  = os.getenv("MODEL_NAME", "lightgbm_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Connecting to mlflow at {}...", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info("Loading model {}@{}", MODEL_NAME, MODEL_ALIAS)
    # for production
    ml_model["champion"] = mlflow.lightgbm.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    # for experiment
    for name in ["lightgbm", "xgboost", "catboost", "linear"]:
        ml_model[name] = joblib.load(f"models/{name}_model.joblib")
        logger.info("Loaded {} from disk.", name)

    logger.info("All models loaded successfully.")
    yield
    logger.info("Shutting down. Clearing models.")
    ml_model.clear()


# App
app = FastAPI(
    title="Flight Price Predictor",
    description="Predicts Indian domestic flight ticket prices using a tuned LightGBM model.",
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded="model" in ml_model,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(flight: FlightInput):
    logger.info("Prediction request — airline={} is_business={} month={}",
                flight.airline, flight.is_business, flight.month)
    if "champion" not in ml_model:
        logger.error("Champion model not loaded — returning 503")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        data = flight.model_dump()

        if data["is_business"] == 1 and data["airline"] not in ["Air India", "Vistara"]:
            logger.warning("Invalid business class airline: {}", data["airline"])
            raise HTTPException(
                status_code=422,
                detail=f"{data['airline']} does not operate business class. Choose Air India or Vistara."
            )
        # Build lookup key using (value, is_business) tuple
        is_business = data["is_business"]
        data["airline_encoded"] = ENCODING_MAP["airline"][(data.pop("airline"), is_business)]
        data["from_encoded"]    = ENCODING_MAP["from"][(data.pop("from_city"), is_business)]
        data["to_encoded"]      = ENCODING_MAP["to"][(data.pop("to_city"), is_business)]

        df = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = ml_model["champion"].predict(df)
        price = float(prediction[0])

        logger.info("Prediction complete — price={:.2f} INR", price)

    except KeyError as e:
        logger.warning("Unknown value in request: {}", str(e))
        raise HTTPException(status_code=422, detail=f"Unknown value: {str(e)}")
    except Exception as e:
        logger.error("Prediction failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionOutput(predicted_price=price)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchFlightInput):

    logger.info("Prediction request — airline={} is_business={} month={}",
                flight.airline, flight.is_business, flight.month)
    
    if "champion" not in ml_model:
        logger.error("Champion model not loaded — returning 503")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        rows = []
        for flight in batch.flights:
            data = flight.model_dump()

            if data["is_business"] == 1 and data["airline"] not in ["Air India", "Vistara"]:
                logger.warning("Invalid business class airline: {}", data["airline"])
                raise HTTPException(
                    status_code=422,
                    detail=f"{data['airline']} does not operate business class. Choose Air India or Vistara."
                )
            is_business = data["is_business"]
            data["airline_encoded"] = ENCODING_MAP["airline"][(data.pop("airline"), is_business)]
            data["from_encoded"]    = ENCODING_MAP["from"][(data.pop("from_city"), is_business)]
            data["to_encoded"]      = ENCODING_MAP["to"][(data.pop("to_city"), is_business)]
            rows.append(data)

        df = pd.DataFrame(rows, columns=FEATURE_ORDER)
        predictions = ml_model["champion"].predict(df)
        prices = [float(p) for p in predictions]
        logger.info("Prediction complete — price={:.2f} INR", price)

    except KeyError as e:
        logger.warning("Unknown value in request: {}", str(e))
        raise HTTPException(status_code=422, detail=f"Unknown value: {str(e)}")
    except Exception as e:
        logger.error("Prediction failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    return BatchPredictionOutput(
        predictions=prices,
        count=len(prices),
    )




@app.post("/experiment/predict", response_model=PredictionOutput)
def predict_experiment(
    flight: FlightInput,
    model_name: str = Query(default="lightgbm", enum=["lightgbm", "xgboost", "catboost","linear"])
):
    logger.info("Prediction request — airline={} is_business={} month={}",
            flight.airline, flight.is_business, flight.month)
    if model_name not in ml_model:
        logger.error("Champion model not loaded — returning 503")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        data = flight.model_dump()

        if data["is_business"] == 1 and data["airline"] not in ["Air India", "Vistara"]:
            logger.warning("Invalid business class airline: {}", data["airline"])
            raise HTTPException(
                status_code=422,
                detail=f"{data['airline']} does not operate business class. Choose Air India or Vistara."
            )
        # Build lookup key using (value, is_business) tuple
        is_business = data["is_business"]
        data["airline_encoded"] = ENCODING_MAP["airline"][(data.pop("airline"), is_business)]
        data["from_encoded"]    = ENCODING_MAP["from"][(data.pop("from_city"), is_business)]
        data["to_encoded"]      = ENCODING_MAP["to"][(data.pop("to_city"), is_business)]

        df = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = ml_model[model_name].predict(df)
        price = float(prediction[0])
        logger.info("Prediction complete — price={:.2f} INR", price)

    except KeyError as e:
        logger.warning("Unknown value in request: {}", str(e))
        raise HTTPException(status_code=422, detail=f"Unknown value: {str(e)}")
    except Exception as e:
        logger.error("Prediction failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionOutput(predicted_price=price, model_version = model_name)
