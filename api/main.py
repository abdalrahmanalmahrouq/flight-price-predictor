from contextlib import asynccontextmanager

import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchFlightInput,
    BatchPredictionOutput,
    FlightInput,
    HealthResponse,
    PredictionOutput,
)

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
        "Air India": 23611,
        "GO FIRST": 5644,
        "IndiGo": 5314,
        "SpiceJet": 6215,
        "StarAir": 6215,
        "Trujet": 3237,
        "Vistara": 30382,
    },
    "from": {
        "Bangalore": 21476,
        "Chennai": 22067,
        "Delhi": 18993,
        "Hyderabad": 20052,
        "Kolkata": 21785,
        "Mumbai": 21481,
    },
    "to": {
        "Bangalore": 21509,
        "Chennai": 21932,
        "Delhi": 18553,
        "Hyderabad": 20458,
        "Kolkata": 22003,
        "Mumbai": 21362,
    },
}

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model from MLflow registry...")
    ml_model["model"] = mlflow.lightgbm.load_model(
        "models:/lgb_model@production"
    )
    print("Model loaded successfully.")
    yield
    print("Shutting down. Clearing model.")
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
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        data = flight.model_dump()

        # Step 1 — encode first
        data["airline_encoded"] = ENCODING_MAP["airline"][data.pop("airline")]
        data["from_encoded"]    = ENCODING_MAP["from"][data.pop("from_city")]
        data["to_encoded"]      = ENCODING_MAP["to"][data.pop("to_city")]

        print("DEBUG after encoding:", data)
        # Step 2 — build DataFrame after encoding
        df = pd.DataFrame([data], columns=FEATURE_ORDER)

        # Step 3 — predict
        prediction = ml_model["model"].predict(df)
        price = float(prediction[0])

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Unknown value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionOutput(predicted_price=price)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchFlightInput):
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        rows = []
        for flight in batch.flights:
            data = flight.model_dump()
            data["airline_encoded"] = ENCODING_MAP["airline"][data.pop("airline")]
            data["from_encoded"]    = ENCODING_MAP["from"][data.pop("from_city")]
            data["to_encoded"]      = ENCODING_MAP["to"][data.pop("to_city")]
            rows.append(data)

        df = pd.DataFrame(rows, columns=FEATURE_ORDER)
        predictions = ml_model["model"].predict(df)
        prices = [float(p) for p in predictions]

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Unknown value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    return BatchPredictionOutput(
        predictions=prices,
        count=len(prices),
    )