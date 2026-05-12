from pydantic import BaseModel, Field
from typing import Literal

class FlightInput(BaseModel):
    stops_numeric: int = Field(..., ge=0, le=2, description="Number of stops: 0, 1, or 2")
    duration_minutes: int = Field(..., gt=0, description="Total flight duration in minutes")
    departure_hour: int = Field(..., ge=0, le=23, description="Departure hour (0-23)")
    arrival_hour: int = Field(..., ge=0, le=23, description="Arrival hour (0-23)")
    month: int = Field(..., ge=1, le=12, description="Month of flight (1-12)")
    day: int = Field(..., ge=0, le=6, description="Day of week: 0=Monday, 6=Sunday")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Saturday or Sunday, else 0")
    is_business: int = Field(..., ge=0, le=1, description="1 if business class, else 0")
    airline: Literal["Air India", "GO FIRST", "IndiGo", "SpiceJet", "StarAir", "Trujet", "Vistara"]
    from_city: Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
    to_city: Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "stops_numeric": 2,
                    "duration_minutes": 1265,
                    "departure_hour": 6,
                    "arrival_hour": 17,
                    "month": 2,
                    "day": 0,
                    "is_weekend": 0,
                    "is_business": 1,
                    "airline": "Vistara",
                    "from_city": "Kolkata",
                    "to_city": "Chennai"
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    predicted_price: float
    currency: str = "INR"
    model_version: str = "tunded_lgb"


class BatchFlightInput(BaseModel):
    flights: list[FlightInput] = Field(..., min_length=1, max_length=100)


class BatchPredictionOutput(BaseModel):
    predictions: list[float]
    count: int
    currency: str = "INR"
    model_version: str = "tunded_lgb"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool