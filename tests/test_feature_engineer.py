import pandas as pd
import pytest

from flight_predictor.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "airline": ["Air India", "IndiGo", "Vistara"],
        "from": ["Delhi", "Mumbai", "Kolkata"],
        "to": ["Mumbai", "Delhi", "Chennai"],
        "date": ["11-02-2022", "05-03-2022", "12-02-2022"],
        "departure_hour": [8, 14, 6],
        "arrival_hour": [11, 15, 9],
        "stops_numeric": [0, 1, 2],
        "duration_minutes": [150, 105, 180],
        "is_business": [1, 0, 1],
        "price": [12500.0, 3000.0, 15000.0],
    })


def test_month_extracted(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert result["month"].iloc[0] == 2   # February
    assert result["month"].iloc[1] == 3   # March


def test_day_extracted(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert result["day"].iloc[0] == 4     # 2022-02-11 is a Friday
    assert result["day"].iloc[1] == 5     # 2022-03-05 is a Saturday


def test_is_weekend_correct(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert result["is_weekend"].iloc[0] == 0   # Friday → not weekend
    assert result["is_weekend"].iloc[1] == 1   # Saturday → weekend


def test_date_column_dropped(sample_df):
    result = FeatureEngineer().fit_transform(sample_df)
    assert "date" not in result.columns



def test_original_dataframe_not_modified(sample_df):
    original_date = sample_df["date"].copy()
    FeatureEngineer().fit_transform(sample_df)
    assert sample_df["date"].equals(original_date)
