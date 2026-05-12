# tests/test_preprocessor.py

import pandas as pd
import pytest
from flight_predictor.preprocessor import Preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "airline": ["Air India", "IndiGo"],
        "from": ["Delhi", "Mumbai"],
        "to": ["Mumbai", "Delhi"],
        "date": ["2022-02-11", "2022-03-01"],
        "dep_time": ["08:30", "14:00"],
        "arr_time": ["11:00", "15:45"],
        "time_taken": ["2h 30m", "1h 45m"],
        "stop": ["non-stop", "1-stop"],
        "price": ["12,500", "3,000"],
        "ch_code": ["AI", "6E"],
        "num_code": [101, 202],
        "is_business": [1, 0],
    })


def test_price_converted_to_float(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["price"].dtype == float


def test_price_comma_removed(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["price"].iloc[0] == 12500.0


def test_stops_encoded_as_integer(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["stops_numeric"].dtype == int


def test_nonstop_encoded_as_zero(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["stops_numeric"].iloc[0] == 0


def test_one_stop_encoded_as_one(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["stops_numeric"].iloc[1] == 1


def test_duration_minutes_correct(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["duration_minutes"].iloc[0] == 150  # 2h 30m = 150
    assert result["duration_minutes"].iloc[1] == 105  # 1h 45m = 105


def test_departure_hour_extracted(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["departure_hour"].iloc[0] == 8
    assert result["departure_hour"].iloc[1] == 14


def test_arrival_hour_extracted(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert result["arrival_hour"].iloc[0] == 11
    assert result["arrival_hour"].iloc[1] == 15


def test_dirty_columns_dropped(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    for col in ["ch_code", "num_code", "time_taken", "dep_time", "arr_time", "stop"]:
        assert col not in result.columns


def test_date_column_preserved(sample_df):
    result = Preprocessor().fit_transform(sample_df)
    assert "date" in result.columns


def test_original_dataframe_not_modified(sample_df):
    original_price = sample_df["price"].copy()
    Preprocessor().fit_transform(sample_df)
    assert sample_df["price"].equals(original_price)
