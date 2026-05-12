# tests/test_target_encoder.py

import pandas as pd
import pytest
from flight_predictor.target_encoder import TargetEncoder


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "airline": ["Air India", "Air India", "IndiGo", "IndiGo", "Vistara"],
        "from":    ["Delhi",     "Mumbai",    "Delhi",  "Mumbai",  "Delhi"],
        "to":      ["Mumbai",    "Delhi",     "Chennai","Kolkata", "Mumbai"],
        "stops_numeric":    [0, 1, 0, 1, 0],
        "duration_minutes": [150, 105, 120, 90, 180],
        "departure_hour":   [8, 14, 6, 10, 16],
        "arrival_hour":     [11, 15, 9, 12, 19],
        "month":            [2, 3, 2, 3, 2],
        "day":              [4, 5, 4, 5, 4],
        "is_weekend":       [0, 1, 0, 1, 0],
        "is_business":      [1, 0, 0, 0, 1],
    })


@pytest.fixture
def train_y():
    return pd.Series([12500.0, 3000.0, 2800.0, 3200.0, 15000.0])


@pytest.fixture
def val_df():
    return pd.DataFrame({
        "airline": ["Air India", "IndiGo"],
        "from":    ["Delhi",     "Mumbai"],
        "to":      ["Mumbai",    "Kolkata"],
        "stops_numeric":    [0, 1],
        "duration_minutes": [150, 90],
        "departure_hour":   [8, 10],
        "arrival_hour":     [11, 12],
        "month":            [2, 3],
        "day":              [4, 5],
        "is_weekend":       [0, 1],
        "is_business":      [1, 0],
    })


def test_encoded_columns_created(train_df, train_y):
    result = TargetEncoder().fit(train_df, train_y).transform(train_df)
    assert "airline_encoded" in result.columns
    assert "from_encoded" in result.columns
    assert "to_encoded" in result.columns


def test_original_categorical_columns_dropped(train_df, train_y):
    result = TargetEncoder().fit(train_df, train_y).transform(train_df)
    assert "airline" not in result.columns
    assert "from" not in result.columns
    assert "to" not in result.columns


def test_airline_encoded_values_are_means(train_df, train_y):
    encoder = TargetEncoder()
    result = encoder.fit(train_df, train_y).transform(train_df)

    # Air India rows: prices 12500 and 3000 → mean = 7750
    air_india_rows = result.iloc[[0, 1]]
    assert (air_india_rows["airline_encoded"] == 7750.0).all()

    # IndiGo rows: prices 2800 and 3200 → mean = 3000
    indigo_rows = result.iloc[[2, 3]]
    assert (indigo_rows["airline_encoded"] == 3000.0).all()


def test_val_uses_train_means_not_its_own(train_df, train_y, val_df):
    encoder = TargetEncoder()
    encoder.fit(train_df, train_y)

    val_result = encoder.transform(val_df)

    # Air India mean learned from train = 7750
    # val has only one Air India row but it must use train mean not recalculate
    assert val_result["airline_encoded"].iloc[0] == 7750.0


def test_fit_transform_equals_fit_then_transform(train_df, train_y):
    encoder1 = TargetEncoder()
    result1 = encoder1.fit_transform(train_df, train_y)

    encoder2 = TargetEncoder()
    result2 = encoder2.fit(train_df, train_y).transform(train_df)

    assert result1.equals(result2)


def test_original_dataframe_not_modified(train_df, train_y):
    original_airline = train_df["airline"].copy()
    TargetEncoder().fit(train_df, train_y).transform(train_df)
    assert train_df["airline"].equals(original_airline)
