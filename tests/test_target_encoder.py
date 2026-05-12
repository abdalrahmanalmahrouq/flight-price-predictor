import pandas as pd
import numpy as np
import pytest
from flight_predictor.target_encoder import TargetEncoder


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "airline": ["Air India", "Air India", "IndiGo", "IndiGo", "Vistara", "Vistara"],
        "from":    ["Delhi",     "Mumbai",    "Delhi",  "Mumbai",  "Delhi",   "Mumbai"],
        "to":      ["Mumbai",    "Delhi",     "Mumbai", "Delhi",   "Mumbai",  "Delhi"],
        "is_business": [1, 0, 0, 0, 1, 0],
        "other_col": [1, 2, 3, 4, 5, 6],
    })
    y = pd.Series([50000, 8000, 5000, 6000, 48000, 7000])
    return df, y


def test_fit_creates_encoding_means(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    encoder.fit(df, y)

    assert "airline" in encoder._encoding_means
    assert "from" in encoder._encoding_means
    assert "to" in encoder._encoding_means


def test_fit_creates_multiindex_per_class(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    encoder.fit(df, y)

    # Should have (airline, is_business) as MultiIndex
    assert ("Air India", 1) in encoder._encoding_means["airline"].index
    assert ("Air India", 0) in encoder._encoding_means["airline"].index
    assert ("IndiGo", 0) in encoder._encoding_means["airline"].index


def test_business_and_economy_get_different_encoded_values(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    encoder.fit(df, y)

    air_india_business = encoder._encoding_means["airline"][("Air India", 1)]
    air_india_economy  = encoder._encoding_means["airline"][("Air India", 0)]

    assert air_india_business != air_india_economy
    assert air_india_business > air_india_economy


def test_transform_drops_original_columns(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    result = encoder.fit_transform(df, y)

    assert "airline" not in result.columns
    assert "from" not in result.columns
    assert "to" not in result.columns


def test_transform_adds_encoded_columns(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    result = encoder.fit_transform(df, y)

    assert "airline_encoded" in result.columns
    assert "from_encoded" in result.columns
    assert "to_encoded" in result.columns


def test_transform_business_rows_get_higher_encoding(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    result = encoder.fit_transform(df, y)

    business_encoded = result[result["is_business"] == 1]["airline_encoded"].mean()
    economy_encoded  = result[result["is_business"] == 0]["airline_encoded"].mean()

    assert business_encoded > economy_encoded


def test_transform_does_not_mutate_input(sample_data):
    df, y = sample_data
    original_cols = df.columns.tolist()
    encoder = TargetEncoder()
    encoder.fit_transform(df, y)

    assert df.columns.tolist() == original_cols


def test_unseen_category_returns_none(sample_data):
    df, y = sample_data
    encoder = TargetEncoder()
    encoder.fit(df, y)

    df_new = pd.DataFrame({
        "airline": ["SpiceJet"],   # unseen airline
        "from":    ["Delhi"],
        "to":      ["Mumbai"],
        "is_business": [0],
        "other_col": [1],
    })
    result = encoder.transform(df_new)
    assert result["airline_encoded"].isna().all()


def test_fit_does_not_mutate_input(sample_data):
    df, y = sample_data
    original_cols = df.columns.tolist()
    encoder = TargetEncoder()
    encoder.fit(df, y)

    assert df.columns.tolist() == original_cols