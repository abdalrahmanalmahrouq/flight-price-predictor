# tests/test_data_loader.py


import pandas as pd
import pytest

from flight_predictor.data_loader import DataLoader


@pytest.fixture
def loader(tmp_path):
    # create fake business.csv
    business = pd.DataFrame({
        "airline": ["Air India", "Air India"],
        "from": ["Delhi", "Mumbai"],
        "to": ["Mumbai", "Delhi"],
        "date": ["2022-02-11", "2022-02-12"],
        "dep_time": ["08:00", "09:00"],
        "arr_time": ["10:00", "11:00"],
        "time_taken": ["2h 0m", "2h 0m"],
        "stop": ["non-stop", "non-stop"],
        "price": ["12,000", "13,000"],
        "ch_code": ["AI", "AI"],
        "num_code": [101, 102],
    })

    # create fake economy.csv with one duplicate row
    economy = pd.DataFrame({
        "airline": ["IndiGo", "IndiGo", "IndiGo"],
        "from": ["Delhi", "Mumbai", "Delhi"],
        "to": ["Mumbai", "Delhi", "Mumbai"],
        "date": ["2022-02-11", "2022-02-12", "2022-02-11"],
        "dep_time": ["08:00", "09:00", "08:00"],
        "arr_time": ["10:00", "11:00", "10:00"],
        "time_taken": ["2h 0m", "2h 0m", "2h 0m"],
        "stop": ["non-stop", "1-stop", "non-stop"],
        "price": ["3,000", "4,000", "3,000"],
        "ch_code": ["6E", "6E", "6E"],
        "num_code": [201, 202, 201],
    })

    # save to tmp_path — a temporary folder pytest creates and cleans up
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    business.to_csv(raw_dir / "business.csv", index=False)
    economy.to_csv(raw_dir / "economy.csv", index=False)

    return DataLoader(raw_dir=str(raw_dir))


def test_output_is_dataframe(loader):
    df = loader.load()
    assert isinstance(df, pd.DataFrame)


def test_is_business_column_exists(loader):
    df = loader.load()
    assert "is_business" in df.columns


def test_business_rows_labeled_correctly(loader):
    df = loader.load()
    assert df[df["airline"] == "Air India"]["is_business"].unique()[0] == 1


def test_economy_rows_labeled_correctly(loader):
    df = loader.load()
    assert df[df["airline"] == "IndiGo"]["is_business"].unique()[0] == 0


def test_duplicates_removed(loader):
    df = loader.load()
    assert df.duplicated().sum() == 0


def test_both_files_merged(loader):
    df = loader.load()
    # 2 business + 2 economy after duplicate removed = 4 rows
    assert len(df) == 4
