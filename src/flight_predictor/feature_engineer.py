import pandas as pd


class FeatureEngineer:
    COLUMNS_TO_DROP = ["date"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._extract_date_features(df)
        df = self._drop_columns(df)
        return df

    def _extract_date_features(self, df):
        date = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["month"] = date.dt.month
        df["day"] = date.dt.day_of_week
        df["is_weekend"] = df["day"].isin([5, 6]).astype(int)
        return df

    def _drop_columns(self, df):
        return df.drop(columns=self.COLUMNS_TO_DROP)
