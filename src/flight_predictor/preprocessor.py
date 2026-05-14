import pandas as pd


class Preprocessor:
    COLUMNS_TO_DROP = [
        "ch_code",
        "num_code",
        "time_taken",
        "dep_time",
        "arr_time",
        "stop",
    ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._clean_price(df)
        df = self._clean_stops(df)
        df = self._clean_duration(df)
        df = self._clean_times(df)
        df = self._drop_columns(df)
        return df

    def _clean_price(self, df):
        df["price"] = df["price"].str.replace(",", "").astype(float)
        return df

    def _clean_stops(self, df):
        df["stop"] = df["stop"].str.replace(r"\s+", " ", regex=True).str.strip()

        def parse(val):
            if "non-stop" in val:
                return 0
            elif "2+" in val or "2-stop" in val:
                return 2
            elif "1-stop" in val:
                return 1
            else:
                return None

        df["stops_numeric"] = df["stop"].apply(parse)
        return df

    def _clean_duration(self, df):
        df["duration_minutes"] = df["time_taken"].str.extract(r"(\d+)h")[0].fillna(
            0
        ).astype(int) * 60 + df["time_taken"].str.extract(r"(\d+)m")[0].fillna(
            0
        ).astype(int)
        return df

    def _clean_times(self, df):
        df["departure_hour"] = pd.to_numeric(df["dep_time"].str[:2])
        df["arrival_hour"] = pd.to_numeric(df["arr_time"].str[:2])

        return df

    def _drop_columns(self, df):
        return df.drop(columns=self.COLUMNS_TO_DROP)
