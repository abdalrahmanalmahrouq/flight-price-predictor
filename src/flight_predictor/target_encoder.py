import pandas as pd


class TargetEncoder:

    CATEGORICAL_COLS = ["airline", "from", "to"]

    def __init__(self):
        self._encoding_means = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        df = df.copy()
        df["__target__"] = y.values

        for col in self.CATEGORICAL_COLS:
            self._encoding_means[col] = df.groupby(col)["__target__"].mean()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.CATEGORICAL_COLS:
            df[col + "_encoded"] = df[col].map(self._encoding_means[col])

        df = df.drop(columns=self.CATEGORICAL_COLS)
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
