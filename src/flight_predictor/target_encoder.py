import pandas as pd


class TargetEncoder:
    CATEGORICAL_COLS = ["airline", "from", "to"]

    def __init__(self):
        self._encoding_means = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        df = df.copy()
        df["__target__"] = y.values

        for col in self.CATEGORICAL_COLS:
            # Group by both the categorical column AND is_business
            # This gives separate means for business and economy per category
            self._encoding_means[col] = df.groupby([col, "is_business"])[
                "__target__"
            ].mean()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.CATEGORICAL_COLS:
            # Look up mean using both col value AND is_business as combined key
            df[col + "_encoded"] = df.apply(
                lambda row: self._encoding_means[col].get(
                    (row[col], row["is_business"]), None
                ),
                axis=1,
            )

        df = df.drop(columns=self.CATEGORICAL_COLS)
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
