from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load(self) -> pd.DataFrame:
        business = pd.read_csv(self.raw_dir / "business.csv")
        economy = pd.read_csv(self.raw_dir / "economy.csv")

        business["is_business"] = 1
        economy["is_business"] = 0

        df = pd.concat([business, economy], ignore_index=True)
        df = df.drop_duplicates()

        return df
