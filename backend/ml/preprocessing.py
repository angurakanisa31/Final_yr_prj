import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []

    def fit(self, df: pd.DataFrame, numerical_cols: list[str], categorical_cols: list[str]):
        """Fit preprocessing pipeline on train dataframe."""
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        # 1. Clean duplicates
        df_clean = df.drop_duplicates()

        # 2. Fit Categorical Label Encoders
        for col in self.categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                # Fill missing with 'Unknown'
                filled_series = df_clean[col].fillna("Unknown").astype(str)
                le.fit(filled_series)
                self.label_encoders[col] = le

        # 3. Fit Numerical Scaler
        if self.numerical_cols:
            filled_num = df_clean[self.numerical_cols].fillna(0)
            self.scaler.fit(filled_num)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to transform a dataframe."""
        df_out = df.copy()

        # Fill missing values
        for col in self.categorical_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna("Unknown").astype(str)
                # Map unseen labels to the first class
                le = self.label_encoders[col]
                classes = set(le.classes_)
                df_out[col] = df_out[col].apply(lambda x: x if x in classes else le.classes_[0])
                df_out[col] = le.transform(df_out[col])

        if self.numerical_cols:
            df_out[self.numerical_cols] = df_out[self.numerical_cols].fillna(0)
            df_out[self.numerical_cols] = self.scaler.transform(df_out[self.numerical_cols])

        return df_out

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to drop duplicates and fill simple missing values in-place."""
        df_out = df.drop_duplicates()
        # Handle numerical missing
        num_cols = df_out.select_dtypes(include=[np.number]).columns
        df_out[num_cols] = df_out[num_cols].fillna(0)
        # Handle categorical missing
        cat_cols = df_out.select_dtypes(include=[object]).columns
        df_out[cat_cols] = df_out[cat_cols].fillna("Unknown")
        return df_out
