import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config["data"]
        self.prep_config = config["preprocessing"]
        self.target_col = self.data_config["target_column"]
        self.pipeline = None
        self.label_encoder = None

    def fit_transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].values

        if self.config["model"]["type"] == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        test_size = self.data_config["test_size"]
        val_size = self.data_config["val_size"]
        seed = self.data_config["random_seed"]

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=seed)

        self.pipeline = self._build_pipeline(X_train)
        X_train_t = self.pipeline.fit_transform(X_train)
        X_val_t = self.pipeline.transform(X_val)
        X_test_t = self.pipeline.transform(X_test)

        return X_train_t, X_val_t, X_test_t, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame):
        return self.pipeline.transform(df)

    def _build_pipeline(self, X):
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        ScalerClass = SCALERS.get(self.prep_config.get("scaler", "standard"), StandardScaler)
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=self.prep_config.get("handle_missing", "median"))),
            ("scaler", ScalerClass()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipe, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="passthrough")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved: {path}")

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
        