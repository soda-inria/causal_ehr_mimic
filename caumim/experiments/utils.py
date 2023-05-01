from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


def log_estimate(estimate: Dict, estimate_folder: str):
    estimate_folder_path = Path(estimate_folder)
    estimate_folder_path.mkdir(parents=True, exist_ok=True)

    estimate_ = {k: [v] for k, v in estimate.items()}
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    estimate_["time_stamp"] = [current_time]
    pd.DataFrame(estimate_).to_parquet(
        str(estimate_folder_path / f"{current_time}.parquet")
    )


def make_column_tranformer(
    numerical_features: list, categorical_features: list
) -> Pipeline:
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = make_pipeline(
        *[
            SimpleImputer(strategy="median"),
            StandardScaler(),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            (
                "one-hot-encoder",
                categorical_preprocessor,
                categorical_features,
            ),
            (
                "standard_scaler",
                numerical_preprocessor,
                numerical_features,
            ),
        ],
        remainder="passthrough",
        # The passthrough is necessary for all the event features.
    )
    return column_transformer
