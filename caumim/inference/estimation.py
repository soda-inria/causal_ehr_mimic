import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


# pipelines
ESTIMATOR_LR = {
    "treatment_estimator": LogisticRegression(n_jobs=-1),
    "treatment_estimator_kwargs": {
        "treatment_estimator__C": np.logspace(-4, 3, 1),
    },
}
ESTIMATOR_RF = {
    "treatment_estimator": RandomForestClassifier(),
    "treatment_estimator_kwargs": {
        "treatment_estimator__n_estimators": [10, 100, 500],
        "treatment_estimator__max_depth": [3, 10, 100],
    },
}


#
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
