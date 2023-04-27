import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


# pipelines
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
