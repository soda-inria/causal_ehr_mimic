# %%
import polars as pl
import pandas as pd
import numpy as np
from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.inference.estimation import ESTIMATOR_RF, ESTIMATOR_LR

from caumim.variables.selection import get_albumin_events_zhou_baseline
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)

from dowhy import CausalModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge


# %%
# 1 - Framing
cohort_folder = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
target_trial_population = pl.read_parquet(
    cohort_folder / FILENAME_TARGET_POPULATION
)
target_trial_population.head()
# %%
# 2 - Variable selection

# Static features
# demographics
target_trial_population = feature_emergency_at_admission(
    target_trial_population
)
target_trial_population = feature_insurance_medicare(
    target_trial_population
).with_columns(
    [
        pl.when(pl.col("gender") == "F")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("Female"),
        pl.when(pl.col("race").str.to_lowercase().str.contains("white"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("White"),
    ]
)
static_features = [
    "admission_age",
    "Female",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]
outcome_name = COLNAME_MORTALITY_28D
# %%
# event features
event_features, feature_types = get_albumin_events_zhou_baseline(
    target_trial_population
)
# %%
patient_features_last = event_features.sort(
    [COLNAME_PATIENT_ID, COLNAME_START]
).pivot(
    index=STAY_KEYS,
    columns=COLNAME_CODE,
    values="value",
    aggregate_function=pl.element().median(),
)
event_features_names = list(
    set(patient_features_last.columns).difference(set(STAY_KEYS))
)

X = patient_features_last.join(
    target_trial_population,
    on=STAY_KEYS,
    how="inner",
)[
    [
        *event_features_names,
        *static_features,
        COLNAME_INTERVENTION_STATUS,
        outcome_name,
    ]
].to_pandas()

binary_features = [
    "Glycopeptide",  # J01XA
    "Beta-lactams",  # "J01C",
    "Carbapenems",  # "J01DH",
    "Aminoglycosides",  # "J01G",
    "suspected_infection_blood",
    "RRT",
    "ventilation",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]
categorical_features = ["aki_stage"]
numerical_features = list(
    X.columns.difference(
        set(
            [
                *binary_features,
                *categorical_features,
                outcome_name,
                COLNAME_INTERVENTION_STATUS,
            ]
        )
    )
)
X[binary_features] = X[binary_features].fillna(value=0)

# 3 - Identification
model = CausalModel(
    data=X,
    treatment=COLNAME_INTERVENTION_STATUS,
    outcome=outcome_name,
    common_causes=[*event_features_names, *static_features],
)
# model.view_model(size=(15, 15))
# from IPython.display import Image, display
# display(Image(filename=cohort_folder / "causal_model.png"))

identified_estimand = model.identify_effect(
    optimize_backdoor=True, proceed_when_unidentifiable=True
)
# print(identified_estimand)
# note: long to run on 22 variables if not forcing optimize_backdoor.

# 4 - Estimation

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = make_pipeline(
    *[
        SimpleImputer(strategy="mean"),
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

treatment_estimator = ESTIMATOR_LR

treatment_pipeline = RandomizedSearchCV(
    Pipeline(
        [
            ("preprocessor", column_transformer),
            ("treatment_estimator", treatment_estimator["treatment_estimator"]),
        ]
    ),
    param_distributions=treatment_estimator["treatment_estimator_kwargs"],
    random_state=42,
)

# treatment_model.fit(
#     X.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1),
#     X[COLNAME_INTERVENTION_STATUS],
# )
outcome_model = None
# treatment_pipeline = make_pipeline(*[column_transformer, treatment_estimator])
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    method_params={
        "propensity_score_model": treatment_pipeline,
        "min_ps_score": 0.001,
        "max_ps_score": 0.999,
        # "outcome_model": outcome_model,
    },
    confidence_intervals=False,
)
lower_bound, upper_bound = estimate.get_confidence_intervals()
results = {}
results[RESULT_ATE] = estimate.value
results[RESULT_ATE_LB] = lower_bound
results[RESULT_ATE_UB] = upper_bound
results
# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from econml.dml import LinearDML

outcome_model = make_pipeline(*[column_transformer, LogisticRegression()])
treatment_pipeline = make_pipeline(*[column_transformer, LogisticRegression()])

est = LinearDML(
    model_y=outcome_model,
    model_t=treatment_pipeline,
    featurizer=column_transformer,
    linear_first_stages=False,
    cv=5,
)
est.fit(
    X[outcome_name],
    X[COLNAME_INTERVENTION_STATUS],
    X=X.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1),
)
# %% TLearner does not work since it does not support the pipeline, so I can't
# use it properly (eg. with a CV)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

outcome_model = make_pipeline(*[column_transformer, LogisticRegression()])
# outcome_model = HistGradientBoostingClassifier()
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.metalearners.TLearner",
    method_params={
        "init_params": {"models": outcome_model},
        "fit_params": {},
    },
    confidence_intervals=False,
)
lower_bound, upper_bound = estimate.get_confidence_intervals()
results = {}
results[RESULT_ATE] = estimate.value
results[RESULT_ATE_LB] = lower_bound
results[RESULT_ATE_UB] = upper_bound
results
# %%
from econml.metalearners import TLearner

c_estimator = TLearner(
    models=outcome_model,
)
transformed_data = column_transformer.fit_transform(
    X.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1),
)
transformed_data = pd.DataFrame(
    transformed_data, columns=column_transformer.get_feature_names_out()
)
c_estimator.fit(
    X[outcome_name],
    X[COLNAME_INTERVENTION_STATUS],
    X=X.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1),
)
# %%
# Refute/Test hypothesis.
interpretation = causal_estimate_ipw.interpret(
    method_name="confounder_distribution_interpreter",
    fig_size=(8, 8),
    font_size=12,
    var_name="W4",
    var_type="discrete",
)
# %%
# Naive DM estimate:
from zepid import RiskDifference

dm = RiskDifference()
dm.fit(X, COLNAME_INTERVENTION_STATUS, outcome_name)
# dm.summary()
dm_results = {
    RESULT_ATE: dm.results.RiskDifference[1],
    RESULT_ATE_LB: dm.results.RiskDifference[1],
    RESULT_ATE_UB: dm.results.RiskDifference[1],
}
