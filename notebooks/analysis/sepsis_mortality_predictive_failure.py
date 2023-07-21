# %%
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from caumim.constants import COLNAME_CODE, COLNAME_EMERGENCY_ADMISSION, COLNAME_INCLUSION_START, COLNAME_INSURANCE_MEDICARE, COLNAME_MORTALITY_28D, COLNAME_PATIENT_ID, COLNAME_START, COLNAME_VALUE, DIR2EXPERIENCES, FILENAME_TARGET_POPULATION, STAY_KEYS
from copy import deepcopy
from caumim.experiments.configurations import ESTIMATOR_HGB
from caumim.experiments.utils import score_binary_classification
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS

from caumim.framing.utils import create_cohort_folder
from caumim.inference.utils import make_random_search_pipeline
from caumim.utils import to_polars
from caumim.variables.selection import get_event_covariates_albumin_zhou
from caumim.variables.utils import feature_emergency_at_admission, feature_insurance_medicare

%load_ext autoreload
%autoreload 2

# %%
config = deepcopy(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
observation_period_day = 1
observation_period_hour = observation_period_day * 24
config.min_icu_survival_unit_day = observation_period_day
config.min_los_icu_unit_day = observation_period_day
config.treatment_observation_window_unit_day = observation_period_day
albumin_cohort_folder = create_cohort_folder(config)
target_trial_population = pd.read_parquet(
    albumin_cohort_folder / FILENAME_TARGET_POPULATION
)
# Create static features
# demographics
target_trial_population = feature_emergency_at_admission(
    target_trial_population
)
target_trial_population = (
    feature_insurance_medicare(target_trial_population)
    .with_columns(
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
    .to_pandas()
)

# %%
# rename crystalloid date to trace pretreatment variables
outcome_name = COLNAME_MORTALITY_28D
COLNAME_CRYSTALLOID_START = "crystalloid_start"
target_trial_population[COLNAME_CRYSTALLOID_START] = target_trial_population[COLNAME_INCLUSION_START]
# split train (pre/post treatment fetaures) and test (pretreatment features only)
train, test = train_test_split(target_trial_population[COLNAME_PATIENT_ID], test_size=0.2, random_state=42, stratify=target_trial_population[outcome_name])
train_population = target_trial_population[target_trial_population[COLNAME_PATIENT_ID].isin(train)]
test_population = target_trial_population[target_trial_population[COLNAME_PATIENT_ID].isin(test)]

# redefinition of inclusion start to get back features after treatment for train population
train_population[
        COLNAME_INCLUSION_START
    ] = train_population["intime"] + pd.Timedelta(observation_period_hour, unit="h")
train_event_features, train_feature_types = get_event_covariates_albumin_zhou(
    train_population
)
test_event_features, test_feature_types = get_event_covariates_albumin_zhou(
    test_population
)
# %%
train_patient_features_aggregated = (
    train_event_features.sort([COLNAME_PATIENT_ID, COLNAME_START])
    .groupby(STAY_KEYS + [COLNAME_CODE])
    .agg(pl.col(COLNAME_VALUE).last().alias("last"))
).pivot(
    index=STAY_KEYS,
    columns=COLNAME_CODE,
    values="last",
    aggregate_function=None,
)
test_patient_features_aggregated = (
    test_event_features.sort([COLNAME_PATIENT_ID, COLNAME_START])
    .groupby(STAY_KEYS + [COLNAME_CODE])
    .agg(pl.col(COLNAME_VALUE).last().alias("last"))
).pivot(
    index=STAY_KEYS,
    columns=COLNAME_CODE,
    values="last",
    aggregate_function=None,
)
# %%
event_features_names = list(
            set(train_patient_features_aggregated.columns).difference(set(STAY_KEYS))
        )
static_features = [
        "admission_age",
        "Female",
        COLNAME_EMERGENCY_ADMISSION,
        COLNAME_INSURANCE_MEDICARE,
    ]
# Join static and dynamic features
train_X_Y = train_patient_features_aggregated.join(
    to_polars(train_population),
    on=STAY_KEYS,
    how="inner",
)[
    [
        *event_features_names,
        *static_features,
        #COLNAME_INTERVENTION_STATUS,
        outcome_name,
    ]
].to_pandas()
test_X_Y = test_patient_features_aggregated.join(
    to_polars(test_population),
    on=STAY_KEYS,
    how="inner",
)[
    [
        *event_features_names,
        *static_features,
        #COLNAME_INTERVENTION_STATUS,
        outcome_name,
    ]

].to_pandas()
val_size = test_X_Y.shape[0]
train_X, val_X, train_Y, val_Y = train_test_split(
    train_X_Y.drop(outcome_name, axis=1),
    train_X_Y[outcome_name],
    test_size=val_size,
    random_state=42,
    stratify=train_X_Y[outcome_name],
)

test_X = test_X_Y.drop(outcome_name, axis=1)
test_Y = test_X_Y[outcome_name]

# %%
# training
# column_transformer = make_column_transformer(
#             numerical_features=colnames_numerical_features,
#             categorical_features=colnames_categorical_features,
#         )
# using a classifier for the binary outcome (don't want to estimate a proba here)

outcome_pipeline = make_random_search_pipeline(
    estimator=ESTIMATOR_HGB["treatment_estimator"],
    param_distributions=ESTIMATOR_HGB["treatment_param_distributions"],
)
outcome_pipeline.fit(
    train_X, train_Y)
outcome_best_pipeline = outcome_pipeline.best_estimator_
# %%
# evaluation
# in-domain evaluation
hat_y_val = outcome_best_pipeline.predict_proba(val_X)[:, 1]
val_scores = score_binary_classification(val_Y, hat_y_val)
val_scores["dataset"] = "validation"
# out-of-domain evaluation
hat_y_test = outcome_best_pipeline.predict_proba(test_X)[:, 1]
test_scores = score_binary_classification(test_Y, hat_y_test)
test_scores["dataset"] = "test"
all_scores = pd.DataFrame([val_scores, test_scores]).set_index("dataset")

display(all_scores)
#%%