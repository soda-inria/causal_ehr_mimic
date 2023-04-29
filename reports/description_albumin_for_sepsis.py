# %%
# %load_ext autoreload
# %autoreload 2
import polars as pl
import pandas as pd
from caumim.variables.selection import (
    get_albumin_events_zhou_baseline,
    get_antibiotics_event_from_atc4,
    get_antibiotics_event_from_drug_name,
)
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
)

from caumim.framing.albumin_for_sepsis import (
    COHORT_CONFIG_ALBUMIN_FOR_SEPSIS,
)
from caumim.framing.utils import create_cohort_folder
from caumim.constants import *
from caumim.description.utils import COMMON_DELTAS, add_delta, describe_delta
from caumim.utils import to_polars

# %%
tables = [file_.name for file_ in list(DIR2MIMIC.iterdir())]
tables.sort(reverse=True)
print(tables)

# %%
# Load target population
albumin_cohort_folder = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
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
# deltas
target_trial_population.head()
target_trial_population["delta"] = (
    target_trial_population["dod"]
    - target_trial_population[COLNAME_INCLUSION_START]
)
target_trial_population_w_deltas = add_delta(target_trial_population)
deltas = describe_delta(target_trial_population, unit="hours")
deltas["IR"] = deltas["75%"] - deltas["25%"]
print(deltas[["count", "50%", "IR"]])

# %%
# Variable selection
# Get baseline events
event_features = get_albumin_events_zhou_baseline(
    albumin_cohort_folder / FILENAME_TARGET_POPULATION
)
print(event_features.shape)
event_features["code"].value_counts().sort("counts", descending=True)
# %% Variable aggregation
patient_features_last = event_features.sort(
    [COLNAME_PATIENT_ID, COLNAME_START]
).pivot(
    index=STAY_KEYS,
    columns=COLNAME_CODE,
    values="value",
    aggregate_function=pl.element().last(),
)
# %%
# Join with static features:
baseline_statics = [
    "admission_age",
    "Female",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]

patient_full_features = patient_features_last.join(
    to_polars(target_trial_population).select(
        [
            *STAY_KEYS,
            *baseline_statics,
            *COMMON_DELTAS,
            COLNAME_INTERVENTION_STATUS,
            COLNAME_INCLUSION_START,
            COLNAME_MORTALITY_28D,
            COLNAME_MORTALITY_90D,
        ]
    ),
    on=STAY_KEYS,
    how="inner",
).to_pandas()

BINARY_FEATURES = [
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
CATEGORICAL_FEATURES = ["aki_stage"]
# dialysis needs a bit of rework
NUMERICAL_FEATURES = list(
    set(patient_full_features.columns).difference(
        set(
            [
                *BINARY_FEATURES,
                *CATEGORICAL_FEATURES,
                COLNAME_INCLUSION_START,
                COLNAME_INTERVENTION_STATUS,
                COLNAME_MORTALITY_28D,
                COLNAME_MORTALITY_90D,
                *STAY_KEYS,
                *COMMON_DELTAS,
            ]
        )
    )
)
# %%
# save the features
patient_matrix = patient_full_features[
    [
        COLNAME_PATIENT_ID,
        *NUMERICAL_FEATURES,
        *CATEGORICAL_FEATURES,
        *BINARY_FEATURES,
        COLNAME_MORTALITY_28D,
        COLNAME_MORTALITY_90D,
        COLNAME_INTERVENTION_STATUS,
    ]
]
# %% [markdown]
# ## Table 1
# %%
for col in BINARY_FEATURES:
    patient_full_features[col] = patient_full_features[col].fillna(0)

from sklearn.preprocessing import OneHotEncoder

categorical_features_one_hot = []
for cat_col in CATEGORICAL_FEATURES:
    enc = OneHotEncoder(sparse_output=False)
    categorical_encode = enc.fit_transform(patient_full_features[[cat_col]])
    categorical_encode = pd.DataFrame(
        categorical_encode,
        columns=[
            (enc.feature_names_in_[0] + "_" + str(cat_))
            for cat_ in enc.categories_[0]
        ],
    )
    patient_full_features = pd.concat(
        [patient_full_features, categorical_encode], axis=1
    )
categorical_features_one_hot += categorical_encode.columns.tolist()
# %%
table_1 = (
    patient_full_features[
        [
            COLNAME_INTERVENTION_STATUS,
            *BINARY_FEATURES,
            *categorical_features_one_hot,
            *NUMERICAL_FEATURES,
            *COMMON_DELTAS,
        ]
    ]
    .groupby(COLNAME_INTERVENTION_STATUS)
    .mean()
    .transpose()
)
table_1
