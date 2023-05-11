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
event_features, feature_types = get_albumin_events_zhou_baseline(
    pd.read_parquet(albumin_cohort_folder / FILENAME_TARGET_POPULATION)
)
feature_types["binary_features"] += [
    "Female",
    "Emergency admission",
    "Insurance, Medicare",
]
feature_types["numerical_features"] += ["admission_age"]

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


# %%
# save the features
patient_matrix = patient_full_features[
    [
        COLNAME_PATIENT_ID,
        *feature_types["binary_features"],
        *feature_types["categorical_features"],
        *feature_types["numerical_features"],
        COLNAME_MORTALITY_28D,
        COLNAME_MORTALITY_90D,
        COLNAME_INTERVENTION_STATUS,
    ]
]
# %% [markdown]
# ## Table 1
# %%
for col in feature_types["binary_features"]:
    patient_full_features[col] = patient_full_features[col].fillna(0)

from sklearn.preprocessing import OneHotEncoder

categorical_features_one_hot = []
for cat_col in feature_types["categorical_features"]:
    enc = OneHotEncoder()
    categorical_encode = enc.fit_transform(
        patient_full_features[[cat_col]]
    ).todense()
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
"""table_1 = (
    patient_full_features[
        [
            COLNAME_INTERVENTION_STATUS,
            *feature_types["binary_features"],
            *categorical_features_one_hot,
            *feature_types["numerical_features"],
            *COMMON_DELTAS,
        ]
    ]
    .groupby(COLNAME_INTERVENTION_STATUS)
    .mean()
    .transpose()
)
table_1"""

from tableone import TableOne

# To avoid plotting both category for binary features:
limit_binary = {
    k: 1
    for k in [*feature_types["binary_features"], *categorical_features_one_hot]
}
mytable = TableOne(
    patient_full_features[
        [
            COLNAME_INTERVENTION_STATUS,
            *feature_types["binary_features"],
            *categorical_features_one_hot,
            *feature_types["numerical_features"],
            *COMMON_DELTAS,
        ]
    ],
    categorical=[
        *feature_types["binary_features"],
        *categorical_features_one_hot,
    ],
    limit=limit_binary,
    groupby=COLNAME_INTERVENTION_STATUS,
)
cohort_name = albumin_cohort_folder.name
# small esthetical changes
table_1_ = mytable.tableone.droplevel(1)
table_1_.columns.set_levels(
    [
        *list(table_1_.columns.levels[1][:2]),
        "Cristalloids only",
        "Cristalloids + Albumin",
    ],
    level=1,
    inplace=True,
)
mytable.tableone = table_1_
mytable.to_latex = mytable.tableone.to_latex
mytable.to_latex(DIR2DOCS_IMG / cohort_name / "table1.tex")
# %%
