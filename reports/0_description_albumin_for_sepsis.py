# %%
# %load_ext autoreload
# %autoreload 2
import polars as pl
import pandas as pd
from caumim.variables.selection import (
    get_albumin_events_zhou_baseline,
    get_antibiotics_event_from_atc4,
    get_antibiotics_event_from_drug_name,
    get_comorbidity,
)
from caumim.variables.aggregation import aggregate_medically_sepsis_albumin
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
inclusion_criteria_full_stay = pd.read_parquet(
    albumin_cohort_folder / FILENAME_TARGET_POPULATION
)
# Change the inclusion criteria to be the full stay / or the first 24 hours
# instead of only up to the intervention.
acceptable_followup_windows = ["full_icu_stay", "24h", "up_to_intervention"]
followup_window = "24h"

if followup_window == "full_icu_stay":
    inclusion_criteria_full_stay[
        COLNAME_INCLUSION_START
    ] = inclusion_criteria_full_stay["outtime"]
elif followup_window == "24h":
    inclusion_criteria_full_stay[
        COLNAME_INCLUSION_START
    ] = inclusion_criteria_full_stay["intime"] + pd.Timedelta(24, unit="h")
elif followup_window == "up_to_intervention":
    pass
else:
    raise ValueError(
        f"followup_window must be one of {acceptable_followup_windows}"
    )
# %%
## Adding comorbidities
comorbidities_events, comorbidities_feature_types = get_comorbidity(
    inclusion_criteria_full_stay
)
albumin_comorbidities = [
    "myocardial_infarct",
    "malignant_cancer",
    "diabetes_with_cc",
    "diabetes_without_cc",
    "metastatic_solid_tumor",
    "severe_liver_disease",
    "renal_disease",
]
comorbidities_events = comorbidities_events.filter(
    pl.col("code").is_in(albumin_comorbidities)
).select(COLNAMES_EVENTS)
# %%
event_features, feature_types = get_albumin_events_zhou_baseline(
    inclusion_criteria_full_stay
)
# Add the static feature types
feature_types.binary_features += [
    "Female",
    "White",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]
feature_types.numerical_features += ["admission_age"]

print(event_features.shape)
event_features["code"].value_counts().sort("counts", descending=True)

event_features = pl.concat([event_features, comorbidities_events])
feature_types.binary_features += albumin_comorbidities
# %%
# medically grounded variable aggregation
aggregated_events = aggregate_medically_sepsis_albumin(
    event_features=event_features
)
# normalize urine output:
aggregated_events = aggregated_events.with_columns(
    (pl.col("urineoutput") / pl.col("Weight").fill_null(70)).alias(
        "urineoutput"
    )
)
# %%
# Join with static features:
baseline_statics = [
    "admission_age",
    "Female",
    "White",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]

patient_full_features = aggregated_events.join(
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
        *feature_types.binary_features,
        *feature_types.categorical_features,
        *feature_types.numerical_features,
        COLNAME_MORTALITY_28D,
        COLNAME_MORTALITY_90D,
        COLNAME_INTERVENTION_STATUS,
    ]
]
# %% [markdown]
# ## Table 1
# %%
for col in feature_types.binary_features:
    patient_full_features[col] = patient_full_features[col].fillna(0)

from sklearn.preprocessing import OneHotEncoder

categorical_features_one_hot = []
for cat_col in feature_types.categorical_features:
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
from tableone import TableOne

# To avoid plotting both category for binary features:
# limit_binary = {
#     k: 1
#     for k in [*feature_types.binary_features, *categorical_features_one_hot]
# }
mytable = TableOne(
    patient_full_features[
        [
            COLNAME_INTERVENTION_STATUS,
            *feature_types.binary_features,
            *categorical_features_one_hot,
            *feature_types.numerical_features,
            *COMMON_DELTAS,
        ]
    ],
    categorical=[
        *feature_types.binary_features,
        *categorical_features_one_hot,
    ],
    # limit=limit_binary,
    groupby=COLNAME_INTERVENTION_STATUS,
)
cohort_name = albumin_cohort_folder.name

# dirty fix to keep only class one for  binary features
table_1_ = mytable.tableone.reset_index()
table_1_ = table_1_.loc[table_1_["level_1"].isin(["", "1", "1.0"])].drop(
    columns="level_1"
)
table_1_.columns = table_1_.columns.droplevel(0)
table_1_.rename(
    columns={"0": "Cristalloids only", "1": "Cristalloids + Albumin"},
    inplace=True,
)
table_1_.set_index("", inplace=True)
mytable.tableone = table_1_
mytable.to_latex = mytable.tableone.to_latex
mytable.to_latex(DIR2DOCS_IMG / cohort_name / f"table1_{followup_window}.tex")
mytable.tableone
# %%
