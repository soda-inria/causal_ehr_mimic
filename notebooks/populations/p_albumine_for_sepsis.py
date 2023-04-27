# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
import numpy as np
from caumim.framing.utils import create_cohort_folder, get_base_population, roll_inclusion_criteria
from caumim.constants import *
import polars as pl
import pandas as pd
from sklearn.utils import Bunch
from datetime import datetime
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.utils import to_lazyframe
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
# %%
tables = [file_.name for file_ in list(DIR2MIMIC.iterdir())]
tables.sort(reverse=True)
print(tables)
# %%
cohort_config = Bunch(**COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
print(cohort_config)
create_cohort_folder(cohort_config)
# %%
# 1 - Define the inclusion events, ie. the event that defines when a patient
# enter the cohort.

# Inclusion start: First administration of crystalloids during the 24 first
# hours of ICU stay
input_events = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.inputevents/*")
icu_stays = pl.scan_parquet(DIR2MIMIC/ "mimiciv_icu.icustays/*")
# full list of crystalloids taken from :https://www.ncbi.nlm.nih.gov/books/NBK537326/
crystalloids_itemids = [
    226364,  # operating room crystalloids
    226375,  # post-anesthesia care unit crystalloids
    225158,  # NaCl 0.9%,
    225159,  # NaCl 0.45%,
    225161,  # NaCl 3%
    220967,  # Dextrose 5% / Ringers Lactate,
    220968,  # Dextrose 10% / Ringers
    220964,  # "Dextrose 5% / Saline 0,9%"
    220965,  # "Dextrose 5% / Saline 0,45%"
]
crystalloids_inputs = (
    input_events.filter(pl.col("itemid").is_in(crystalloids_itemids)).join(
        icu_stays.select(["stay_id", "intime"]), on="stay_id", how="inner"
    )
)
first_crystalloids = (
    crystalloids_inputs.sort(["stay_id", "starttime"])
    .groupby("stay_id")
    .agg([pl.first("starttime"), pl.first("intime")])
    .collect()
    .to_pandas()
    .rename(columns={"starttime": COLNAME_INCLUSION_START})
)
first_crystalloids["delta_crystalloids_icu_intime"] = (
    first_crystalloids[COLNAME_INCLUSION_START]
    - first_crystalloids["intime"]
)
# Only during first day
inclusion_event = first_crystalloids.loc[
    first_crystalloids["delta_crystalloids_icu_intime"].dt.days == 0
]
# %%
base_population = get_base_population(
    min_age=cohort_config.min_age,
    min_icu_survival_unit_day=cohort_config.min_icu_survival_unit_day,
    min_los_icu_unit_day=cohort_config.min_los_icu_unit_day,
)
# sepsis
sepsis3_stays = pd.read_parquet(DIR2MIMIC / "mimiciv_derived.sepsis3")
sepsis3_stays = sepsis3_stays.loc[sepsis3_stays["sepsis3"] == True, ["stay_id"]]
inclusion_criteria = {
    'base_population': base_population,
    'sepsis3': sepsis3_stays,
    'inclusion_event': inclusion_event
}
target_population, inclusion_counts = roll_inclusion_criteria(inclusion_criteria)
target_population.head(3)
# %%
# 3 - Define the treatment events  
albumin_itemids = [
    # 220861, #"Albumin (Human) 20% Not in use
    220862, #Albumin 25%,Albumin 25%
    # 220863, #Albumin (Human) Not in use
    220864, #Albumin 5%
]
albumin = input_events.filter(
    pl.col("itemid").is_in(albumin_itemids))
combined_albumin_for_target_population = to_lazyframe(
    target_population[["stay_id", 'icu_intime', COLNAME_INCLUSION_START]].drop_duplicates()).join(
    albumin, on="stay_id", how="inner")

# First albumin
first_albumin = (
    combined_albumin_for_target_population.sort("starttime").groupby("stay_id")
        .agg([pl.first("starttime"), pl.first("icu_intime"), pl.first(COLNAME_INCLUSION_START)])
        .collect().to_pandas().rename(columns={"starttime": COLNAME_INTERVENTION_START})
) 
# Consider only first day albumin
first_albumin["delta_albumin_icu_intime"] = (
    first_albumin[COLNAME_INTERVENTION_START] - first_albumin["icu_intime"]
)
first_albumin_in24h = first_albumin.loc[
    first_albumin["delta_albumin_icu_intime"].dt.days == 0 
]
first_albumin_in24h = first_albumin_in24h.loc[
    first_albumin_in24h[COLNAME_INTERVENTION_START] > first_albumin_in24h[COLNAME_INCLUSION_START]
]
first_albumin_in24h
# %%
# 4- Define treatment and control population:
target_trial_population = target_population.merge(
    first_albumin_in24h[["stay_id", COLNAME_INTERVENTION_START]].drop_duplicates(), on="stay_id", how="left")

target_trial_population[COLNAME_INTERVENTION_STATUS] = target_trial_population[COLNAME_INTERVENTION_START].notnull()

print("Number of treated patients (sepsis3 and crystalloids/albumin combination) in 24h:", target_trial_population[COLNAME_INTERVENTION_STATUS].sum())
print("Number of control patients (sepsis3 and crystalloids only in 24h:", (1 - target_trial_population[COLNAME_INTERVENTION_STATUS]).sum())
# %%
# 5 - Define outcomes
cohort_folder = create_cohort_folder(cohort_config)
target_trial_population = pd.read_parquet(cohort_folder/"target_population")

# 5 - Define outcomes
# 28-days and 90-days mortality
mask_dod = target_trial_population["dod"].notnull()
days_to_death = (
    target_trial_population["dod"]
    - target_trial_population[COLNAME_INCLUSION_START]
).dt.days

target_trial_population[COLNAME_MORTALITY_28D] = (
    mask_dod & (days_to_death <= 28)
).astype(int)
target_trial_population[COLNAME_MORTALITY_90D] = (
    mask_dod & (days_to_death <= 90)
).astype(int)

col_name_outcomes = [COLNAME_MORTALITY_28D, COLNAME_MORTALITY_90D]
for outcome in col_name_outcomes:
    print(
        f"Outcome `{outcome}` prevalence: {100 * target_trial_population[outcome].mean():.2f}%"
    )
