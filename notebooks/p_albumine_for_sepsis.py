# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
import numpy as np
from caumim.target_population.utils import create_cohort_folder, get_base_population, get_cohort_hash, get_flat_information, get_drug_names_from_str, roll_inclusion_criteria
from caumim.constants import *
import polars as pl
import pandas as pd
from sklearn.utils import Bunch
from datetime import datetime
from caumim.target_population.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
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

# Inclusion event: First administration of crystalloids during the 24 first
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
target_population
# TODO:WARNING: I lost 7930 persons comapred to the initial notebook.
# %%
albumin_itemids = [
    # 220861, #"Albumin (Human) 20% Not in use
    220862, #Albumin 25%,Albumin 25%
    # 220863, #Albumin (Human) Not in use
    220864, #Albumin 5%
]
albumin = input_events.filter(
    pl.col("itemid").is_in(albumin_itemids)).collect().to_pandas()
albumin_events_for_sepsis = crystalloids_events_for_sepsis[["stay_id", 'icu_intime']].drop_duplicates().merge(
    albumin, on="stay_id", how="inner")
print("Number of patients with sepsis3 and albumin: ", len(albumin_events_for_sepsis["subject_id"].unique()))
# %% timing of treatments
# first crystalloids
first_crystalloids = (
    crystalloids_events_for_sepsis.sort_values("starttime").groupby("stay_id")
        .first()[["starttime", "icu_intime"]].reset_index().rename(columns={"starttime": "crystalloids_starttime"})
)
first_crystalloids["delta_crystalloids_icu_intime"] = (
    first_crystalloids["crystalloids_starttime"] - first_crystalloids["icu_intime"]
)
first_crystalloids_in24h = first_crystalloids.loc[
    first_crystalloids["delta_crystalloids_icu_intime"].dt.days == 0 
]
# # first albumin
first_albumin = (
    albumin_events_for_sepsis.sort_values("starttime").groupby("stay_id")
        .first()[["starttime", "icu_intime"]].reset_index().rename(columns={"starttime": "albumin_starttime"})
) 
first_albumin["delta_albumin_icu_intime"] = (
    first_albumin["albumin_starttime"] - first_albumin["icu_intime"]
)
first_albumin_in24h = first_albumin.loc[
    first_albumin["delta_albumin_icu_intime"].dt.days == 0 
]
# %%
# albumin should not preced crystalloids
treaments_start_per_stay = first_crystalloids_in24h.merge(
    first_albumin_in24h, on="stay_id", how="left"
)
albumin_preceded_crystalloids = treaments_start_per_stay.loc[
    treaments_start_per_stay["crystalloids_starttime"] > treaments_start_per_stay["albumin_starttime"]
]
# %%
# treatment_stays
treatment_stays = sepsis3_stays.merge(
    first_albumin_in24h[["stay_id"]].drop_duplicates(), on="stay_id", how="inner"
)  
treatment_stays.loc[
    ~treatment_stays["stay_id"].isin(albumin_preceded_crystalloids["stay_id"])
]

# control stays
control_stays = sepsis3_stays.merge(
    first_crystalloids_in24h[["stay_id"]].drop_duplicates(), on="stay_id", how="left")
control_stays = control_stays.loc[
    ~control_stays["stay_id"].isin(treatment_stays["stay_id"])
]
print("Number of treated patients (sepsis3 and crystalloids/albumin combination) in 24h:", len(control_stays["subject_id"].unique()))
print("Number of control patients (sepsis3 and crystalloids only in 24h:", len(treatment_stays["subject_id"].unique()))
