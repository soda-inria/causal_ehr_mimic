# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
from caumim.target_population.utils import get_flat_information, get_drug_names_from_str
from caumim.constants import *
import polars as pl
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
# %%
tables = [file_.name for file_ in list(DIR2MIMIC.iterdir())]
tables.sort(reverse=True)
print(tables)
patients = pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.patients/*")
patients.head()
# %%
# sepsis3 ? 
sepsis3 = pl.read_parquet(DIR2MIMIC / "mimiciv_derived.sepsis3")
sepsis3_subject_ids = sepsis3["subject_id"].unique().to_pandas()
sepsis3_stay_ids = sepsis3["stay_id"].unique().to_pandas()
sepsis3_hadm_ids = pl.read_parquet(DIR2MIMIC / "mimiciv_icu.icustays/*").join(
    pl.DataFrame({"stay_id": sepsis3_stay_ids}),
    on="stay_id",
).to_pandas()["hadm_id"].unique()
print("Number of patients with sepsis3: ", len(sepsis3_subject_ids))
print("Number of icu admissions with sepsis3: ", len(sepsis3_stay_ids))
print("Number of hospital admissions with sepsis3: ", len(sepsis3_hadm_ids))
# %%
# scan emar 
emar = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.emar/*")
emar_detail = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.emar_detail/*")
# %%
hydrocortisone_str = "hydrocortisone"
hydrocortisone_medications_names = get_drug_names_from_str(emar, hydrocortisone_str)
hydrocortisone_emar = emar.join(
    pl.DataFrame({"medication": hydrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()
# %%
pharmacy = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.pharmacy/*")
hydrocortisone_medications_names = get_drug_names_from_str(pharmacy, hydrocortisone_str)
hydrocortisone_pharmacy = hydrocortisone_delivrances = emar.join(
    pl.DataFrame({"medication": hydrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()
# %%
hydrocortisone_hadm_ids = pd.concat(
    [ hydrocortisone_emar["hadm_id"], hydrocortisone_pharmacy["hadm_id"] ]
).unique()
hydrocortisone_subject_ids = pd.concat(
    [ hydrocortisone_emar["subject_id"], hydrocortisone_pharmacy["subject_id"] ]
).unique()
print("Number of patients with hydrocortisone: ", len(hydrocortisone_subject_ids))
print("Number of patients with sepsis and hydrocortisone: ", len(set(hydrocortisone_subject_ids).intersection(set(sepsis3_subject_ids))))
print("Number of admissions with sepsis and hydrocortisone: ", len(set(hydrocortisone_hadm_ids).intersection(set(sepsis3_hadm_ids))))
# %%
fludrocortisone_str = "fludrocortisone"
fludrocortisone_medications_names = get_drug_names_from_str(emar, fludrocortisone_str)
fludrocortisone_emar = emar.join(
    pl.DataFrame({"medication": fludrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()

fludrocortisone_medications_names = get_drug_names_from_str(pharmacy, fludrocortisone_str)
fludrocortisone_pharmacy = fludrocortisone_delivrances = emar.join(
    pl.DataFrame({"medication": fludrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()
# %%
fludrocortisone_hadm_ids = pd.concat(
    [ fludrocortisone_emar["hadm_id"], fludrocortisone_pharmacy["hadm_id"] ]
).unique()
fludrocortisone_subject_ids = pd.concat(
    [ fludrocortisone_emar["subject_id"], fludrocortisone_pharmacy["subject_id"] ]
).unique()

print("Number of patients with fludrocortisone: ", len(fludrocortisone_subject_ids))
print("Number of admissions with fludrocortisone: ", len(fludrocortisone_hadm_ids))
# %%
# How many patient with sepsis3 and both hydrocortisone and fludrocortisone?
both_cortisones_subject_ids = set(hydrocortisone_subject_ids).intersection(set(fludrocortisone_subject_ids)).intersection(set(sepsis3_subject_ids))
both_cortisones_hadm_ids = set(hydrocortisone_hadm_ids).intersection(set(fludrocortisone_hadm_ids))
print("Number of patients with sepsis and both hydrocortisone and fludrocortisone: ", len(both_cortisones_subject_ids))
print("Number of admissions with sepsis and both hydrocortisone and fludrocortisone: ", len(set(both_cortisones_hadm_ids).intersection(set(sepsis3_hadm_ids))))
# %% Is it given in the ICU?
d_items = pl.read_parquet(DIR2MIMIC / "mimiciv_icu.d_items/*").to_pandas()
medication_names = d_items[d_items["label"].str.contains("cortisone", case=False)]["label"].unique()
medication_names
# %%
