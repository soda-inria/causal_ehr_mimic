# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
import numpy as np
from caumim.target_population.utils import get_base_population, get_flat_information, get_drug_names_from_str
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
base_population = get_base_population(
    min_icu_survival_unit_day=1, min_los_icu_unit_day=1
)
# sepsis3 ? 
sepsis3 = pl.read_parquet(DIR2MIMIC / "mimiciv_derived.sepsis3").filter(
    pl.col("sepsis3").eq(True)
).join(base_population, on="subject_id", how="inner")
sepsis3_subject_ids = sepsis3["subject_id"].unique().to_pandas()
sepsis3_stay_ids = sepsis3["stay_id"].unique().to_pandas()
print("Number of patients with sepsis3: ", len(sepsis3_subject_ids))
print("Number of icu admissions with sepsis3: ", len(sepsis3_stay_ids))
# %%
# scan emar and pharmacy for fludrocortisone and hydrocortisone and norepinephrine
emar = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.emar/*")
emar_detail = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.emar_detail/*")
pharmacy = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.pharmacy/*")
# %%
norepinephrine_str = "norepinephrine"
norepinephrine_medications_names = get_drug_names_from_str(emar, norepinephrine_str)
norepinephrine_emar = emar.join(
    pl.DataFrame({"medication": norepinephrine_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()

norepinephrine_medications_names = get_drug_names_from_str(pharmacy, norepinephrine_str)
norepinephrine_pharmacy = norepinephrine_delivrances = emar.join(
    pl.DataFrame({"medication": norepinephrine_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()

norepinephrine_input = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.inputevents/*").filter(
    pl.col("itemid").is_in([221906])).collect().to_pandas()
# %%
norepinephrine_events = pd.concat(
    [
    norepinephrine_emar[["subject_id", "hadm_id", "charttime"]].rename(
        columns={"charttime": "starttime"}
    ).assign(
        medication_source_table="emar"
    ), 
    norepinephrine_pharmacy[["subject_id", "hadm_id", "charttime"]].rename(
        columns={"charttime": "starttime"}
    ).assign(
        medication_source_table="pharmacy"
    ),
    norepinephrine_input[["subject_id", "hadm_id", "starttime"]].assign(
        medication_source_table="inputevents"
    )]
)
# %%
sepsis3_w_norepinephrine_first_event = sepsis3.to_pandas().merge(
    norepinephrine_events.groupby(["subject_id", "hadm_id"]).agg(
    {"starttime": "min"}
).reset_index().rename(columns={"starttime": "first_norepinephrine"}), on=["subject_id", "hadm_id"]
)
# %%
# Control
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
fludrocortisone_subject_ids = pd.concat(
    [ fludrocortisone_emar["subject_id"], fludrocortisone_pharmacy["subject_id"] ]
).unique()
print("Number of patients with fludrocortisone : ", len(fludrocortisone_subject_ids))
print("Number of patients with fludrocortisone and sepsis: ", len(set(fludrocortisone_subject_ids).intersection(set(sepsis3_w_norepinephrine_first_event["subject_id"]))))

# %%
# %% intervention
hydrocortisone_str = "hydrocortisone"
hydrocortisone_medications_names = get_drug_names_from_str(emar, hydrocortisone_str)
hydrocortisone_emar = emar.join(
    pl.DataFrame({"medication": hydrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()

hydrocortisone_medications_names = get_drug_names_from_str(pharmacy, hydrocortisone_str)
hydrocortisone_pharmacy = hydrocortisone_delivrances = emar.join(
    pl.DataFrame({"medication": hydrocortisone_medications_names}).lazy(),
    on="medication",
).collect().to_pandas()

hydrocortisone_injected = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.chartevents/*").filter(
    pl.col("itemid").is_in(np.array([220611, 227463]).astype("int32"))).collect().to_pandas()
hydrocortisone_injected
# %%
hydrocortisone_subject_ids = pd.concat(
    [ hydrocortisone_emar["subject_id"], hydrocortisone_pharmacy["subject_id"], hydrocortisone_injected["subject_id"]]
).unique()
print("Number of patients with hydrocortisone: ", len(hydrocortisone_subject_ids))
print("Number of patients with sepsis and hydrocortisone-only: ", len(
    set(hydrocortisone_subject_ids)
        .intersection(set(sepsis3_w_norepinephrine_first_event["subject_id"]))
        .difference(set(fludrocortisone_subject_ids))
))
# %%
# How many patient with sepsis3 and both hydrocortisone and fludrocortisone?
both_cortisones_subject_ids = set(hydrocortisone_subject_ids).intersection(set(fludrocortisone_subject_ids)).intersection(set(sepsis3_subject_ids))
print("Number of patients with sepsis and both hydrocortisone and fludrocortisone: ", len(both_cortisones_subject_ids))
# %% Is it given in the ICU? Fludrocortisone does not exists for injection But
# Hyrocortisone does. In the d_items table, we can find  Zcortisol and cortisone entry,
# another name for hydrocortisone.  
d_items = pl.read_parquet(DIR2MIMIC / "mimiciv_icu.d_items/*").to_pandas()
medication_names = d_items[d_items["label"].str.contains("cortisone", case=False)]["label"].unique()
medication_names
# %%
