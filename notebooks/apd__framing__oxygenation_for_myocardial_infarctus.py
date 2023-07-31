# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
from copy import deepcopy

import numpy as np
from caumim.framing.utils import get_base_population, get_drug_names_from_str
from caumim.constants import *
import polars as pl
import pandas as pd

from caumim.utils import to_polars
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
# %%
tables = [file_.name for file_ in list(DIR2MIMIC.iterdir())]
tables.sort(reverse=True)
print(tables)
# %%
billing_diagnoses = pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.diagnoses_icd/*")
billing_diagnoses.head()
# %%
d_diagnoses = pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.d_icd_diagnoses/*")
# %%
# exploration by hand
#infarction = pl.col("long_title").str.to_lowercase().str.contains("infarction")
#myocardial = pl.col("long_title").str.to_lowercase().str.contains("myocardial")
myocardial_infarction_codes = ["410", "I21", "I22"]
mask_myocardial_infarction_l = [pl.col("icd_code").str.starts_with(code) for code in myocardial_infarction_codes]
mask_myocardial_infarction = mask_myocardial_infarction_l[0]
for mask_ in mask_myocardial_infarction_l[1:]:
    mask_myocardial_infarction = mask_myocardial_infarction | mask_
# %%
infarction_codes = d_diagnoses.filter(mask_myocardial_infarction)
infarction_codes.to_pandas().to_csv(DIR2META_CONCEPTS / "infarction_codes.csv", index=False)
infarction_codes

# %% [markdown]
# Miocardial infarctus from ICD codes
# %%
base_population = get_base_population(min_icu_survival_unit_day=1, min_los_icu_unit_day=1)
print(base_population.shape)

mi_diagnoses = billing_diagnoses.filter(
    pl.col("icd_code").is_in(infarction_codes["icd_code"])
)
mi_diagnoses_icu_detail = mi_diagnoses[["hadm_id"]].unique().join(to_polars(base_population), on="hadm_id", how="inner")
n_patients_mi = mi_diagnoses_icu_detail.to_pandas()["subject_id"].nunique()
n_hadm_mi = mi_diagnoses_icu_detail.to_pandas()["hadm_id"].nunique()
print(f"Number of patients with myocardial infarction: {n_patients_mi}")
print(f"Number of admissions with myocardial infarction: {n_hadm_mi}")
# %% 
# Oxygen therapy should be searched for the first hours only to avoid ITB

# Acute Heart failure 

# Hypoxemia (SpO2 <90% or PaO2/FiO2 < 300 mmHg) 
# %%
# aggregation (min, max, mean) is too coarse for proper definition of hypoxemia at admission.
#first_day_vitalsign = pl.read_parquet(DIR2MIMIC / "mimiciv_derived.first_day_vitalsign")
## define hypoxemia as median(SpO2) < 90% or meadian(PaO2/FiO2) < 300 mmHg during the first 6 hours.
### SpO2
hypoxemia_grace_period_unit_hour = 2
vitalsign  = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.vitalsign")
mi_population_vitalsign = vitalsign.join(
    mi_diagnoses_icu_detail[["stay_id", "subject_id", "icu_intime"]].lazy(), on="stay_id", how="inner"
    ).collect()
mi_population_spo2_at_admission = mi_population_vitalsign.filter(
    (pl.col("charttime") - pl.col("icu_intime")).dt.hours() < hypoxemia_grace_period_unit_hour
).groupby(["subject_id"]).agg(
    pl.median("spo2").alias(f"spo2_median_at_admission")
)

mi_population_bg = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.bg").join(
    mi_diagnoses_icu_detail[["hadm_id","subject_id","icu_intime"]].lazy(), on="hadm_id", how="inner"
).collect()
mi_population_bg_at_admission = mi_population_bg.filter(
    (pl.col("charttime") - pl.col("icu_intime")).dt.hours() <  hypoxemia_grace_period_unit_hour
).groupby(["subject_id"]).agg(
    pl.median("so2").alias(f"so2_median_at_admission"),
    pl.median("pao2fio2ratio").alias(f"pao2fio2ratio_median_at_admission"),
)
mi_hypoxemia_at_admission = mi_population_spo2_at_admission.join(
    mi_population_bg_at_admission, on="subject_id", how="left"
).to_pandas()
mi_hypoxemia_at_admission["hypoxemia_at_admission"] = (
    (mi_hypoxemia_at_admission[f"spo2_median_at_admission"] < 90) |
    (mi_hypoxemia_at_admission["so2_median_at_admission"] < 90) |
    (mi_hypoxemia_at_admission["pao2fio2ratio_median_at_admission"] < 300)
)
mi_wo_hypoxemia_at_admission = mi_hypoxemia_at_admission[mi_hypoxemia_at_admission["hypoxemia_at_admission"] == False]
print(
    f"Number of MI patients with hypoxemia at admission: {mi_hypoxemia_at_admission['hypoxemia_at_admission'].sum()}"
    )
# number of nan for all 3 measures
n_patients_wo_o2_measures = (mi_hypoxemia_at_admission[["spo2_median_at_admission", "so2_median_at_admission", "pao2fio2ratio_median_at_admission"]].isna().sum(axis=1) == 3).sum()
print("Number of patients without any O2 measures: ", n_patients_wo_o2_measures)

eligible_population = mi_diagnoses_icu_detail.to_pandas().merge(
    mi_wo_hypoxemia_at_admission[["subject_id"]].drop_duplicates(), on="subject_id", how="inner"
)
print(f"Number of patients with MI and without hypoxemia at admission (within 2 hours): {eligible_population.shape[0]}")
# %%
# %% [markdown]
# Looking on treatments
# %%
mi_population_ventilation = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.ventilation").join(
    pl.DataFrame(eligible_population)[["stay_id", "subject_id", "icu_intime"]].lazy(), on="stay_id", how="inner"
).collect()
ventilation_grace_period_unit_hour = 12
mi_population_ventilation_stay_beginning = mi_population_ventilation.filter(
    (pl.col("starttime") - pl.col("icu_intime")).dt.hours() < ventilation_grace_period_unit_hour
)
n_ventilated = mi_population_ventilation_stay_beginning["stay_id"].n_unique()
print(f"Number of patients with ventilation during first {ventilation_grace_period_unit_hour} hours: {n_ventilated}")
print(mi_population_ventilation_stay_beginning["ventilation_status"].value_counts())
# %%
mi_population_ventilation_stay_beginning.with_columns( 
    (mi_population_ventilation_stay_beginning["endtime"] - mi_population_ventilation_stay_beginning["starttime"]).dt.minutes().alias("ventilation_len")
).describe()

supplemental_o2 = mi_population_ventilation_stay_beginning.filter(
    pl.col("ventilation_status").is_in(["NonInvasiveVent", "SupplementalOxygen"])
).to_pandas()
# %%
# Intervention
intervention_population = eligible_population.merge(
    supplemental_o2[["subject_id"]].drop_duplicates(), on="subject_id", how="inner"
)
# Control
control_population = eligible_population.loc[
    (eligible_population["subject_id"].isin(
    mi_population_ventilation_stay_beginning["subject_id"].unique().to_list()) == False)
]
print(f"Number of patients in intervention group: {intervention_population.shape[0]}")
print(f"Number of patients in control group: {control_population.shape[0]}")

# %%
d_item = pd.read_parquet(DIR2MIMIC / "mimiciv_icu.d_items")
d_item.to_csv(DIR2RESOURCES / "d_item.csv", index=False)