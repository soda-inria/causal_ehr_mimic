# %%
""""
This notebook searches for population prevalences for candidate trials to replication  
in the MIMIC-IV database.
"""
%reload_ext autoreload
%autoreload 2
from copy import deepcopy

import numpy as np
from caumim.target_population.utils import get_base_population
from caumim.constants import *
import polars as pl
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
# %%
tables = [file_.name for file_ in list(DIR2MIMIC.iterdir())]
tables.sort(reverse=True)
print(tables)
# %%
base_population = get_base_population(min_icu_survival_unit_day=1, min_los_icu_unit_day=1)
print(base_population.shape)

# %% [markdown]
# ARDS
ards_grace_period_unit_hour = 12
vitalsign  = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.vitalsign")
base_population_vitalsign = vitalsign.join(
    base_population[["stay_id", "subject_id", "icu_intime"]].lazy(), on="stay_id", how="inner"
    ).collect()
mi_population_spo2_at_admission = base_population_vitalsign.filter(
    (pl.col("charttime") - pl.col("icu_intime")).dt.hours() < ards_grace_period_unit_hour
).groupby(["subject_id"]).agg(
    pl.median("spo2").alias(f"spo2_median_at_admission")
)

base_population_bg = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.bg").join(
    base_population[["hadm_id","subject_id","icu_intime"]].lazy(), on="hadm_id", how="inner"
).collect()
mi_population_bg_at_admission = base_population_bg.filter(
    (pl.col("charttime") - pl.col("icu_intime")).dt.hours() <  ards_grace_period_unit_hour
).groupby(["subject_id"]).agg(
    pl.median("so2").alias(f"so2_median_at_admission"),
    pl.median("pao2fio2ratio").alias(f"pao2fio2ratio_median_at_admission"),
)
base_population_with_ards_info = mi_population_spo2_at_admission.join(
    mi_population_bg_at_admission, on="subject_id", how="left"
).to_pandas()
base_population_with_ards_info["ards_at_admission"] = (
    #(mild_ards_at_admission[f"spo2_median_at_admission"] < 90) |
    #(mild_ards_at_admission["so2_median_at_admission"] < 90) |
    (base_population_with_ards_info["pao2fio2ratio_median_at_admission"] < 300)
)
# number of nan for all 3 measures
n_patients_wo_o2_measures = (base_population_with_ards_info[["pao2fio2ratio_median_at_admission"]].isna().sum(axis=1) == 1).sum()
patients_w_ards_at_admission = base_population_with_ards_info.loc[
    base_population_with_ards_info["ards_at_admission"] == True
]
print("Number of patients without any Fi/O2 measures: ", n_patients_wo_o2_measures)
print(
    f"Number of patients with ards at admission: {patients_w_ards_at_admission.shape[0]}"
    )


# %%
# Nothing in the procedures
procedures = pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.d_icd_procedures/*")
procedures.write_csv(DIR2RESOURCES / "procedures.csv")
# %% [markdown]
# Looking for prone positioning
# candidates are 
# position por : 228474, no log
# position change: 227952 or 224066
# position : 224093
# position rest 227915 : Nope, very few logs (160)
chart_events = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.chartevents/*")
d_items = pl.read_parquet(DIR2MIMIC / "mimiciv_icu.d_items/*")
# %% 
position_events = chart_events.filter(
    pl.col("itemid").is_in([227952, 224066, 224093])
).collect().join(
    d_items.select(["itemid", "label"]), on="itemid", how="inner"
)
# %% 
# position change
print(
    position_events.filter(pl.col("itemid").is_in([227952, 224066]))["value"].value_counts().to_pandas()
)
# position
position_events.filter(pl.col("itemid")==224093)["value"].value_counts().to_pandas()

# type of positions 
# %%
# Intervention
position_prone = position_events.filter(pl.col("value").str.contains("Prone")).to_pandas()
intervention_pop = patients_w_ards_at_admission.merge(
    position_prone[["subject_id"]].drop_duplicates(), on="subject_id", how="inner"
)
intervention_pop

# %%
# Control
position_supine = position_events.filter(pl.col("value").str.contains("Supine")).to_pandas()
control_pop = patients_w_ards_at_admission.merge(
    position_supine[["subject_id"]].drop_duplicates(), on="subject_id", how="inner"
)
control_pop = control_pop.loc[~control_pop["subject_id"].isin(intervention_pop["subject_id"])]
control_pop
# TODO: ventilation for both population