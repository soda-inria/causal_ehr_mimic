import polars as pl
from caumim.constants import DIR2MIMIC
import pandas as pd


def get_flat_information(dir2mimic: str = DIR2MIMIC):

    """Build flat informations for all mimic, used for population description."""
    patients = pl.read_parquet(
        dir2mimic / "mimiciv_hosp.patients/*"
    ).to_pandas()
    admissions = pl.read_parquet(
        dir2mimic / "mimiciv_hosp.admissions/*"
    ).to_pandas()
    admissions = admissions.drop(["subject_id"], axis=1)
    icustays = pl.read_parquet(dir2mimic / "mimiciv_icu.icustays/*").to_pandas()

    extended_icustays = icustays.merge(admissions, on="hadm_id", how="left")
    flat_information = extended_icustays.merge(
        patients, on="subject_id", how="left"
    )

    # age at hospital admission
    flat_information["age"] = (
        pd.to_datetime(flat_information["admittime"]).dt.year
        - flat_information["anchor_year"]
        + flat_information["anchor_age"]
    )
    # delta between ICU and hospital admission in hours
    flat_information["delta_icu_admission_hours"] = (
        pd.to_datetime(flat_information["intime"])
        - pd.to_datetime(flat_information["admittime"])
    ).astype("timedelta64[m]") / 60

    # The date of birth was then set to exactly 300 years before their first admission
    older_than_89_mask = flat_information["age"] <= -100
    flat_information.loc[older_than_89_mask, "age"] = 90
    flat_information["age>89"] = older_than_89_mask
    flat_information["gender"] = (flat_information["gender"] == "M").astype(int)
    return flat_information


def get_drug_names_from_str(df: pl.DataFrame, medication_str: str):
    """Get drugs from a string."""
    medication_names = list(df.select("medication").unique().collect()["medication"])
    medication_names = [medication for medication in medication_names if medication is not None]
    medications_names = [medication for medication in medication_names if (medication.lower().find(medication_str) != -1)]
    return medications_names