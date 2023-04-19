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
    medication_names = list(
        df.select("medication").unique().collect()["medication"]
    )
    medication_names = [
        medication for medication in medication_names if medication is not None
    ]
    medications_names = [
        medication
        for medication in medication_names
        if (medication.lower().find(medication_str) != -1)
    ]
    return medications_names


def get_base_population(
    min_age: float = 18,
    min_los_icu_unit_day: float = 1,
    min_icu_survival_unit_day: float = 1,
):
    """
    Get base population for the study filtering on minimal age, minimal icu_los
    and minimal in_icu survival.
    Take only the first icu stay for each patient.

    Args:
        min_age (float, optional): _description_. Defaults to 18.
        min_los_icu_unit_day (float, optional): _description_. Defaults to 1.
        min_icu_survival_unit_day (float, optional): _description_. Defaults to
        1.

    Returns:
        _type_: _description_
    """
    first_icu_stay_over18 = (
        pl.read_parquet(DIR2MIMIC / "mimiciv_derived.icustay_detail")
        .filter(
            (pl.col("first_icu_stay") == True)
            & (pl.col("admission_age") >= min_age)
        )
        .sort(["subject_id", "icu_intime"])
        .groupby("subject_id")
        .first()
    )

    mask_icu_los_gt_24 = pl.col("los_icu") >= min_los_icu_unit_day
    mask_alive_at_24hours = (
        first_icu_stay_over18["dod"] - first_icu_stay_over18["icu_intime"]
    ).dt.hours() >= (min_icu_survival_unit_day / 24)
    base_population = first_icu_stay_over18.filter(
        mask_icu_los_gt_24 & mask_alive_at_24hours
    )

    return base_population
