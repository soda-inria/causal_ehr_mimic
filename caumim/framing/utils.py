from datetime import datetime
from pathlib import Path
from typing import Dict
import polars as pl
from caumim.constants import DIR2COHORT, DIR2MIMIC
import pandas as pd

from caumim.utils import to_lazyframe, to_pandas


def roll_inclusion_criteria(
    inclusion_criteria: Dict[str, pd.DataFrame], cohort_folder: str = None
) -> pd.DataFrame:
    if "inclusion_event" not in inclusion_criteria.keys():
        raise ValueError(
            "a inclusion_event dataframe must be in the inclusion criteria dictionary"
        )
    if cohort_folder is not None:
        cohort_path = Path(cohort_folder)
        cohort_path.mkdir(parents=True, exist_ok=True)

    target_population = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.icustays/*")
    inclusion_counts = {
        "full_population": target_population.select(pl.n_unique("stay_id"))
        .collect()
        .to_numpy()[0][0]
    }
    for inclusion_name, inclusion_df in inclusion_criteria.items():
        target_population = target_population.join(
            to_lazyframe(inclusion_df), on="stay_id", how="inner"
        )
        inclusion_counts[inclusion_name] = (
            target_population.select(pl.n_unique("stay_id"))
            .collect()
            .to_numpy()[0][0]
        )
    if cohort_folder is not None:
        pd.DataFrame.from_dict(inclusion_counts, orient="index").to_csv(
            cohort_path / "inclusion_counts.csv"
        )
    return to_pandas(target_population)


def get_base_population(
    min_age: float = 18,
    min_los_icu_unit_day: float = 1,
    min_icu_survival_unit_day: float = 1,
) -> pd.DataFrame:
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

    return to_pandas(base_population)


def get_cohort_hash(cohort_config: Dict):
    return str(hash("__".join([f"{k}_{v}" for k, v in cohort_config.items()])))


def create_cohort_folder(cohort_config: Dict) -> Path:
    # cohort_hash = get_cohort_hash(cohort_config)
    # t0 = datetime.now().strftime("%Y-%m-%d")
    cohort_folder = DIR2COHORT / f"{cohort_config.cohort_name}"
    cohort_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(
        cohort_config, orient="index", columns=["value"]
    ).reset_index(names="config").to_csv(
        cohort_folder / "cohort_config.csv", index=False
    )
    return cohort_folder


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
