import pandas as pd
from sklearn.utils import Bunch
import polars as pl
from caumim.constants import DIR2COHORT, DIR2MIMIC, COLNAME_INCLUSION_START

from caumim.target_population.utils import (
    get_base_population,
    get_cohort_hash,
    roll_inclusion_criteria,
)

COHORT_CONFIG_ALBUMIN_FOR_SEPSIS = {
    "min_age": 18,
    "min_icu_survival_unit_day": 1,
    "min_los_icu_unit_day": 1,
    "cohort_name": "albumin_for_sepsis",
}


def get_population(cohort_config):
    cohort_hash = get_cohort_hash(cohort_config)
    cohort_folder = DIR2COHORT / cohort_hash
    cohort_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(cohort_config, orient="index").to_csv(
        cohort_folder / "cohort_config.csv"
    )
    # 1 - Define the inclusion events, ie. the event that defines when a patient
    # enter the cohort.
    # Inclusion event: First administration of crystalloids during the 24 first
    # hours of ICU stay
    input_events = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.inputevents/*")
    icu_stays = pl.scan_parquet(DIR2MIMIC / "mimiciv_icu.icustays/*")
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
        input_events.filter(pl.col("itemid").is_in(crystalloids_itemids))
        .join(
            icu_stays.select(["stay_id", "intime"]), on="stay_id", how="inner"
        )
        .collect()
        .to_pandas()
    )
    first_crystalloids = (
        crystalloids_inputs.sort_values("starttime")
        .groupby("stay_id")
        .first()[["starttime", "intime"]]
        .reset_index()
        .rename(columns={"starttime": COLNAME_INCLUSION_START})
    )
    first_crystalloids["delta_crystalloids_icu_intime"] = (
        first_crystalloids[COLNAME_INCLUSION_START]
        - first_crystalloids["intime"]
    )
    inclusion_event = first_crystalloids.loc[
        first_crystalloids["delta_crystalloids_icu_intime"].dt.days == 0
    ]

    # Then define different inclusion criteria, applied at the statistical unit
    # level: here it is the **stay level**.
    base_population = get_base_population(
        min_age=cohort_config.min_age,
        min_icu_survival_unit_day=cohort_config.min_icu_survival_unit_day,
        min_los_icu_unit_day=cohort_config.min_los_icu_unit_day,
    )
    # sepsis
    sepsis3_stays = pd.read_parquet(DIR2MIMIC / "mimiciv_derived.sepsis3")
    sepsis3_stays = sepsis3_stays.loc[
        sepsis3_stays["sepsis3"] == True, ["stay_id"]
    ]
    inclusion_criteria = {
        "base_population": base_population,
        "sepsis3": sepsis3_stays,
        "inclusion_event": inclusion_event,
    }
    target_population = roll_inclusion_criteria(inclusion_criteria)


if __name__ == "__main__":
    get_population(cohort_config=Bunch(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS))
