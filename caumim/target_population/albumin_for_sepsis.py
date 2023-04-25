import pandas as pd
from sklearn.utils import Bunch
import polars as pl
from caumim.constants import (
    COLNAME_TREATMENT_START,
    COLNAME_TREATMENT_STATUS,
    DIR2COHORT,
    DIR2MIMIC,
    COLNAME_INCLUSION_START,
    FILENAME_TARGET_POPULATION,
)
from loguru import logger

from caumim.target_population.utils import (
    create_cohort_folder,
    get_base_population,
    get_cohort_hash,
    roll_inclusion_criteria,
)
from caumim.utils import to_lazyframe

COHORT_CONFIG_ALBUMIN_FOR_SEPSIS = {
    "min_age": 18,
    "min_icu_survival_unit_day": 1,
    "min_los_icu_unit_day": 1,
    "cohort_name": "albumin_for_sepsis",
    "save_cohort": True,
}


def get_population(cohort_config):
    cohort_folder = create_cohort_folder(cohort_config)
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
    crystalloids_inputs = input_events.filter(
        pl.col("itemid").is_in(crystalloids_itemids)
    ).join(icu_stays.select(["stay_id", "intime"]), on="stay_id", how="inner")
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
    # Consider only first day crystalloids
    inclusion_event = first_crystalloids.loc[
        first_crystalloids["delta_crystalloids_icu_intime"].dt.days == 0
    ]
    # 2 - Then define different inclusion criteria, applied at the statistical unit
    # level: here it is the **stay level**.
    #
    # First ICU stay of patients older than 18 years old, with at least 1 day of
    # ICU survival and 1 day of ICU.
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
    # Run successively the inclusion criteria
    target_population = roll_inclusion_criteria(
        inclusion_criteria, cohort_folder=cohort_folder
    )
    # 3 - Define the treatment events
    albumin_itemids = [
        # 220861, #"Albumin (Human) 20% Not in use
        220862,  # Albumin 25%,Albumin 25%
        # 220863, #Albumin (Human) Not in use
        220864,  # Albumin 5%
    ]
    albumin = input_events.filter(pl.col("itemid").is_in(albumin_itemids))
    combined_albumin_for_target_population = to_lazyframe(
        target_population[
            ["stay_id", "icu_intime", COLNAME_INCLUSION_START]
        ].drop_duplicates()
    ).join(albumin, on="stay_id", how="inner")

    # First albumin
    first_albumin = (
        combined_albumin_for_target_population.sort("starttime")
        .groupby("stay_id")
        .agg(
            [
                pl.first("starttime"),
                pl.first("icu_intime"),
                pl.first(COLNAME_INCLUSION_START),
            ]
        )
        .collect()
        .to_pandas()
        .rename(columns={"starttime": COLNAME_TREATMENT_START})
    )
    # Consider only first day albumin
    first_albumin["delta_albumin_icu_intime"] = (
        first_albumin[COLNAME_TREATMENT_START] - first_albumin["icu_intime"]
    )
    first_albumin_in24h = first_albumin.loc[
        first_albumin["delta_albumin_icu_intime"].dt.days == 0
    ]
    first_albumin_in24h = first_albumin_in24h.loc[
        first_albumin_in24h[COLNAME_TREATMENT_START]
        > first_albumin_in24h[COLNAME_INCLUSION_START]
    ]
    first_albumin_in24h
    # %%
    # 4- Define treatment and control population:
    target_trial_population = target_population.merge(
        first_albumin_in24h[
            ["stay_id", COLNAME_TREATMENT_START]
        ].drop_duplicates(),
        on="stay_id",
        how="left",
    )

    target_trial_population[COLNAME_TREATMENT_STATUS] = target_trial_population[
        COLNAME_TREATMENT_START
    ].notnull()
    if cohort_config.save_cohort:
        target_trial_population.to_parquet(
            cohort_folder / (FILENAME_TARGET_POPULATION)
        )
    logger.info(
        f"Number of treated patients: {target_trial_population[COLNAME_TREATMENT_STATUS].sum()}",
    )
    logger.info(
        f"Number of control patients: {(1 - target_trial_population[COLNAME_TREATMENT_STATUS]).sum()}",
    )


if __name__ == "__main__":
    get_population(cohort_config=Bunch(**COHORT_CONFIG_ALBUMIN_FOR_SEPSIS))
