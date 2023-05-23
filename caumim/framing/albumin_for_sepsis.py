import pickle
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.utils import Bunch
import polars as pl
from caumim.constants import (
    COLNAME_MORTALITY_28D,
    COLNAME_MORTALITY_90D,
    COLNAME_INTERVENTION_START,
    COLNAME_INTERVENTION_STATUS,
    COLNAME_PATIENT_ID,
    DIR2COHORT,
    DIR2MIMIC,
    COLNAME_INCLUSION_START,
    FILENAME_INCLUSION_CRITERIA,
    FILENAME_TARGET_POPULATION,
)
from loguru import logger

from caumim.framing.utils import (
    create_cohort_folder,
    get_base_population,
    get_cohort_hash,
    roll_inclusion_criteria,
)
from caumim.utils import to_lazyframe

"""
This script defines the cohort of patients that will be used for the albumin. I
timplements the framing of the question by building: Population, Intervention,
Control and Outcome elements as well as the Time of followup.
"""

observation_window_in_day = 1
COHORT_CONFIG_ALBUMIN_FOR_SEPSIS = Bunch(
    **{
        "min_age": 18,
        "min_icu_survival_unit_day": observation_window_in_day,  # the patient should survive at least one day.
        "min_los_icu_unit_day": observation_window_in_day,  # the patient should stay in ICU at least one day.
        "treatment_observation_window_unit_day": observation_window_in_day,  # the treatment should happen during the first day.
        "cohort_name": "albumin_for_sepsis",
        "save_cohort": True,
    }
)


def get_population(cohort_config) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    This function defines the population of interest for the albumin for sepsis.
    It returns static information with treatment status and important timestamps such as:
    COLNAME_INCLUSION_START, COLNAME_INTERVENTION_START and outcomes.
    """
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
    # Consider only crystalloids before max_los_before_treatment
    crystralloids_first_24h = first_crystalloids.loc[
        (
            first_crystalloids[
                "delta_crystalloids_icu_intime"
            ].dt.total_seconds()
            <= (cohort_config.treatment_observation_window_unit_day * 24 * 3600)
        )
        & (
            first_crystalloids[
                "delta_crystalloids_icu_intime"
            ].dt.total_seconds()
            >= 0
        )
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
    observation_window_in_hour_str = str(
        int(24 * cohort_config.treatment_observation_window_unit_day)
    )
    inclusion_criteria = {
        f"Aged over 18, ICU lOS >= {cohort_config.min_los_icu_unit_day}": base_population,
        "Sepsis patients": sepsis3_stays,
        f"inclusion_event": crystralloids_first_24h,
    }
    # Run successively the inclusion criteria
    target_population, inclusion_ids = roll_inclusion_criteria(
        inclusion_criteria
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
        .rename(columns={"starttime": COLNAME_INTERVENTION_START})
    )
    # Consider only first day albumin
    first_albumin["delta_albumin_icu_intime"] = (
        first_albumin[COLNAME_INTERVENTION_START] - first_albumin["icu_intime"]
    )
    first_albumin_in24h = first_albumin.loc[
        (
            (
                first_albumin["delta_albumin_icu_intime"].dt.total_seconds()
                <= (
                    cohort_config.treatment_observation_window_unit_day
                    * 24
                    * 3600
                )
            )
            & (
                first_albumin["delta_albumin_icu_intime"].dt.total_seconds()
                >= 0
            )
        )
    ]
    first_albumin_in24h = first_albumin_in24h.loc[
        first_albumin_in24h[COLNAME_INTERVENTION_START]
        > first_albumin_in24h[COLNAME_INCLUSION_START]
    ]

    # 4- Define treatment and control population:
    target_trial_population = target_population.merge(
        first_albumin_in24h[
            ["stay_id", COLNAME_INTERVENTION_START]
        ].drop_duplicates(),
        on="stay_id",
        how="left",
    )

    target_trial_population[COLNAME_INTERVENTION_STATUS] = (
        target_trial_population[COLNAME_INTERVENTION_START]
        .notnull()
        .astype(int)
    )
    # target_trial_population[COLNAME_FOLLOWUP_START] = target_trial_population[
    #     COLNAME_INCLUSION_START
    # ]
    # # forcing followup to be either inclusion or treatment start.
    # # It introduces a blan
    # mask_treated = target_trial_population[COLNAME_INTERVENTION_STATUS] == 1
    # target_trial_population.loc[
    #     mask_treated,
    #     COLNAME_FOLLOWUP_START,
    # ] = target_trial_population.loc[mask_treated, COLNAME_INTERVENTION_START]

    # 5 - Define outcomes
    # 28-days and 90-days mortality
    mask_dod = target_trial_population["dod"].notnull()
    days_to_death = (
        target_trial_population["dod"]
        - target_trial_population[COLNAME_INCLUSION_START]
    ).dt.days

    target_trial_population[COLNAME_MORTALITY_28D] = (
        mask_dod & (days_to_death <= 28)
    ).astype(int)
    target_trial_population[COLNAME_MORTALITY_90D] = (
        mask_dod & (days_to_death <= 90)
    ).astype(int)

    col_name_outcomes = [COLNAME_MORTALITY_28D, COLNAME_MORTALITY_90D]
    # 6 - Save the cohort
    for outcome in col_name_outcomes:
        logger.info(
            f"Outcome `{outcome}` prevalence: {100 * target_trial_population[outcome].mean():.2f}%"
        )
    logger.info(
        f"Number of treated patients: {target_trial_population[COLNAME_INTERVENTION_STATUS].sum()}",
    )
    logger.info(
        f"Number of control patients: {(1 - target_trial_population[COLNAME_INTERVENTION_STATUS]).sum()}",
    )
    if cohort_config.save_cohort:
        target_trial_population.to_parquet(
            cohort_folder / (FILENAME_TARGET_POPULATION)
        )
        logger.info(
            f"Saved cohort at {cohort_folder / (FILENAME_TARGET_POPULATION)}"
        )

    # create inclusion criteria dictionnary
    inclusion_ids[
        f"Crystalloids in first {observation_window_in_hour_str}h"
    ] = inclusion_ids[f"inclusion_event"]
    inclusion_ids.pop("inclusion_event")
    inclusion_ids[f"Albumin in first {observation_window_in_hour_str}h"] = (
        target_trial_population.loc[
            target_trial_population[COLNAME_INTERVENTION_STATUS] == 1,
            COLNAME_PATIENT_ID,
        ]
        .unique()
        .tolist()
    )
    pickle.dump(
        inclusion_ids,
        open(str(cohort_folder / FILENAME_INCLUSION_CRITERIA), "wb"),
    )
    return target_population, inclusion_criteria


if __name__ == "__main__":
    get_population(cohort_config=COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
