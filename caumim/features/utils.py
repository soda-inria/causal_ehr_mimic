from typing import List
import polars as pl

from caumim.constants import (
    COLNAME_CODE,
    COLNAME_DOMAIN,
    COLNAME_END,
    COLNAME_LABEL,
    COLNAME_START,
    COLNAME_VALUE,
    COLNAMES_EVENTS,
    COLNAME_HADM_ID,
    COLNAME_ICUSTAY_ID,
    COLNAME_INCLUSION_START,
    COLNAME_PATIENT_ID,
)
from caumim.utils import to_lazyframe


def restrict_event_to_observation_period(
    target_trial_population: pl.LazyFrame,
    event: pl.LazyFrame,
):
    """
    Restrict events to the observation period
    of a given cohort.
    """
    if COLNAME_ICUSTAY_ID in event.columns:
        join_col = COLNAME_ICUSTAY_ID
    elif COLNAME_HADM_ID in event.columns:
        join_col = COLNAME_HADM_ID
    elif COLNAME_PATIENT_ID in event.columns:
        join_col = COLNAME_PATIENT_ID
    else:
        raise ValueError(
            f"Event does not contain any of the following columns: "
            f"{COLNAME_ICUSTAY_ID}, {COLNAME_HADM_ID}, {COLNAME_PATIENT_ID}"
        )
    # adding missing keys to the event table
    missing_keys = list(
        set([COLNAME_PATIENT_ID, COLNAME_HADM_ID, COLNAME_ICUSTAY_ID])
        .difference(event.columns)
        .difference(join_col)
    )
    event_for_target_population = event.join(
        to_lazyframe(target_trial_population).select(
            [join_col, *missing_keys, COLNAME_INCLUSION_START]
        ),
        on=join_col,
        how="inner",
    )
    # By design I am forcing the FOLLOWUP_START to be the inclusion start. This should
    # avoid making time-zero bias errors but might not be super practical.
    event_in_observation_period = event_for_target_population.filter(
        pl.col("starttime") <= pl.col(COLNAME_INCLUSION_START)
    )
    # force value to be float64
    event_in_observation_period = event_in_observation_period.with_columns(
        pl.col(COLNAME_VALUE).cast(pl.Float64)
    )
    return event_in_observation_period.select(COLNAMES_EVENTS)


def get_measurement_from_mimic_concept_tables(
    measurement_concepts: List[str],
    measurement_table: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Extract non null measurement from mimic measurement concepts tables (eg.
    bg.sql, or vitalsign).
    """
    if COLNAME_ICUSTAY_ID in measurement_table.columns:
        stay_level = COLNAME_ICUSTAY_ID
    elif COLNAME_HADM_ID in measurement_table.columns:
        stay_level = COLNAME_HADM_ID
    else:
        raise ValueError(
            "measurement_table does not contain any of the following columns: "
            f"{COLNAME_ICUSTAY_ID}, {COLNAME_HADM_ID}"
        )
    stay_keys = [stay_level]
    if COLNAME_PATIENT_ID in measurement_table.columns:
        stay_keys.append(COLNAME_PATIENT_ID)

    measurement_event_list = []
    for measurement_col in measurement_concepts:
        measurement_event_ = (
            measurement_table.select(
                [
                    *stay_keys,
                    pl.col("charttime").alias(COLNAME_START),
                    pl.col(measurement_col).alias(COLNAME_VALUE),
                ]
            )
            .filter(pl.col(COLNAME_VALUE).is_not_null())
            .with_columns(
                [
                    pl.lit("measurement").alias(COLNAME_DOMAIN),
                    pl.lit(measurement_col).alias(COLNAME_CODE),
                    pl.lit(measurement_col).alias(COLNAME_LABEL),
                ]
            )
        )
        # add separetely if needed
        if COLNAME_END not in measurement_event_.columns:
            measurement_event_ = measurement_event_.with_columns(
                [pl.col(COLNAME_START).alias(COLNAME_END)]
            )
        measurement_event_list.append(measurement_event_)
    return pl.concat(measurement_event_list)
