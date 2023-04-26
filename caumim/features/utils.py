from typing import List
import polars as pl

from caumim.constants import (
    COLNAME_CODE,
    COLNAME_DOMAIN,
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_END,
    COLNAME_INSURANCE_MEDICARE,
    COLNAME_LABEL,
    COLNAME_START,
    COLNAME_VALUE,
    COLNAMES_EVENTS,
    COLNAME_HADM_ID,
    COLNAME_ICUSTAY_ID,
    COLNAME_INCLUSION_START,
    COLNAME_PATIENT_ID,
    DIR2MIMIC,
    DIR2RESOURCES,
)
from caumim.utils import to_lazyframe, to_polars


def get_antibiotics_event_from_atc4(atc4_codes) -> pl.LazyFrame:
    drugs = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.prescriptions/*")
    ndc_map = pl.scan_parquet(DIR2RESOURCES / "ontology" / "ndc_map")
    # Using the NDC map to get the ATC4 code, NDC of 10 digits should be padded with
    # a leading 0
    atb_ndc_map = (
        ndc_map.filter(pl.col("atc4").is_in(atc4_codes))
        .select(["ndc", "atc4", "atc4_name", "in_name"])
        .with_columns(
            (pl.lit("0") + pl.col("ndc").str.replace_all("-", "")).alias("ndc")
        )
    )
    antibiotics_of_interest = drugs.join(atb_ndc_map, on="ndc", how="inner")
    # %%
    antibiotics_event = antibiotics_of_interest.with_columns(
        [
            pl.col("stoptime").alias(COLNAME_END),
            pl.lit("drug").alias(COLNAME_DOMAIN),
            pl.col("atc4").alias(COLNAME_CODE),
            pl.col("atc4_name").alias(COLNAME_LABEL),
            pl.lit(1).alias(COLNAME_VALUE),
        ]
    )
    return antibiotics_event


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


def feature_emergency_at_admission(target_population: pl.DataFrame):
    emergency_admission = (
        pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.admissions/*")
        .filter(pl.col("admission_type").is_in(["DIRECT EMER.", "EW EMER."]))
        .with_columns(
            pl.lit(1).alias(COLNAME_EMERGENCY_ADMISSION),
        )
        .select([COLNAME_HADM_ID, COLNAME_EMERGENCY_ADMISSION])
    )
    target_population_w_emergency = (
        to_polars(target_population)
        .join(emergency_admission, on=COLNAME_HADM_ID, how="left")
        .with_columns(pl.col(COLNAME_EMERGENCY_ADMISSION).fill_null(0))
    )
    return target_population_w_emergency


def feature_insurance_medicare(target_population: pl.DataFrame):
    insurance = pl.read_parquet(DIR2MIMIC / "mimiciv_hosp.admissions/*").select(
        [COLNAME_HADM_ID, "insurance"]
    )

    target_population_w_insurance = target_population.join(
        insurance, on=COLNAME_HADM_ID, how="left"
    ).with_columns(
        pl.when(pl.col("insurance").is_in(["Medicare", "Medicaid"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias(COLNAME_INSURANCE_MEDICARE)
    )
    return target_population_w_insurance
