from dataclasses import dataclass
from typing import List, Tuple
import polars as pl
from sklearn.utils import Bunch

from caumim.constants import (
    COLNAME_CODE,
    COLNAME_DOMAIN,
    COLNAME_END,
    COLNAME_HADM_ID,
    COLNAME_PATIENT_ID,
    COLNAME_START,
    COLNAMES_EVENTS,
    COLNAME_ICUSTAY_ID,
    COLNAME_LABEL,
    COLNAME_VALUE,
    DIR2MIMIC,
    DIR2RESOURCES,
)
from caumim.variables.utils import (
    get_antibiotics_event_from_atc4,
    get_antibiotics_event_from_drug_name,
    get_measurement_from_mimic_concept_tables,
    restrict_event_to_observation_period,
)
from caumim.utils import to_lazyframe, to_polars

from joblib import Memory

location = "./cachedir"
memory = Memory(location, verbose=0)


@dataclass
class VariableTypes:
    binary_features: List[str] = None
    categorical_features: List[str] = None
    numerical_features: List[str] = None


@memory.cache()
def get_comorbidity(
    target_trial_population: pl.DataFrame,
) -> Tuple[pl.DataFrame, VariableTypes]:
    """These comorbidities are computed [from icd codes](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/comorbidity/charlson.sql)"""
    comorbidities = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.charlson/")
    index_cols = [
        COLNAME_PATIENT_ID,
        COLNAME_HADM_ID,
        COLNAME_ICUSTAY_ID,
        "dischtime",
    ]
    comorbidities_filtered = comorbidities.join(
        to_lazyframe(target_trial_population).select(index_cols),
        on=[COLNAME_PATIENT_ID, COLNAME_HADM_ID],
        how="inner",
    )
    # only keep non null diagnosis
    comorbidities_event = (
        comorbidities_filtered.melt(
            id_vars=index_cols,
            value_vars=[
                "myocardial_infarct",
                "congestive_heart_failure",
                "peripheral_vascular_disease",
                "cerebrovascular_disease",
                "dementia",
                "chronic_pulmonary_disease",
                "rheumatic_disease",
                "peptic_ulcer_disease",
                "mild_liver_disease",
                "diabetes_without_cc",
                "diabetes_with_cc",
                "paraplegia",
                "renal_disease",
                "malignant_cancer",
                "severe_liver_disease",
                "metastatic_solid_tumor",
                "aids",
                "charlson_comorbidity_index",
            ],
            variable_name=COLNAME_CODE,
            value_name=COLNAME_VALUE,
        )
        .filter(
            (pl.col(COLNAME_VALUE) != 0)
            | (pl.col(COLNAME_CODE) == "charlson_comorbidity_index")
        )
        .rename({"dischtime": COLNAME_START})
        .collect()
        .with_columns(
            pl.lit("condition").alias(COLNAME_DOMAIN),
            pl.lit("charlson_comorbidity_index").alias(COLNAME_LABEL),
            pl.col(COLNAME_START).alias(COLNAME_END),
            pl.col(COLNAME_VALUE).cast(pl.Float64).alias(COLNAME_VALUE),
        )
    )
    # Add feature types
    feature_types = VariableTypes(
        binary_features=comorbidities_event[COLNAME_CODE].unique().to_list(),
    )
    return comorbidities_event, feature_types


@memory.cache()
def get_event_covariates_albumin_zhou(
    target_trial_population: pl.DataFrame,
) -> Tuple[pl.DataFrame, VariableTypes]:
    """Get the baseline variables from the [Zhou et al., 2021](https://link.springer.com/article/10.1186/s13613-021-00830-8) paper.

    Returns:
        pl.DataFrame: Events in the observation period (before followup), for the target trial population.
    """
    # Describe baseline caracteristics
    event_list = []
    # 1 - antibiotics
    # antibiotic_atc4 = {
    #     "Carbapenems": ["J01DH"],
    #     "Glycopeptide": ["J01XA"],
    #     "Beta-lactams": [
    #         "J01CA",  # penicillins
    #         "J01CE"  # "Beta-lactamase-sensitive penicillins",
    #         "J01CF",  # "Beta-lactamase-resistant penicillins",
    #         "J01CG",  # "Beta-lactamase inihibitors",
    #         "J01CR",  # "Combinations of penicillins, incl. beta-lactamase inhibitors",
    #     ],
    #     "Aminoglycosides": [
    #         "J01GA",  # Aminoglycosides - Streptomycins
    #         "J01GB",  # "Aminoglycosides - Other aminoglycosides"
    #     ],
    # }
    antibiotic_str = {
        "Carbapenems": ["meropenem"],
        "Glycopeptide": ["vancomycin"],
        "Beta-lactams": ["ceftriaxone", "cefotaxime", "cefepime"],
        "Aminoglycosides": ["gentamicin", "amikacin"],
    }
    antibiotics_event = get_antibiotics_event_from_drug_name(
        antibiotic_str
    )  # get_antibiotics_event_from_atc4(antibiotic_atc.keys())
    # map to ATC4 or ATC3 depending on the antibiotics
    antibiotics_event_renamed_for_study = antibiotics_event.with_columns(
        [
            pl.when(pl.col(COLNAME_CODE).str.starts_with("J01C"))
            .then("J01C")
            .when(pl.col(COLNAME_CODE).str.starts_with("J01G"))
            .then("J01G")
            .otherwise(pl.col(COLNAME_CODE))
            .alias(COLNAME_CODE),
            pl.when(pl.col(COLNAME_CODE).str.starts_with("J01C"))
            .then("Beta-lactams")
            .when(pl.col(COLNAME_CODE).str.starts_with("J01G"))
            .then("Aminoglycosides")
            .otherwise(pl.col(COLNAME_LABEL))
            .alias(COLNAME_LABEL),
            pl.col("stoptime").dt.cast_time_unit("ns").alias(COLNAME_END),
            pl.col(COLNAME_START).dt.cast_time_unit("ns").alias(COLNAME_START),
        ],
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=antibiotics_event_renamed_for_study,
        )
    )
    # 2 - Measurements
    ### 2.1 - Weight
    weight_event = pl.scan_parquet(
        DIR2MIMIC / "mimiciv_derived.weight_durations/"
    ).with_columns(
        [
            pl.lit("measurement").alias(COLNAME_DOMAIN),
            pl.lit("Weight").alias(COLNAME_CODE),
            pl.lit("Weight").alias(COLNAME_LABEL),
            pl.col("weight").alias("value"),
        ]
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=weight_event,
        )
    )
    ### 2.2 - Vital signs:
    vitalsign = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.vitalsign/")
    # Respiratory Rate, Heart Rate, Mean Arterial Pressure, Temperature, SPO2
    vital_signs_in_study = [
        "resp_rate",
        "heart_rate",
        "mbp",
        "temperature",
        "spo2",
    ]
    vital_signs_event = get_measurement_from_mimic_concept_tables(
        measurement_concepts=vital_signs_in_study, measurement_table=vitalsign
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=vital_signs_event,
        )
    )
    ### 2.3 - Other measurements
    # lactate,
    blood_gaz_list = ["lactate"]
    blood_gaz = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.bg/")
    blood_gaz_event = get_measurement_from_mimic_concept_tables(
        measurement_concepts=blood_gaz_list, measurement_table=blood_gaz
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=blood_gaz_event,
        )
    )
    # urine outputs
    urine_output_list = ["urineoutput"]
    urine_output = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.urine_output/")
    urine_output_event = get_measurement_from_mimic_concept_tables(
        measurement_concepts=urine_output_list, measurement_table=urine_output
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=urine_output_event,
        )
    )
    ### 2.4 - Scores
    # Saps2 mimic_derived.sapsii(computed on the first day, so might be a colider)
    sapsii_event = (
        pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.sapsii/")
        .select(
            [
                COLNAME_PATIENT_ID,
                COLNAME_HADM_ID,
                COLNAME_ICUSTAY_ID,
                COLNAME_START,
                COLNAME_END,
                "sapsii",
            ]
        )
        .with_columns(
            pl.lit("measurement").alias(COLNAME_DOMAIN),
            pl.lit("SAPSII").alias(COLNAME_CODE),
            pl.lit("SAPSII").alias(COLNAME_LABEL),
            pl.col("sapsii").alias("value"),
        )
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=sapsii_event,
        )
    )
    # Sofa, (computed on the first day, so might be a colider)
    first_day_sofa_event = (
        pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.first_day_sofa/")
        .select(
            [COLNAME_PATIENT_ID, COLNAME_HADM_ID, COLNAME_ICUSTAY_ID, "sofa"]
        )
        .with_columns(
            pl.lit("measurement").alias(COLNAME_DOMAIN),
            pl.lit("SOFA").alias(COLNAME_CODE),
            pl.lit("SOFA").alias(COLNAME_LABEL),
            pl.col("sofa").alias("value"),
        )
        .join(
            to_lazyframe(target_trial_population)
            .select(
                pl.col(COLNAME_ICUSTAY_ID),
                pl.col("icu_intime").alias(COLNAME_START),
            )
            .with_columns(
                (pl.duration(days=1) + pl.col(COLNAME_START)).alias(
                    COLNAME_END
                ),
            ),
            on=COLNAME_ICUSTAY_ID,
            how="inner",
        )
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=first_day_sofa_event,
        )
    )

    # AKI mimic_derived.kdigo_stages
    kdigo_stages = pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.kdigo_stages/")
    kdigo_event = get_measurement_from_mimic_concept_tables(
        measurement_concepts=["aki_stage"], measurement_table=kdigo_stages
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=kdigo_event,
        )
    )
    # 3 - procedures: ventilator_setting, rrt
    ventilation_event = pl.scan_parquet(
        DIR2MIMIC / "mimiciv_derived.ventilation/"
    ).with_columns(
        pl.lit("procedure").alias(COLNAME_DOMAIN),
        pl.lit("ventilation").alias(COLNAME_CODE),
        pl.lit("ventilation").alias(COLNAME_LABEL),
        pl.lit(1).alias("value"),
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=ventilation_event,
        )
    )

    rrt_event = pl.scan_parquet(
        DIR2MIMIC / "mimiciv_derived.rrt/"
    ).with_columns(
        pl.lit("procedure").alias(COLNAME_DOMAIN),
        pl.col("charttime").alias(COLNAME_START),
        pl.col("charttime").alias(COLNAME_END),
        pl.lit("RRT").alias(COLNAME_CODE),
        pl.col("dialysis_type").alias(COLNAME_LABEL),
        pl.col("dialysis_active").alias(COLNAME_VALUE),
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=rrt_event,
        )
    )
    # 4 - medications: vasopressors
    vasopressors_str = [
        "dopamine",
        "epinephrine",
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
        "dobutamine",
        "milrinone",
    ]

    vasopressors = pl.scan_parquet(
        DIR2MIMIC / "mimiciv_derived.vasoactive_agent/"
    ).with_columns(
        pl.col("starttime").alias("charttime"),
    )
    vasopressors_event = (
        get_measurement_from_mimic_concept_tables(
            measurement_concepts=vasopressors_str,
            measurement_table=vasopressors,
        )
        .drop("domain")
        .with_columns(
            pl.lit("medication").alias(COLNAME_DOMAIN),
            pl.lit("vasopressors").alias(COLNAME_CODE),
            pl.when(pl.col(COLNAME_VALUE) > 0)
            .then(1)
            .otherwise(0)
            .alias(COLNAME_VALUE),
        )
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=vasopressors_event,
        )
    )
    # 5 - other features:
    # Suspected_infection_blood
    suspicion_of_infection_event = (
        pl.scan_parquet(DIR2MIMIC / "mimiciv_derived.suspicion_of_infection/")
        .filter(
            pl.col("specimen").str.to_lowercase().str.contains("blood")
            & (pl.col("positive_culture") == 1)
        )
        .drop(COLNAME_ICUSTAY_ID)
        .with_columns(
            pl.lit("observation").alias(COLNAME_DOMAIN),
            pl.lit("suspected_infection_blood").alias(COLNAME_CODE),
            pl.lit("suspected_infection_blood").alias(COLNAME_LABEL),
            pl.lit(1).alias("value"),
            pl.col("suspected_infection_time").alias(COLNAME_START),
            pl.col("suspected_infection_time").alias(COLNAME_END),
        )
    )
    event_list.append(
        restrict_event_to_observation_period(
            target_trial_population=target_trial_population,
            event=suspicion_of_infection_event,
        )
    )
    # Get all features together
    event_features = pl.concat(event_list).collect()
    # Restrict to cohort and observation period (before inclusion start)
    event_features = event_features.filter(~event_features.is_duplicated())

    # Add feature types
    feature_types = VariableTypes(
        binary_features=[
            "Glycopeptide",  # J01XA
            "Beta-lactams",  # "J01C",
            "Carbapenems",  # "J01DH",
            "Aminoglycosides",  # "J01G",
            "suspected_infection_blood",
            "RRT",
            "ventilation",
            "vasopressors",
        ],
        categorical_features=["aki_stage"],
        numerical_features=[
            "SOFA",
            "SAPSII",
            "Weight",
            "temperature",
            "mbp",
            "resp_rate",
            "heart_rate",
            "spo2",
            "lactate",
            "urineoutput",
        ],
    )

    return event_features, feature_types
