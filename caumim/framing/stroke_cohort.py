from pathlib import Path
from mimproc.experiments.utils import (
    read_icd_procedures,
    read_icu_procedures,
    read_icd_diagnoses,
)

from mimproc.data.events_transformers import make_measure_sequences
from typing import List, Union
from caumim.constants import (
    COLNAME_DOMAIN,
    COLNAME_CODE,
    COLNAME_ICUSTAY_ID,
    COLNAME_ICU_INTIME,
    COLNAME_ICU_OUTTIME,
    COLNAME_START,
)

from caumim.constants import (
    COL_INTERVENTION_TS,
    DIR2RESOURCES,
    EXPERT_VARIABLES,
    DIR2MIMIC,
    TOP_50_VARIABLES,
    BASELINE_VARIABLES,
)

import logging
import os
import re
import numpy as np
import pandas as pd
import polars as pl

from caumim.framing.utils import (
    get_flat_information,
    add_inhospital_mortality_to_icustays,
    add_inunit_mortality_to_icustays,
    print_selection,
    read_icu_procedures,
)

"""
Requirements: This cohort uses the parquet dataframes extracted with the cli
bin.cli.duckdb2parquet. 

Input: ehr_dir : Directory containing a preprocessed extraction of mimic in the
events format. Ouput: It generates a task-specific cohort similar to the one of
mimic3benchmark for ATE estimation in the case of patient with stroke related
diagnosis with two targets: remaining_stay_los (defined as the remaining time of
hospitalisation since icu admission), head_imagery and mortality after 48h (main
outcome). Description: 
    - inclusion Criteria: 
        - patient aged above 18 (already done in mimic3benchmark preproc)
        - patient aged above 18 (already done in mimic3benchmark preproc)
        - more than n_hours of in-icu hospital stay data, 
        - a diagnosis related to stroke, TIA
        
Launch as : `python mimproc/exp/scripts/create_events_cohort_stroke_imagery.py
--ehr_dir data/interim/ehr_cohort --output_dir data/clean/`
"""


# TODO: split into cohort creation and variable extractions
def create_cohort(
    code_whitelist: Union[List[str], str],
    n_hours=24,
    eps_truncation=1,
    eps=1e-6,
    random_state=42,
    keep_only_primo_admission: bool = False,
    time_to_treatment: int = None,
):
    """_summary_

    Args:
        code_whitelist (Union[List[str], str]): _description_
        n_hours (int, optional): _description_. Defaults to 24.
        eps_truncation (int, optional): _description_. Defaults to 1.
        eps (_type_, optional): _description_. Defaults to 1e-6.
        random_state (int, optional): _description_. Defaults to 42.
        keep_only_primo_admission (bool, optional): Remove patients where ICU INTIME != HADM ADMITEDTIME. Defaults to False.
        time_to_treatment (int, optional):  Number of hours after ICU admission where we are searching for the treatment.. Defaults to None.
    """
    if type(code_whitelist) == str:
        if code_whitelist == "expert_variables":
            code_whitelist = EXPERT_VARIABLES
        elif code_whitelist == "top_50_measured":
            code_whitelist = TOP_50_VARIABLES
        elif code_whitelist == "baseline":
            code_whitelist = BASELINE_VARIABLES
    included_stays = get_stroke_imagery_stay(
        dir2mimic=DIR2MIMIC,
        n_hours=n_hours,
        eps=eps,
        keep_only_primo_admission=keep_only_primo_admission,
        time_to_treatment=time_to_treatment,
    )
    valid_intervention_delta = included_stays.loc[
        ~included_stays["delta_icu_intervention"].isna(),
        "delta_icu_intervention",
    ]
    np.random.seed(random_state)
    included_stays["truncation_time"] = included_stays[
        "delta_icu_intervention"
    ].map(
        lambda x: np.random.choice(
            valid_intervention_delta,
        )
        if np.isnan(x)
        else x
    )
    included_stays["truncation_time"] = (
        included_stays["truncation_time"] - eps_truncation
    ).astype(float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract relevant events
    logging.info("Reading ehr events")
    events = pd.read_parquet(os.path.join(ehr_dir, "events.parquet")).merge(
        included_stays[
            [
                COLNAME_ICUSTAY_ID,
                COLNAME_ICU_INTIME,
                COLNAME_ICU_OUTTIME,
                "truncation_time",
            ]
        ],
        on=COLNAME_ICUSTAY_ID,
        how="inner",
    )
    measures = events.query(f"{COLNAME_DOMAIN} == 'measure'").merge(
        pd.DataFrame({COLNAME_CODE: code_whitelist}),
        on=COLNAME_CODE,
        how="inner",
    )
    logging.info("{} events of interest".format(measures.shape[0]))
    # keep events only after intime and before intervention or end of icu stay (troncated to n_hours)
    measures = measures.loc[
        (
            (
                pd.to_datetime(measures[COLNAME_START])
                - pd.to_datetime(measures[COLNAME_ICU_INTIME])
            ).astype("timedelta64[s]")
            >= -eps
        )
        & (
            (
                pd.to_datetime(measures[COLNAME_START])
                - pd.to_datetime(measures[COLNAME_ICU_INTIME])
            ).astype("timedelta64[s]")
            <= eps + measures["truncation_time"] * 3600
        )
        & (
            (
                pd.to_datetime(measures[COLNAME_START])
                - pd.to_datetime(measures[COLNAME_ICU_INTIME])
            ).astype("timedelta64[s]")
            <= eps + n_hours * 3600
        ),
        :,
    ]
    logging.info(
        "{} events of interest after intervention truncation".format(
            measures.shape[0]
        )
    )
    # Discard stays without any events before intervention
    included_stays = included_stays.merge(
        measures[COLNAME_ICUSTAY_ID].drop_duplicates(),
        on=COLNAME_ICUSTAY_ID,
        how="inner",
    )
    print_selection(included_stays, "Final intervention cohort")
    included_stays.to_csv(os.path.join(output_dir, "statics.csv"), index=False)
    measures_of_interest = measures.drop(["intime", "truncation_time"], axis=1)
    measures_of_interest.to_csv(
        os.path.join(output_dir, "events.csv"), index=False
    )
    sequences, mask = make_measure_sequences(
        included_stays,
        measures_of_interest,
        output_dir,
        code_whitelist=code_whitelist,
    )


def get_stroke_imagery_stay(
    dir2mimic: Path = DIR2MIMIC,
    n_hours=24,
    eps=1e-6,
    keep_only_primo_admission: bool = False,
    time_to_treatment: int = None,
):
    print("Building included stays statics and labels")
    flat_information = get_flat_information()

    # population inclusion on diagnoses
    cerebrovascular_regex = ["^" + str(x) for x in range(430, 439)]
    included_diagnoses = pl.read_parquet(dir2mimic).to_pandas()
    included_diagnoses = included_diagnoses.loc[
        included_diagnoses["ICD9_CODE"].map(
            lambda x: re.search("|".join(cerebrovascular_regex), x) is not None
        ),
        ["ICUSTAY_ID"],
    ].rename(columns={"ICUSTAY_ID": COLNAME_ICUSTAY_ID})
    included_stays = flat_information.merge(
        included_diagnoses.drop_duplicates(), on=COLNAME_ICUSTAY_ID, how="inner"
    )
    print_selection(included_stays, "Targeted on diagnoses")
    # Filter on length of stay
    included_stays["icu_los"] = included_stays["los"] * 24
    included_stays = included_stays.loc[
        included_stays["icu_los"] > n_hours + eps, :
    ]
    print_selection(included_stays, "Avoiding right censoring")
    # add mortality
    included_stays = add_inhospital_mortality_to_icustays(included_stays)
    included_stays = add_inunit_mortality_to_icustays(included_stays)
    if keep_only_primo_admission:
        # we consider primo_admitted if delta is < 10min
        delta_primo_admission = 1 / 6
        included_stays = included_stays.loc[
            included_stays["delta_icu_admission_hours"] <= delta_primo_admission
        ]
    print_selection(included_stays, "Remove non primo-admissions")

    # creating intervention labels
    imagery_procedures_codes = pd.read_csv(
        os.path.join(DIR2RESOURCES, "head_imagery_icd9.csv")
    )[["ICD9_CODE"]]

    icd9_procedures = read_icd_procedures(dir2mimic)
    icu_procedures = read_icu_procedures(dir2mimic)
    intervened_stayids = icd9_procedures.merge(
        imagery_procedures_codes, on="ICD9_CODE", how="inner"
    )[["ICUSTAY_ID"]]

    icu_procedures = (
        icu_procedures.loc[
            icu_procedures["ITEMID"].isin(
                [
                    223253,  # MRI
                    221214,  # CT Scan
                    # 221217,  # Ultrasound to be included for ethiology!
                    # 221216,  # X-ray : to be included for ethiology ?
                    # 225402 # EKG : to be included for ethiology ?
                ]
            )
            & (icu_procedures["CANCELREASON"] == 0),
            :,
        ]
        .rename(columns={"STARTTIME": COL_INTERVENTION_TS})
        .sort_values(COL_INTERVENTION_TS)
    )  # cancel reason are tagged "rewritten" (not sure of the meaning)),

    # we only use the in-icu ct-scan and MRI information, not the incomplete in-hospital billing codes

    interventions = (
        icu_procedures.groupby("ICUSTAY_ID")
        .agg(**{COL_INTERVENTION_TS: pd.NamedAgg(COL_INTERVENTION_TS, "first")})
        .reset_index()
    ).rename(columns={"ICUSTAY_ID": COLNAME_ICUSTAY_ID})
    # only in-icu intervention are labelled positevily
    included_stays = included_stays.merge(
        interventions, on=COLNAME_ICUSTAY_ID, how="left"
    )
    if time_to_treatment is not None:
        mask_time_to_treatment = (
            pd.to_datetime(included_stays[COL_INTERVENTION_TS])
            - pd.to_datetime(included_stays[COLNAME_ICU_INTIME])
        ).astype("timedelta64[h]") <= time_to_treatment
    else:
        mask_time_to_treatment = True
    included_stays["intervention"] = (
        (
            pd.to_datetime(included_stays[COL_INTERVENTION_TS])
            >= pd.to_datetime(included_stays[COLNAME_ICU_INTIME])
        )
        & (
            pd.to_datetime(included_stays[COL_INTERVENTION_TS])
            <= pd.to_datetime(included_stays[COLNAME_ICU_OUTTIME])
        )
        & mask_time_to_treatment
    ).astype(int)

    # building interesting deltas
    included_stays["delta_icu_intervention"] = (
        pd.to_datetime(included_stays[COL_INTERVENTION_TS])
        - pd.to_datetime(included_stays["intime"])
    ).astype("timedelta64[s]") / 3600
    included_stays["delta_icu_hosp_admissions"] = (
        pd.to_datetime(included_stays["intime"])
        - pd.to_datetime(included_stays["admittime"])
    ).astype("timedelta64[s]") / 3600
    included_stays["delta_icu_death"] = (
        pd.to_datetime(included_stays["dod"])
        - pd.to_datetime(included_stays["intime"])
    ).astype("timedelta64[s]") / 3600
    # remaining los (proxy of patient severity, unused for now)
    included_stays["los_since_icu_intime"] = (
        pd.to_datetime(included_stays["dischtime"])
        - pd.to_datetime(included_stays["intime"])
    ).astype("timedelta64[s]") / 3600
    included_stays["hosp_los"] = (
        pd.to_datetime(included_stays["dischtime"])
        - pd.to_datetime(included_stays["admittime"])
    ).astype("timedelta64[s]") / 3600
    included_stays.columns = [col.lower() for col in included_stays.columns]

    # force to datetime (conversion does no loose any death)
    mask_not_nan = ~included_stays["deathtime"].isna()
    included_stays["deathtime"] = pd.to_datetime(included_stays["deathtime"])
    nb_lost_deathtime = (
        mask_not_nan.sum() - (~included_stays["deathtime"].isna()).sum()
    )
    assert (
        nb_lost_deathtime == 0
    ), "{} deathtimes have been lost in conversion, look at included_stays before conversion".format(
        nb_lost_deathtime
    )
    included_stays["edregtime"] = pd.to_datetime(included_stays["edregtime"])
    included_stays["edouttime"] = pd.to_datetime(included_stays["edouttime"])
    included_stays["dod_ssn"] = pd.to_datetime(included_stays["dod_ssn"])
    included_stays["dod"] = pd.to_datetime(included_stays["dod"])
    included_stays["dod_hosp"] = pd.to_datetime(included_stays["dod_hosp"])
    included_stays["dob"] = pd.to_datetime(included_stays["dob"])
    included_stays["language"] = included_stays["language"].astype(str)
    included_stays["marital_status"] = included_stays["marital_status"].astype(
        str
    )
    nb_intervened = np.sum(included_stays["intervention"])
    nb_dead_hosp = np.sum(1 - included_stays["dod_hosp"].isna())
    N = included_stays.shape[0]
    print(f"Intervention ratio {round(nb_intervened / N, 4)}")
    print(f"Hospital mortality ratio {round(nb_dead_hosp / N, 4)}")

    return included_stays
