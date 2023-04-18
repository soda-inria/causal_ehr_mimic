from datetime import datetime
import duckdb
import polars as pl

from tqdm import tqdm
from caumim.constants import DIR2MIMIC
import logging


DERIVED_TABLE_NAMES = [
    "icustay_times",
    "icustay_hourly",
    "weight_durations",
    "urine_output",
    "kdigo_uo",
    "age",
    "icustay_detail",
    "bg",
    "blood_differential",
    "cardiac_marker",
    "chemistry",
    "coagulation",
    "complete_blood_count",
    "creatinine_baseline",
    "enzyme",
    "gcs",
    "height",
    "icp",
    "inflammation",
    "oxygen_delivery",
    "rhythm",
    "urine_output_rate",
    "ventilator_setting",
    "vitalsign",
    "charlson",
    "antibiotic",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "milrinone",
    "neuroblock",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "crrt",
    "invasive_line",
    "rrt",
    "ventilation",
    "first_day_bg",
    "first_day_bg_art",
    "first_day_gcs",
    "first_day_height",
    "first_day_lab",
    "first_day_rrt",
    "first_day_urine_output",
    "first_day_vitalsign",
    "first_day_weight",
    "kdigo_creatinine",
    "meld",
    "apsiii",
    "lods",
    "oasis",
    "sapsii",
    "sirs",
    "sofa",
    "suspicion_of_infection",
    "kdigo_stages",
    "first_day_sofa",
    "sepsis3",
    "vasoactive_agent",
    "norepinephrine_equivalent_dose",
]


def convert_mimic_from_duckdb_to_parquet(path2duckdb: str):
    t0 = datetime.now()
    con = duckdb.connect(database=path2duckdb)
    all_tables = con.execute(
        "SELECT * from information_schema.tables"
    ).fetch_df()
    all_tables = (
        all_tables["table_schema"] + "." + all_tables["table_name"]
    ).values
    for table_name in all_tables:
        path2table = DIR2MIMIC / f"{table_name}"
        logging.info(f"Converting to parquet at {path2table}")
        con.execute(
            f"COPY {table_name} TO '{str(path2table)}' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);"
        )
    t_finished = datetime.now()
    logging.info(f"Finished parquet conversion in {t_finished - t0}")


def convert_mimic_derived_from_postgresql_to_parquet(psql_con: str):
    for mimic_derived_table_name in tqdm(DERIVED_TABLE_NAMES):
        query = f"SELECT * FROM mimiciv_derived.{mimic_derived_table_name}"
        df = pl.read_sql(query, psql_con)
        df.write_parquet(
            DIR2MIMIC / f"mimiciv_derived.{mimic_derived_table_name}"
        )
    return
