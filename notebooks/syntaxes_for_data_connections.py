# %%
from pathlib import Path
import ibis
from ibis import deferred as c
import duckdb

ibis.options.interactive = True
print(ibis.__version__)
print(duckdb.__version__)

from caumim.constants import DIR2MIMIC
from caumim.acquisition.build_concepts import (
    get_concept_query,
    register_sql_functions,
)

"""
This notebook was used to test the different syntaxes for data connections: 
- Duckdb in memory
- 
"""

# %%
dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"
dir2mimic = "/home/mdoutrel/projets/inria/mimiciv-2.2/"
# %% [markdown]
# # Duckdb
# ## Duckdb in memory [partially working]

# Trying to connect to the on-memory database is partially working.

# I can get the head, but not scan the entire files (eg. for counts). However
# this works if I use duckdb alone (see pure duckdb)
# %%"
duckdb_in_memory = ibis.duckdb.connect(database=":memory:")
patients = duckdb_in_memory.read_csv(
    dir2mimic + "hosp/patients.csv.gz", sep="\t"
)
events = duckdb_in_memory.read_csv(
    dir2mimic + "hosp/drgcodes.csv.gz", sep=",", compression="gzip"
)
events.head()
# %%
# buggy
events.limit(60000).count()
# Is it related to a wrong type ? No, it fails even on subject id. Is it related
# to a separator ? Seems so looking at the error msg and the fact that it does
# not fails with a limit statement.

# events["subject_id"].count()
# %%
# also buggy
nb_labs_per_patients = events.group_by("subject_id").count()
nb_labs_per_patients
# %%
# %% [markdown]
# ## Duckdb on-disk database [not working]
# %%
duckdb_on_disk_conn = ibis.duckdb.connect(database=dir2mimic_duckdb)
# %%
patients = duckdb_on_disk_conn.table("patients")
patients.columns
patients.count()
# %% [markdown]
# ## Pure Duckdb [working]
# (not very practical since SQL only)
# %%
import duckdb

# %%
con = duckdb.connect(database=dir2mimic_duckdb)
# %%
patients_ddb = con.execute(
    "SELECT * FROM mimiciv_hosp.patients LIMIT 10"
).fetch_df()
print(patients.columns)
lab_head = con.execute(
    "SELECT * FROM mimiciv_hosp.labevents LIMIT 10"
).fetch_df()
lab_head
# %%
nb_labs_per_patients = con.execute(
    """
    SELECT subject_id, Count(labevent_id) AS Count
FROM
    (
        SELECT DISTINCT subject_id, labevent_id
        FROM mimiciv_hosp.labevents
    ) AS whatever
GROUP BY subject_id
"""
).exe()
nb_labs_per_patients
# %%
events_count = con.execute(
    "SELECT count(*) FROM mimiciv_hosp.drgcodes"
).fetch_df()
events_count
# %%
from caumim.constants import DIR2MIMIC

events_ddb = con.execute("SELECT * FROM mimiciv_icu.icustays").fetch_df()
events_ddb.to_parquet(DIR2MIMIC / "drgcodes.parquet")
# %% [markdown]
# ## Save a parquet from duckdb, then access with Ibis [working]
#
# So it is indeed an issue with separators in the compressed csv that are badly
# interpreted by Ibis but correctly from duckdb %% save with duckdb wo
# collecting the dataframe.
import duckdb
from caumim.constants import DIR2MIMIC

dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"

con = duckdb.connect(database=dir2mimic_duckdb)
all_tables = con.execute("SELECT * from information_schema.tables").fetch_df()
all_tables = (
    all_tables["table_schema"] + "." + all_tables["table_name"]
).values
all_tables
table_name = all_tables[0]
path2table = DIR2MIMIC / f"{table_name}"
# %%
# events_ddb = con.execute("SELECT * FROM mimiciv_hosp.drgcodes").fetch_df()
con.execute(
    f"COPY {table_name} TO '{str(path2table)}' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);"
)
# %%
# read from ibis+duckdb
import ibis

ibis.options.interactive = True

table_name = "labevents"
table_name = "patients"

path2table = DIR2MIMIC / f"{table_name}/*"
duckdb_in_memory = ibis.duckdb.connect(database=":memory:")
events = duckdb_in_memory.read_parquet(source_list=str(path2table))
events.count()
# %%
events.group_by("subject_id").count()


# %% [markdown]
# ## duckdb reading directly from the csv.gz, then loading into parquet [not working, OOM]
#
# Too memory hungry since it loads the full csv inmemory
# at copy time, so pass by the duckdb database)
#  %%
def convert_table_to_parquet_duckdb(
    path2read_table: Path, path2write_table: Path, table_name: str
):
    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE TABLE {table_name} as SELECT * FROM read_csv_auto('{str(path2read_table)}'); "
    )
    con.execute(
        f"COPY {table_name} TO '{str(path2write_table)}' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);"
    )


# %%
dir2mimic_csv = "/home/mdoutrel/projets/inria/mimiciv-2.2"

module = "hosp"
table_name = "labevents"
table_name = "patients"
path2read_table = Path(dir2mimic_csv) / module / f"{table_name}.csv.gz"
path2write_table = DIR2MIMIC / f"{table_name}"
convert_table_to_parquet_duckdb(path2read_table, path2write_table, table_name)
# %%
# ## Polars [not working]

# The sink_parquet method is not working since it seems to load all data in
# memory, I think that it is linked to strange separators in the original files.
# I have to pass by duckdb to dump parquet, then I'll use polars ?
import polars as pl


def convert_table_to_parquet_polars(
    path2read_table: Path, path2write_table: Path
):
    df = pl.scan_csv(path2read_table, cache=False, low_memory=True)
    breakpoint()
    df.sink_parquet(path2write_table, row_group_size=100_000)


# %%
dir2mimic_csv = "/home/mdoutrel/projets/inria/mimiciv-2.2"

module = "hosp"
table_name = "labevents"
# table_name = "patients"
path2read_table = Path(dir2mimic_csv) / module / f"{table_name}.csv"
path2write_table = DIR2MIMIC / f"{table_name}.parquet"
convert_table_to_parquet_polars(path2read_table, path2write_table)
# %%

# %% [markdown]
# # Build mimic concepts sql queries on mimic-iv with duckdb [not working]
# It works but the syntax between postgresql and duckdb is too different and
# need to be heavily adpated.
# %%
dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"
con = duckdb.connect(database=dir2mimic_duckdb)
register_sql_functions(con)
con.execute("USE mimiciv_derived")
# %%
# Building sofa
query = get_concept_query("demographics/icustay_times")
con.execute(query)
# %%
icustay_times = con.execute(
    "SELECT * FROM mimiciv_derived.icustay_times"
).fetch_df()
icustay_times
# %%
query = get_concept_query("demographics/icustay_hourly")
"""
-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS icustay_hourly; CREATE TABLE icustay_hourly AS
-- This query generates a row for every hour the patient is in the ICU.
-- The hours are based on clock-hours (i.e. 02:00, 03:00).
-- The hour clock starts 24 hours before the first heart rate measurement.
-- Note that the time of the first heart rate measurement is ceilinged to
-- the hour.

-- this query extracts the cohort and every possible hour they were in the ICU
-- this table can be to other tables on stay_id and (ENDTIME - 1 hour,ENDTIME]

-- get first/last measurement time
WITH all_hours AS (
    SELECT
        it.stay_id

        -- ceiling the intime to the nearest hour by adding 59 minutes,
        -- then applying truncate by parsing as string
        -- string truncate is done to enable compatibility with psql
        , PARSE_DATETIME(
            '%Y-%m-%d %H:00:00'
            , FORMAT_DATETIME(
                '%Y-%m-%d %H:00:00'
                , DATETIME_ADD(CAST(it.intime_hr AS TIMESTAMP), INTERVAL '59' MINUTE)
            )) AS endtime

        -- create integers for each charttime in hours from admission
        -- so 0 is admission time, 1 is one hour after admission, etc,
        -- up to ICU disch
        --  we allow 24 hours before ICU admission (to grab labs before admit)
        , ARRAY(SELECT * FROM generate_series(-24, CEIL(DATETIME_DIFF(it.outtime_hr, it.intime_hr, 'HOUR')))) AS hrs -- noqa: L016
    FROM mimiciv_derived.icustay_times it
)

SELECT stay_id
    , CAST(hr AS bigint) AS hr
    , DATETIME_ADD(endtime, interval '1' hour * CAST(hr AS bigint)) AS endtime
FROM all_hours
CROSS JOIN UNNEST(all_hours.hrs) AS hr;

"""

con.execute(query)
# %%
patients_ddb = con.execute(
    "SELECT * FROM mimiciv_hosp.patients LIMIT 10"
).fetch_df()
# %%
query = get_concept_query("score/sofa")
con.execute(query)
