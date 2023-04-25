# %%
from pathlib import Path
import ibis
from ibis import deferred as c
import duckdb

ibis.options.interactive = True
print(ibis.__version__)
print(duckdb.__version__)

from caumim.constants import DIR2MIMIC

# %%
dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"
dir2mimic = "/home/mdoutrel/projets/inria/mimiciv-2.2/"
# %% [markdown]
# 1. Trying to connect to the on-memory database is partially working

# I can get the head, but not scan the entire files (eg. for counts). However this works if I use duckdb alone (see 3.)
# %%
duckdb_in_memory = ibis.duckdb.connect(database=":memory:")
# %%"
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
# 2. Trying to connect to the on-disk database is not working
# %%
duckdb_on_disk_conn = ibis.duckdb.connect(database=dir2mimic_duckdb)
# %%
patients = duckdb_on_disk_conn.table("patients")
patients.columns
patients.count()
# %% [markdown]
# 3. Using pure duck-db (not very practical since SQL only)
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
#%%
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
# 4. Save a parquet from duckdb, then access with Ibis
# It works perfectly. So it is indeed an issue with separators in the compressed csv that are badly interpreted by Ibis but correctly from duckdb
# %%
# save with duckdb wo collecting the dataframe.
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

# 5. try read directly from the csv.gz and load into parquet (TOO memory hungry since it loads the full csv inmemory at copy time, so pass by the duckdb database)
# %%
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
# 6. try using polars, the sink_parquet method is not working since it seems to
#    load all data in memory, I think that it is linked to strange separators in
#    the original files. I have to pass by duckdb to dump parquet, then I'll use
#    polars ?
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
#%%
