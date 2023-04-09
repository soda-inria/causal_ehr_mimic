# %%
import ibis
from ibis import deferred as c

ibis.options.interactive = True
ibis.__version__
# %%
import duckdb

duckdb.__version__
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

events_ddb = con.execute("SELECT * FROM mimiciv_hosp.drgcodes").fetch_df()
events_ddb.to_parquet(DIR2MIMIC / "drgcodes.parquet")
# %% [markdown]
# 4. Save a parquet from duckdb, then access with Ibis
# It works perfectly. So it is indeed an issue with separators in the compressed csv that are badly interpreted by Ibis but correctly from duckdb
# %%
# save with duckdb wo collecting the dataframe.
import duckdb
from caumim.constants import DIR2MIMIC

dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"
table_name = "labevents"
path2table = DIR2MIMIC / f"{table_name}"

con = duckdb.connect(database=dir2mimic_duckdb)
# events_ddb = con.execute("SELECT * FROM mimiciv_hosp.drgcodes").fetch_df()
con.execute(
    f"COPY mimiciv_hosp.{table_name} TO '{str(path2table)}' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);"
)
# %%
# read from ibis+duckdb
import ibis

ibis.options.interactive = True
from caumim.constants import DIR2MIMIC

table_name = "labevents"
path2table = DIR2MIMIC / f"{table_name}/*"

duckdb_in_memory = ibis.duckdb.connect(database=":memory:")
events = duckdb_in_memory.read_parquet(source_list=str(path2table))
events.count()
# %%
events.group_by("subject_id").count()
