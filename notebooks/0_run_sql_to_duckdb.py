# %%
""""
This notebook tries to build [mimic concepts]() sql queries on mimic-iv with duckdb.
"""
%reload_ext autoreload
%autoreload 2
import duckdb
import requests
from caumim.data_acquisition.build_concepts import MIMICIV_PSQL_CONCEPTS_URL, get_concept_query, register_sql_functions
# %%$
dir2mimic_duckdb = "/home/mdoutrel/projets/inria/mimic-code/mimic-iv/buildmimic/duckdb/mimic4.db"
con = duckdb.connect(database=dir2mimic_duckdb)
register_sql_functions(con)
con.execute("USE mimiciv_derived")
# %%
# Building sofa
query = get_concept_query("demographics/icustay_times")
con.execute(query)
# %%
icustay_times = con.execute("SELECT * FROM mimiciv_derived.icustay_times").fetch_df()
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