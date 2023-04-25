# %%
import polars as pl
from caumim.data_acquisition.data_acquisition import DERIVED_TABLE_NAMES
from caumim.constants import DIR2MIMIC

# %%
conn = "postgres://mdoutrel:mimic@127.0.0.1:5432/mimiciv"
# %%
mimic_derived_table_name = DERIVED_TABLE_NAMES[0]
query = f"SELECT * FROM mimiciv_derived.{mimic_derived_table_name}"
df = pl.read_sql(query, conn)
df.write_parquet(DIR2MIMIC / f"mimic_derived.{mimic_derived_table_name}")
# %%
derived_tables = pl.read_sql("select * from information_schema.tables;", conn)
# %%
