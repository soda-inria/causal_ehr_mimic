# Causal analysis of EHRs to ground decision making: a tutorial on mimic-iv 


## Overview

This code was used to study the effect of albumin+crystalloids compared to crystalloids only for sepsis patients on 28 day mortality in the Mimic-iv database. 

## Data

### Data acquisition

0. We first built the data using postgresql following [mimic-code instructions](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) *=30min-1h*.
1. We then used built the concepts from the [postgresql concepts scripts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres) *=1h-2h*.
2. We built the original database with duckdb *=16min* following [mimic-code instructions](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/duckdb). This is done to copy the large files to parquet without memory overflow.
3. We convert all tables from the original database to parquet files using duckdb: `cli.duckdb2parquet` *=5min*.
4. We convert all derived tables from postgresql to parquet using polars: `cli.mimiciv_derived2parquet` *=2min*.

NB: Step 2 and 3 could be done from polars or duckdb directly from the csv, I think. But I had some bugs with the `pl.sink_parquet` when trying to convert the csv.