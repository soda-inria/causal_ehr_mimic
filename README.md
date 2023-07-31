# Step-by-step Causal Analysis Of Ehrs To Ground Decision Making


## Overview

This code was used to study the effect of a combination of Albumin+crystalloids compared to crystalloids only for sepsis patients on 28 day mortality in the Mimic-iv database. 

The corresponding paper is: 

```
Step-by-step Causal Analysis Of Ehrs To Ground Decision Making, 
```

## Step 0: data acquisition

We used a combination of duckdb and postgresql to build mimic for a laptop into parquet files easy to process during the analysis.

<details>
<summary>Steps to reproduce data acquisition on a laptop</summary>
        
    0. We first built the data using postgresql following [mimic-code instructions](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) *=30min-1h*.
    1. We then used built the concepts from the [postgresql concepts scripts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres) *=1h-2h*.
    2. We built the original database with duckdb *=16min* following [mimic-code instructions](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/duckdb). This is done to copy the large files to parquet without memory overflow.
    3. We convert all tables from the original database to parquet files using duckdb: `cli.duckdb2parquet` *=5min*.
    4. We convert all derived tables from postgresql to parquet using polars: `cli.mimiciv_derived2parquet` *=2min*.

    NB: Step 2 and 3 could be done from polars or duckdb directly from the csv, I think. But I had some bugs with the `pl.sink_parquet` when trying to convert the csv.

    **Computing setup:** The analysis was performed on a laptop running Ubuntu 22.04.2 LTS with the following hardware: CPU 12th Gen
    Intel(R) Core(TM) i7-1270P with 16 threads and 15 GB of RAM.
    
</details>

## Step 1: study design – Frame the question to avoid biases

 - Code for main paper: `caumim.framing.albumin_for_sepsis.py`
 - Step-by-step notebook: `notebooks/_1_framing_albumin_for_sepsis.py`

## Step 2: identification – List necessary information to answer the causal question

We built the causal graph with a clinician, using the online graphical tool [Daggity](https://dagitty.net/).

The practical implementation of the selection of these covariates in MIMIC-IV is done with the `get_event_covariates_albumin_zhou` function in `caumim.variables.selection.py`
## Step 3: Estimation – Compute the causal effect of interest

- Code for main paper: `caumim.experiments.sensitivity_albumin_for_sepsis.py`
- Step-by-step notebook: `notebooks/_3_estimation__albumin_for_sepsis.py`

## Step 4: Vibration analysis – Assess the robustness of the hypotheses

- Code for main paper: 
  - Vibration analysis on causal and statistical estimators: `caumim.experiments.sensitivity_albumin_for_sepsis.py`
  - Vibration analysis on immortal time bias: `caumim.experiments.immortal_time_bias_albumin_for_sepsis.py`
  - Vibration analysis on causal and statistical estimators: `caumim.experiments.sensitivity_feature_aggregation_albumin_for_sepsis.py`

- Step-by-step notebook: `notebooks/_3_estimation__albumin_for_sepsis.py`

## Step 5: Treatment heterogeneity – Compute treatment effects on subpopulations

- Code for the main paper: `caumim.experiments.cate_exploration_albumin_for_sepsis.py`
