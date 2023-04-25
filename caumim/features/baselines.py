import polars as pl

from caumim.constants import (
    COLNAME_CODE,
    COLNAME_DOMAIN,
    COLNAME_END,
    COLNAMES_EVENTS,
    COLNAME_ICUSTAY_ID,
    COLNAME_LABEL,
    COLNAME_VALUE,
    DIR2MIMIC,
    DIR2RESOURCES,
)


def get_baseline_sepsis():
    return


def get_antibiotics_event_from_atc4(atc4_codes) -> pl.LazyFrame:
    drugs = pl.scan_parquet(DIR2MIMIC / "mimiciv_hosp.prescriptions/*")
    ndc_map = pl.scan_parquet(DIR2RESOURCES / "ontology" / "ndc_map")
    # Using the NDC map to get the ATC4 code, NDC of 10 digits should be padded with
    # a leading 0
    atb_ndc_map = (
        ndc_map.filter(pl.col("atc4").is_in(atc4_codes))
        .select(["ndc", "atc4", "atc4_name", "in_name"])
        .with_columns(
            (pl.lit("0") + pl.col("ndc").str.replace_all("-", "")).alias("ndc")
        )
    )
    antibiotics_of_interest = drugs.join(atb_ndc_map, on="ndc", how="inner")
    # %%
    antibiotics_event = antibiotics_of_interest.with_columns(
        [
            pl.col("stoptime").alias(COLNAME_END),
            pl.lit("drug").alias(COLNAME_DOMAIN),
            pl.col("atc4").alias(COLNAME_CODE),
            pl.col("atc4_name").alias(COLNAME_LABEL),
            pl.lit(1).alias(COLNAME_VALUE),
        ]
    )
    return antibiotics_event
