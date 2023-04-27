#! /usr/bin/env python
import logging

import click
from dotenv import load_dotenv

from caumim.acquisition.data_acquisition import (
    convert_mimic_derived_from_postgresql_to_parquet,
    convert_mimic_from_duckdb_to_parquet,
)
from caumim.constants import LOG_LEVEL
from caumim.framing.albumin_for_sepsis import (
    COHORT_CONFIG_ALBUMIN_FOR_SEPSIS,
    get_population,
)

# see `.env` for requisite environment variables
load_dotenv()

logging.basicConfig(
    level=logging.getLevelName(LOG_LEVEL),
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path2duckdb", help="Path to the mimic duckdb database")
def duckdb2parquet(path2duckdb: str) -> None:
    """Convert the mimic database from duckdb to parquet."""
    convert_mimic_from_duckdb_to_parquet(path2duckdb)


@cli.command()
@click.option(
    "--psql_con",
    help="psql connection information (template: https://pola-rs.github.io/polars-book/user-guide/howcani/io/postgres.html)",
)
def mimiciv_derived2parquet(psql_con: str) -> None:
    """Convert the mimic derived tables from postgresql to parquet."""
    convert_mimic_derived_from_postgresql_to_parquet(psql_con)


available_populations = ["albumin_for_sepsis"]


@cli.command()
@click.option(
    "-tp",
    "--target_population",
    type=click.Choice(available_populations, case_sensitive=False),
    help="Population of interest",
)
def framing(target_population: str) -> None:
    """Build the target population for the study."""
    if target_population == "albumin_for_sepsis":
        get_population(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)


if __name__ == "__main__":
    cli()
