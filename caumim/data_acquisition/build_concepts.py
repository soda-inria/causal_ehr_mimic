import requests

"""

Unsucessful attempt to run on duckdb the concept queries from the mimic-iv github repo.
"""
#MIMICIVCONCEPTS_URL = "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/"
MIMICIV_PSQL_CONCEPTS_URL = "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts_postgres/"

def register_sql_functions(con):
    """
    Register sql functions in duckdb to convert postgresql queries to duckdb.

    Args:
        con (_type_): _description_
    """
    con.execute(
        "CREATE OR REPLACE TEMP MACRO DATETIME_DIFF(e, s, p) AS datesub(CAST(p AS VARCHAR), CAST(s AS TIMESTAMP), CAST(e AS TIMESTAMP));"
    )
    con.execute(
        "CREATE OR REPLACE TEMP MACRO DATETIME(y, mo, d, h, m, s) AS make_timestamp(y, mo, d, h, m, s);"
    )
    con.execute(
        "CREATE OR REPLACE TEMP MACRO PARSE_DATETIME(f, dt) AS strptime(dt, '%Y-%m-%d %H:00:00');"
    )
    con.execute(
        "CREATE OR REPLACE TEMP MACRO DATETIME_ADD(dt, i) AS CAST(dt + i AS TIMESTAMP);"
    )
    con.execute(
        "CREATE OR REPLACE TEMP MACRO FORMAT_DATETIME(f, dt) AS strftime(CAST(dt AS TIMESTAMP), '%Y-%m-%d %H:00:00');"
    )
    return

def get_concept_query(concept_name: str, base_url:str = MIMICIV_PSQL_CONCEPTS_URL):
    """
    Get the sql query from the mimic-iv github repo.

    Args:
        concept_name (str): name of the concept to get the query from.

    Returns:
        str: sql query
    """
    concept_url = base_url + f"{concept_name}.sql"
    concept_query = requests.get(concept_url).text
    return concept_query