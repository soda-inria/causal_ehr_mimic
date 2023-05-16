from typing import Dict, List
from loguru import logger
import numpy as np
import polars as pl

from caumim.constants import (
    COLNAME_CODE,
    COLNAME_INCLUSION_START,
    COLNAME_PATIENT_ID,
    COLNAME_START,
    COLNAME_VALUE,
)

PERIOD_MAP = {"all": [100, 10, 25, 50, -10, -25, -50], "100": [100]}
DEFAULT_AGG_FUNCTIONS = {
    "min": pl.col(COLNAME_VALUE).min(),
    "max": pl.col(COLNAME_VALUE).max(),
    "mean": pl.col(COLNAME_VALUE).mean(),
    "std": pl.col(COLNAME_VALUE).std(),
    "count": pl.col(COLNAME_VALUE).count(),
}
OBSERVATION_START = "observation_start"
OBSERVATION_END = "observation_end"


def restrict_to_vocabulary(
    event: pl.DataFrame,
    vocabulary: List,
    colname_code: str = COLNAME_CODE,
) -> pl.DataFrame:
    restricted_event = event.join(
        pl.DataFrame(vocabulary, columns=[colname_code]),
        on=colname_code,
        how="inner",
    )
    return restricted_event


# TODO: debug this. It should work though.
def get_event_aggregation_polars(
    event: pl.DataFrame,
    aggregation_periods: str = "100",
    aggregation_functions: Dict = DEFAULT_AGG_FUNCTIONS,
    aggregation_col: str = COLNAME_PATIENT_ID,
    colname_code: str = COLNAME_CODE,
    vocabulary: List[str] = None,
    eps=1e-6,
) -> pl.DataFrame:
    """
    Compute aggregate statistics by patient trajectory and chosen events over
    different subsequences of the stay

    Parameters
    ----------
    person : pd.DataFrame
        _description_
    event : pd.DataFrame
        _description_
    aggregation_periods : str, optional
        _description_, by default "all"
    aggregation_functions : List, optional
        _description_, by default None
    aggregation_col : str, optional
        _description_, by default "person_id"
    colname_code : str, optional
        Choose the column on which do the aggregation, by default
        "event_source_concept_id"
    event_codes : str, optional
        If set, force at least these codes to appears in the columns
    eps : _type_, optional
        _description_, by default 1e-6

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # TODO: truncation time should be computed from events
    if aggregation_periods not in PERIOD_MAP.keys():
        raise ValueError(f"Supported aggregation periods are {PERIOD_MAP}")

    # need to fill the value for end of events
    if vocabulary is not None:
        code_whitelist = vocabulary.copy()
        event_restricted = restrict_to_vocabulary(
            event=event, vocabulary=code_whitelist
        )
    else:
        code_whitelist = (
            event[colname_code].value_counts()[colname_code].to_numpy()
        )
        event_restricted = event
    fake_lines_l = []
    for code in code_whitelist:
        fake_line = event_restricted[0].with_columns(
            pl.lit(-1).alias(aggregation_col),
            pl.lit(code).alias(colname_code),
            pl.col(COLNAME_START).alias(OBSERVATION_START),
            pl.col(COLNAME_START).alias(OBSERVATION_END),
            pl.lit(3600).alias("observation_delta").cast(pl.Int64),
            pl.lit(3600).alias("period_span_second").cast(pl.Int64),
        )
        fake_lines_l.append(fake_line)
    # pandas magic: the casting is necessary to avoid type error in the groupby operation
    fake_lines = pl.concat(fake_lines_l)
    aggregation_functions = {
        k: v.alias(k) for k, v in aggregation_functions.items()
    }
    periods = PERIOD_MAP[aggregation_periods]
    for sub_period in periods:
        if (sub_period <= -100) | (sub_period > 100):
            raise ValueError(
                f"period is a percentage and should be in ]-100, 100], got {sub_period}"
            )
    # add the delta observation as the difference between first and last
    # event observed by person
    first_event_datetime = (
        event_restricted.sort(COLNAME_START)
        .groupby(aggregation_col)
        .agg(pl.col(COLNAME_START).first().alias(OBSERVATION_START))
    )
    last_event_datetime = (
        event_restricted.sort(COLNAME_START)
        .groupby(aggregation_col)
        .agg(pl.col(COLNAME_START).last().alias(OBSERVATION_END))
    )
    events_of_interest = event_restricted.join(
        first_event_datetime, on=aggregation_col, how="inner"
    ).join(last_event_datetime, on=aggregation_col, how="inner")

    events_of_interest = events_of_interest.with_columns(
        (
            events_of_interest[OBSERVATION_END]
            - events_of_interest[OBSERVATION_START]
        )
        .dt.seconds()
        .alias("observation_delta")
    )
    X_list = []
    for sub_period in periods:
        logger.info(f"Computing period {sub_period}")
        events_of_interest = events_of_interest.with_columns(
            (events_of_interest["observation_delta"] * sub_period / 100)
            .alias("period_span_second")
            .cast(pl.Int64)
        )
        if sub_period >= 0:
            mask = (
                (
                    events_of_interest[COLNAME_START]
                    - events_of_interest[OBSERVATION_START]
                ).dt.seconds()
                >= -eps
            ) & (
                (
                    events_of_interest[COLNAME_START]
                    - events_of_interest[OBSERVATION_START]
                ).dt.seconds()
                <= (events_of_interest["period_span_second"] + eps)
            )
        else:
            mask = (
                (
                    events_of_interest[COLNAME_START]
                    - events_of_interest[OBSERVATION_START]
                ).dt.seconds()
                >= (
                    (
                        events_of_interest["observation_delta"]
                        + events_of_interest["period_span_second"]
                    )
                    - eps
                )
            ) & (
                (
                    events_of_interest[COLNAME_START]
                    - events_of_interest[OBSERVATION_END]
                ).dt.seconds()
                <= eps + events_of_interest["observation_delta"]
            )
        period_events = events_of_interest.filter(mask)
        period_events = pl.concat((period_events, fake_lines))

        period_statistics = (
            period_events.groupby([aggregation_col, colname_code])
            .agg(list(aggregation_functions.values()))
            .pivot(
                index=aggregation_col,
                columns=colname_code,
                values=list(aggregation_functions.keys()),
            )
        )
        period_statistics.columns = [
            f"{col}__p{sub_period}" if col != aggregation_col else col
            for col in period_statistics.columns
        ]
        X_list.append(period_statistics)
    X_final = X_list[0]
    for period_agg in X_list[1:]:
        X_final = X_final.join(period_agg, on=aggregation_col, how="left")
    return X_final
