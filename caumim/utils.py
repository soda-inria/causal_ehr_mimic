import polars as pl
from typing import Union
import pandas as pd


def to_pandas(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """Convert a polars dataframe to a pandas dataframe.

    Args:
        df (Union[pl.LazyFrame, pl.DataFrame]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect().to_pandas()
    elif isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise ValueError(
            f"df must be a polars dataframe or a pandas dataframe, got {type(df)} instead"
        )
    return df


def to_polars(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pl.DataFrame:
    """Convert a pandas dataframe to a polars dataframe.

    Args:
        df (Union[pl.LazyFrame, pl.DataFrame]): _description_

    Returns:
        pl.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif isinstance(df, pl.DataFrame):
        pass
    elif isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    else:
        raise ValueError(
            f"df must be a polars dataframe or a pandas dataframe, got {type(df)} instead"
        )
    return df


def to_lazyframe(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pl.LazyFrame:
    """Convert a pandas dataframe to a polars dataframe.

     Args:
        df (Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]): _description_

    Returns:
        pl.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        pass
    else:
        df = to_polars(df).lazy()
    return df
