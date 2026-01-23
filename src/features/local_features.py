import pandas as pd
import numpy as np
from typing import Sequence


def generate_lags(
    df: pd.DataFrame,
    grouping_col: str = "M4id",
    time_col: str = "time_idx",
    value_col: str = "value",
    lags_list: Sequence[int] | None = None
) -> pd.DataFrame:
    """
    Add lagged features to a time series dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format time series data.
    grouping_col : str
        Column identifying individual time series.
    time_col : str
        Time index column (must be sortable).
    value_col : str
        Column containing observed values.
    lags_list : sequence of int
        Lags to generate (e.g. [1, 7, 14]).

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features added.
    """

    # Check for empty lags_list
    if lags_list is None or len(lags_list) == 0:
        return df.copy()

    # Check for positive integers in lags_list
    if not all(isinstance(lag, int) and lag > 0 for lag in lags_list):
        raise ValueError("All lags must be positive integers")

    # Order by time_col to ensure proper lag computation
    df_lag = df.sort_values([grouping_col, time_col]).copy()

    # Create grouped df object and iterate
    grouped = df_lag.groupby(grouping_col)[value_col]

    for lag in lags_list:
        df_lag[f"lag_{lag}"] = grouped.shift(lag)

    return df_lag


def generate_log_change(
    df: pd.DataFrame,
    grouping_col: str = "M4id",
    time_col: str = "time_idx",
    value_col: str = "value",
    logch_col: str = "log_change",
) -> pd.DataFrame:
    """
    Computes the log-return of a time series within each group.

    Log-return formula:
        log_return_t = log(value_t) - log(value_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        Long-format time series data.
    grouping_col : str
        Column identifying individual time series (e.g., 'M4id').
    time_col : str
        Column representing the time index.
    value_col : str
        Column containing the observed values (must be strictly positive).
    logch_col : str
        Name of the new column to store log-returns.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with a new column containing log-returns.
        The first row of each series will be NaN because there is no previous value.
    """
    # Ensure correct temporal order
    df_lr = df.sort_values([grouping_col, time_col]).copy()

    # Previous value within each series
    prev = df_lr.groupby(grouping_col)[value_col].shift(1)

    # Compute log-return
    df_lr[logch_col] = np.log(df_lr[value_col]) - np.log(prev)

    return df_lr


