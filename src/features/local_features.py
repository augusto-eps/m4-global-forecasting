import pandas as pd
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
