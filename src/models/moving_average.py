import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Sequence
from .base import BaseLocalModel


class MovingAverageLocalModel(BaseLocalModel):
    """
    Local moving-average model using selected lags.
    """

    def __init__(self, lags: Sequence[int] = (1, 7, 14, 21, 28)):
        self.lags = sorted(lags)
        self.max_lag = max(self.lags)
        self.state: Dict[str, deque] = {}

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "M4id",
        time_col: str = "time_idx",
        value_col: str = "value",
    ) -> None:
        """
        Store last `max_lag` observations for each series.
        """

        df = df.sort_values([id_col, time_col])

        for series_id, group in df.groupby(id_col):
            tail = group[value_col].tail(self.max_lag).values
            self.state[series_id] = deque(tail, maxlen=self.max_lag)

    def predict(
        self,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Recursive multi-step forecast.
        """

        forecasts = []

        for series_id, history in self.state.items():
            # Copy history to append horizon 1 and forecast recursively
            hist = history.copy()

            for h in range(horizon):
                lag_values = [hist[-lag] for lag in self.lags if lag <= len(hist)]
                y_hat = np.mean(lag_values)
                hist.append(y_hat)

                forecasts.append({
                    "M4id": series_id,
                    "horizon": h + 1,
                    "y_hat": y_hat,
                })

        return pd.DataFrame(forecasts)

