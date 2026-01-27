from abc import ABC, abstractmethod
import pandas as pd

class BaseLocalModel(ABC):
    """Base class to define local (one time-series) model classes"""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        pass
