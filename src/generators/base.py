"""Abstract base class defining interface for synthetic data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseGenerator(ABC):
    """Common interface requiring fit() and sample() methods."""

    def __init__(self, params: dict) -> None:
        self.params = params

    @abstractmethod
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        """Train and persist model state to model_dir."""
        ...

    @abstractmethod
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        """Generate n_samples synthetic samples from persisted model."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"