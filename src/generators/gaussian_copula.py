"""GaussianCopulaSynthesizer-based synthetic data generator (SDV)."""

from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv import utils as sdv_utils
 
from src.generators.base import BaseGenerator
 
 
class GaussianCopulaGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = Metadata.detect_from_dataframe(df)
 
        model = GaussianCopulaSynthesizer(
            metadata,
            default_distribution=self.params["default_distribution"],
        )
        model.fit(df)
        metadata.save_to_json(str(model_dir.joinpath("metadata.json")))
        model.save(str(model_dir.joinpath("gaussian_copula_model.pkl")))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = sdv_utils.load_synthesizer(
            str(model_dir.joinpath("gaussian_copula_model.pkl"))
        )
        return model.sample(num_rows=n_samples)