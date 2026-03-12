"""GaussianCopulaSynthesizer-based synthetic data generator (SDV)."""

from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sdv import utils as sdv_utils
 
from src.generators.base import BaseGenerator
 
 
class TvaeGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = Metadata.detect_from_dataframe(df)
 
        model = TVAESynthesizer(
            metadata,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            compress_dims=tuple(self.params["compress_dims"]),
            decompress_dims=tuple(self.params["decompress_dims"]),
            embedding_dim=self.params["embedding_dim"],
        )
        model.fit(df)
        model.save(str(model_dir.joinpath("tvae_model.pkl")))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = sdv_utils.load_synthesizer(
            str(model_dir.joinpath("tvae_model.pkl"))
        )
        return model.sample(num_rows=n_samples)