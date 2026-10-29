"""TVAE-based synthetic data generator (SDV)."""

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
        from src.revision_sdv import fit_metadata
        metadata = fit_metadata(self.params, "tvae", df, model_dir, Metadata)
 
        model = TVAESynthesizer(
            metadata,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            compress_dims=tuple(self.params["compress_dims"]),
            decompress_dims=tuple(self.params["decompress_dims"]),
            embedding_dim=self.params["embedding_dim"],
        )
        model.fit(df)
        metadata.save_to_json(str(model_dir.joinpath("metadata.json")))
        model.save(str(model_dir.joinpath("tvae_model.pkl")))
        from src.revision_sdv import bind_model
        bind_model(self.params, "tvae", model_dir)
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        from src.revision_sdv import check_sample
        check_sample(self.params, "tvae", model_dir)
        model = sdv_utils.load_synthesizer(
            str(model_dir.joinpath("tvae_model.pkl"))
        )
        return model.sample(num_rows=n_samples)
 