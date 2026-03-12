"""CTGAN-based synthetic data generator (SDV)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from src.generators.base import BaseGenerator


class CtganGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        model = CTGANSynthesizer(
            metadata,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            generator_dim=tuple(self.params["generator_dim"]),
            discriminator_dim=tuple(self.params["discriminator_dim"]),
        )
        model.fit(df)
        model.save(str(model_dir.joinpath("ctgan_model.pkl")))

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = CTGANSynthesizer.load(str(model_dir.joinpath("ctgan_model.pkl")))
        return model.sample(num_rows=n_samples)