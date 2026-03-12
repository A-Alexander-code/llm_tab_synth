"""REaLTabFormer-based synthetic data generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from realtabformer import REaLTabFormer

from src.generators.base import BaseGenerator


class RealtabformerGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model = REaLTabFormer(
            model_type=self.params["model_type"],
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            gradient_accumulation_steps=self.params["gradient_accumulation_steps"],
            early_stopping_patience=self.params["early_stopping_patience"],
        )
        model.fit(df, num_bootstrap=None)
        model.save(str(model_dir))

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = REaLTabFormer.load(str(model_dir))
        return model.sample(n_samples)