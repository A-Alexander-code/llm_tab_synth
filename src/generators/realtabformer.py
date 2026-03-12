"""REaLTabFormer-based synthetic data generator."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from realtabformer import REaLTabFormer

from src.generators.base import BaseGenerator


class RealtabformerGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model_dir.mkdir(parents=True, exist_ok=True)

        model = REaLTabFormer(
            model_type=self.params["model_type"],
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            gradient_accumulation_steps=self.params["gradient_accumulation_steps"],
            early_stopping_patience=self.params["early_stopping_patience"],
        )

        fit_kwargs = {}
        if "num_bootstrap" in self.params:
            fit_kwargs["num_bootstrap"] = self.params["num_bootstrap"]

        model.fit(df, **fit_kwargs)
        model.save(str(model_dir))

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model = REaLTabFormer.load_from_dir(str(model_dir))
        return model.sample(n_samples)