"""MostlyAI-based synthetic data generator (local GPU)."""

from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
from mostlyai import MostlyAI
 
from src.generators.base import BaseGenerator
 
 
class MostlyAIGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model_dir.mkdir(parents=True, exist_ok=True)
 
        mostly = MostlyAI(local=True)
        config = {
            "max_training_time": self.params["max_training_time"],
            "max_epochs":        self.params["max_epochs"],
            "verbose":           self.params["verbose"],
        }
        generator = mostly.train(data=df, config=config)
        generator.save(str(model_dir.joinpath("mostlyai_model")))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        mostly = MostlyAI(local=True)
        generator = mostly.load(str(model_dir.joinpath("mostlyai_model")))
        synthetic = mostly.generate(generator=generator, size=n_samples)
        return synthetic.data()
 