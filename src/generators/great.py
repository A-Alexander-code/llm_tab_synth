"""
src/generators/great.py

Generador basado en GReaT (be_great).
LLM fine-tuned para datos tabulares usando distilgpt2 como backbone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from be_great import GReaT

from src.generators.base import BaseGenerator


class GreatGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model = GReaT(
            llm=self.params["llm"],
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            fp16=self.params["fp16"],
            dataloader_num_workers=self.params["dataloader_num_workers"],
            gradient_checkpointing=self.params["gradient_checkpointing"],
            save_finetuned_model=True,
            trained_model_dir=str(model_dir),
        )
        model.fit(df)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = GReaT.load_finetuned_model(str(model_dir))
        return model.sample(
            n_samples,
            guided_sampling=True,
            max_length=500,
        )