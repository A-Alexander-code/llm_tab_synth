"""GReaT-based synthetic data generator."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from be_great import GReaT

from src.generators.base import BaseGenerator

logger = logging.getLogger(__name__)


class GreatGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model = GReaT(
                llm=self.params["llm"],
                batch_size=self.params["batch_size"],
                epochs=self.params["epochs"],
                fp16=self.params["fp16"],
                dataloader_num_workers=self.params["dataloader_num_workers"],
                gradient_checkpointing=self.params["gradient_checkpointing"],
            )
            model.fit(df)
        except Exception as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.warning(
                    "CUDA error detected: %s. Retrying with CPU...", str(e)[:100]
                )
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                model = GReaT(
                    llm=self.params["llm"],
                    batch_size=self.params["batch_size"],
                    epochs=self.params["epochs"],
                    fp16=False,
                    dataloader_num_workers=0,
                    gradient_checkpointing=False,
                )
                model.fit(df)
            else:
                raise
        
        model.save(str(model_dir.joinpath("great_model")))

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        model = GReaT.load(str(model_dir.joinpath("great_model")))
        return model.sample(
            n_samples,
            guided_sampling=True,
            max_length=500,
        )