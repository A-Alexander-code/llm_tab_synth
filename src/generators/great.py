"""GReaT-based synthetic data generator."""

from __future__ import annotations
 
import os
import shutil
from pathlib import Path
 
import pandas as pd
from be_great import GReaT
 
from src.generators.base import BaseGenerator
 
 
class GreatGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model_dir.mkdir(parents=True, exist_ok=True)
 
        model = GReaT(
            llm=self.params["llm"],
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            fp16=self.params["fp16"],
            dataloader_num_workers=self.params["dataloader_num_workers"],
            gradient_checkpointing=self.params["gradient_checkpointing"],
        )
        model.fit(df)
 
        # Limpiar directorio antes de guardar para evitar warning de be_great
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.save(str(model_dir))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        # load_finetuned_model es método de instancia: instanciar con el mismo llm
        model = GReaT(llm=self.params["llm"])
        model.load_finetuned_model(path=str(model_dir))
        return model.sample(
            n_samples,
            guided_sampling=True,
            max_length=500,
        )
 