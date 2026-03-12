"""GReaT-based synthetic data generator."""

from __future__ import annotations
 
import io
import os
import shutil
from pathlib import Path
 
import pandas as pd
import torch
from be_great import GReaT
 
from src.generators.base import BaseGenerator
 
 
def _load_great_model(model: GReaT, model_dir: Path) -> None:
    model_pt = model_dir.joinpath("model.pt")
    if not model_pt.exists():
        model_pt = model_dir
 
    buffer = io.BytesIO(model_pt.read_bytes())
    state_dict = torch.load(buffer, map_location="cpu")
    model.model.load_state_dict(state_dict)
 
 
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

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.save(str(model_dir))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
 
        model = GReaT(llm=self.params["llm"])
        _load_great_model(model, model_dir)
 
        return model.sample(
            n_samples,
            guided_sampling=True,
            max_length=500,
        )
 