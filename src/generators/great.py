"""GReaT-based synthetic data generator."""

from __future__ import annotations
 
import io
import json
import os
import shutil
from pathlib import Path
 
import pandas as pd
import torch
from be_great import GReaT

from src.generators.base import BaseGenerator
from src.training_workspace import training_workspace
 
 
def _restore_great_from_dir(model_dir: Path) -> GReaT:

    config_path = model_dir.joinpath("config.json")
    model_pt    = model_dir.joinpath("model.pt")
 
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found in {model_dir}")
 
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
 
    llm = config.get("llm", "distilgpt2")
 
    model = GReaT(llm=llm)
 

    for key, value in config.items():
        try:
            setattr(model, key, value)
        except AttributeError:
            pass 
 
    buffer     = io.BytesIO(model_pt.read_bytes())
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state_dict)
 
    return model
 
 
class GreatGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model_dir.mkdir(parents=True, exist_ok=True)

        with training_workspace(model_dir):
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
        model = _restore_great_from_dir(model_dir)
        return model.sample(
            n_samples,
            guided_sampling=True,
            max_length=500,
        )
