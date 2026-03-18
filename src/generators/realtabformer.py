"""REaLTabFormer-based synthetic data generator."""

from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
from realtabformer import REaLTabFormer

from src.generators.base import BaseGenerator
from src.training_workspace import training_workspace
 
 
def _find_rtf_subdir(model_dir: Path) -> Path:
    """
    REaLTabFormer saves the model in a subdirectory with a unique ID.
    Recursively searches for the directory containing rtf_config.json.
    """
    matches = list(model_dir.rglob("rtf_config.json"))
    if not matches:
        raise FileNotFoundError(
            f"No rtf_config.json found under {model_dir}. "
            "The model may not have been saved correctly."
        )
    return matches[0].parent
 
 
class RealtabformerGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        model_dir.mkdir(parents=True, exist_ok=True)

        with training_workspace(model_dir):
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
        rtf_dir = _find_rtf_subdir(model_dir)
        model = REaLTabFormer.load_from_dir(str(rtf_dir))
        return model.sample(n_samples)
 
