"""MostlyAI-based synthetic data generator (local GPU)."""

from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
 
from src.generators.base import BaseGenerator
 
 
class MostlyAIGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        from mostlyai.engine._tabular.interface import encode, split, train  # noqa: PLC0415
 
        model_dir.mkdir(parents=True, exist_ok=True)
        workspace = str(model_dir)
 
        # Paso 1: split (prepara particiones train/val dentro del workspace)
        split(
            tgt_data=df,
            workspace_dir=workspace,
        )
 
        # Paso 2: encode (tokeniza y codifica los datos)
        encode(workspace_dir=workspace)
 
        # Paso 3: train
        train(
            max_training_time=self.params["max_training_time"],
            max_epochs=self.params["max_epochs"],
            device=self.params.get("cuda_device", "0"),
            workspace_dir=workspace,
        )
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        from mostlyai.engine._tabular.interface import generate, load_generated_data  # noqa: PLC0415
 
        workspace = str(model_dir)
 
        generate(
            sample_size=n_samples,
            device=self.params.get("cuda_device", "0"),
            workspace_dir=workspace,
        )
 
        return load_generated_data(workspace_dir=workspace)