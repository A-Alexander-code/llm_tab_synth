"""MostlyAI-based synthetic data generator (local GPU)."""

from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
 
from src.generators.base import BaseGenerator
 
 
class MostlyAIGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        from src.revision_generator_parameters import revised_context
        revised = revised_context(self.params, "mostlyai")
        # The installed API takes MINUTES. No conversion of the effective value.
        training_minutes = self.params["max_training_time_minutes"] if revised else self.params["max_training_time"]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        from mostlyai.engine._tabular.interface import (  # noqa: PLC0415
            analyze, encode, split, train,
        )
 
        model_dir.mkdir(parents=True, exist_ok=True)
        workspace = str(model_dir)
 
        split(tgt_data=df, workspace_dir=workspace)
        analyze(workspace_dir=workspace)
        encode(workspace_dir=workspace)
        train(
            max_training_time=training_minutes,
            max_epochs=self.params["max_epochs"],
            device=self.params.get("cuda_device", "0"),
            workspace_dir=workspace,
        )
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        from mostlyai.engine._tabular.interface import (  # noqa: PLC0415
            generate, load_generated_data,
        )
 
        workspace = str(model_dir)
        generate(
            sample_size=n_samples,
            device=self.params.get("cuda_device", "0"),
            workspace_dir=workspace,
        )
        return load_generated_data(workspace_dir=workspace)
