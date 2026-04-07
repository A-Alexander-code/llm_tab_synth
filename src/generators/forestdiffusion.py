"""ForestDiffusion-based synthetic data generator with iterative pool sampling."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from ForestDiffusion import ForestDiffusionModel

from src.generators.base import BaseGenerator

_POOL_MULTIPLIER = 5


class ForestDiffusionGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        """Train on full dataset with target column included."""
        model_dir.mkdir(parents=True, exist_ok=True)

        if "__target__" not in df.columns:
            raise ValueError(
                "ForestDiffusionGenerator.fit() expects df to include '__target__'."
            )

        feature_cols = [c for c in df.columns if c != "__target__"]
        X = df[feature_cols + ["__target__"]].values.astype(np.float32)
        
        xgb_params = {
            'tree_method': 'hist', 
            'device': 'cpu'       
        }
        
        model = ForestDiffusionModel(
            X,
            n_t=self.params["n_t"],
            duplicate_K=self.params["duplicate_K"],
            n_jobs=self.params["n_jobs"],
            **xgb_params
        )

        with model_dir.joinpath("forestdiffusion_model.pkl").open("wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        with model_dir.joinpath("forestdiffusion_model.pkl").open("rb") as f:
            state = pickle.load(f)

        model: ForestDiffusionModel = state["model"]
        feature_cols: list[str] = state["feature_cols"]

        collected = []
        total = 0
        while total < n_samples:
            pool_size = max(n_samples * _POOL_MULTIPLIER, 500)
            result = model.generate(batch_size=pool_size)

            if result.shape[1] == len(feature_cols) + 1:
                result = result[:, :-1]

            collected.append(result)
            total += len(result)

        all_samples = np.vstack(collected)[:n_samples]
        return pd.DataFrame(all_samples, columns=feature_cols)