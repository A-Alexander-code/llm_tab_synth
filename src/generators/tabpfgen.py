"""TabPFGen-based synthetic data generator with iterative pool sampling."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfgen import TabPFGen

from src.generators.base import BaseGenerator

_POOL_MULTIPLIER = 5


class TabpfgenGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        """Train on full dataset with both classes included."""
        model_dir.mkdir(parents=True, exist_ok=True)

        if "__target__" not in df.columns:
            raise ValueError(
                "TabpfgenGenerator.fit() expects df to include the '__target__' column. "
                "Pass the full training DataFrame with target included."
            )

        target_col = "__target__"
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.int32)

        model = TabPFGen(
            n_sgld_steps=self.params["n_sgld_steps"],
            sgld_step_size=self.params["sgld_step_size"],
            sgld_noise_scale=self.params["sgld_noise_scale"],
            device=self.params["device"],
        )
        model.fit(X, y)

        with model_dir.joinpath("tabpfgen_model.pkl").open("wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        with model_dir.joinpath("tabpfgen_model.pkl").open("rb") as f:
            state = pickle.load(f)

        model: TabPFGen = state["model"]
        feature_cols: list[str] = state["feature_cols"]

        collected = []
        while len(collected) < n_samples:
            pool_size = max(n_samples * _POOL_MULTIPLIER, 500)
            result = model.generate_classification(
                n_samples=pool_size,
                balance_classes=False,
            )
            collected.append(result)
            if sum(len(r) for r in collected) >= n_samples:
                break

        all_samples = np.vstack(collected)[:n_samples]
        return pd.DataFrame(all_samples, columns=feature_cols)