"""TabPFGen-based synthetic data generator with iterative pool sampling."""

from __future__ import annotations
 
import pickle
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from src.generators.base import BaseGenerator
 
 
class TabpfgenGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
 
        if "__target__" not in df.columns:
            raise ValueError(
                "TabpfgenGenerator.fit() expects df to include '__target__'."
            )
 
        feature_cols = [c for c in df.columns if c != "__target__"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["__target__"].values.astype(np.int32)
 
        with model_dir.joinpath("train_data.pkl").open("wb") as f:
            pickle.dump({"X": X, "y": y, "feature_cols": feature_cols}, f)
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:

        from tabpfgen import TabPFGen  
 
        with model_dir.joinpath("train_data.pkl").open("rb") as f:
            state = pickle.load(f)
 
        X_train: np.ndarray   = state["X"]
        y_train: np.ndarray   = state["y"]
        feature_cols: list[str] = state["feature_cols"]
 
        unique_classes = np.unique(y_train).tolist()
 
        cache_path = model_dir.joinpath("samples_cache.pkl")
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            if all(
                len(cache.get(cls, [])) >= n_samples for cls in unique_classes
            ):
                rows = []
                for cls in unique_classes:
                    rows.extend(cache[cls][:n_samples])
                df_all = pd.DataFrame(rows, columns=feature_cols + ["__label__"])
                return df_all
 
        model = TabPFGen(
            n_sgld_steps=self.params["n_sgld_steps"],
            sgld_step_size=self.params["sgld_step_size"],
            sgld_noise_scale=self.params["sgld_noise_scale"],
            device=self.params["device"],
        )
 
        collected: dict[int, list] = {int(cls): [] for cls in unique_classes}
 
        iteration = 1
        while True:
            missing = {
                cls: max(0, n_samples - len(collected[cls]))
                for cls in collected
            }
            total_missing = sum(missing.values())
            if total_missing <= 0:
                break
 
            batch_size = max(total_missing + 10, 50)
            X_batch, y_batch = model.generate_classification(
                X_train, y_train,
                n_samples=batch_size,
                balance_classes=True,
            )
 
            for x_row, y_val in zip(X_batch, y_batch):
                label = int(y_val)
                if label in collected and len(collected[label]) < n_samples:
                    collected[label].append(list(x_row) + [label])
 
            iteration += 1
 
        with cache_path.open("wb") as f:
            pickle.dump(collected, f)
 
        rows = []
        for cls in unique_classes:
            rows.extend(collected[cls][:n_samples])
 
        df_all = pd.DataFrame(rows, columns=feature_cols + ["__label__"])
        return df_all
 