"""TabPFGen-based generator."""

from __future__ import annotations

import pickle
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.generators.base import BaseGenerator
from tabpfgen import TabPFGen  


class TabpfgenGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        """Persist training data for later sampling."""
        from src.revision_generator_parameters import revised_context
        revised = revised_context(self.params, "tabpfgen")
        if revised:
            from src.revision_tabpfgen import check_fit
            check_fit(df, model_dir, self.params)
        model_dir.mkdir(parents = True, exist_ok = True)

        if "__target__" not in df.columns:
            raise ValueError(
                "TabpfgenGenerator.fit() expects df to include '__target__'."
            )

        feature_cols = [c for c in df.columns if c != "__target__"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["__target__"].values.astype(np.int32)

        with model_dir.joinpath("train_data.pkl").open("wb") as f:
            pickle.dump({"X": X, "y": y, "feature_cols": feature_cols}, f)
        if revised:
            from src.revision_tabpfgen import bind_training
            bind_training(model_dir, self.params, feature_cols)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        """Generate and cache samples for all classes, then return them together."""
        from src.revision_generator_parameters import revised_context
        if revised_context(self.params, "tabpfgen"):
            from src.revision_tabpfgen import sample_revised
            self.sampling_observations = {}
            return sample_revised(self.params, n_samples, model_dir, TabPFGen, self.sampling_observations)

        with model_dir.joinpath("train_data.pkl").open("rb") as f:
            state = pickle.load(f)

        X_train: np.ndarray = state["X"]
        y_train: np.ndarray = state["y"]
        feature_cols: list[str] = state["feature_cols"]

        unique_classes = np.unique(y_train).tolist()

        cache_path = model_dir.joinpath("samples_cache.pkl")
        audit_path = model_dir.joinpath("sampling_audit.json")
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
                previous = json.loads(audit_path.read_text()) if audit_path.exists() else None
                self.sampling_observations = {
                    "policy": "existing deterministic cycling to fill nonempty classes; unchanged",
                    "cache_reused": True, "requested_per_class": n_samples,
                    "raw_library_rows": 0, "returned_rows": len(df_all),
                    "cache_origin": previous.get("cache_origin", previous) if previous else None,
                }
                audit_path.write_text(json.dumps(self.sampling_observations, indent=2) + "\n")
                return df_all

        model = TabPFGen(
            n_sgld_steps = self.params["n_sgld_steps"],
            sgld_step_size = self.params["sgld_step_size"],
            sgld_noise_scale = self.params["sgld_noise_scale"],
            device = self.params["device"],
        )

        collected: dict[int, list] = {int(cls): [] for cls in unique_classes}

        max_iterations = self.params.get("max_iterations", 200)
        iteration = 0
        raw_rows = 0
        ignored_rows = 0
        batch_sizes = []

        while iteration < max_iterations:
            missing = {
                cls: max(0, n_samples - len(collected[cls]))
                for cls in collected
            }
            total_missing = sum(missing.values())
            if total_missing <= 0:
                break

            batch_size = max(total_missing * 4, 100)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message = ".*Running on CPU with more than 200 samples.*",
                    category = UserWarning,
                    module = "tabpfn",
                )
                X_batch, y_batch = model.generate_classification(
                    X_train, y_train,
                    n_samples = batch_size,
                    balance_classes = True,
                )
            raw_rows += len(X_batch)
            batch_sizes.append({"X_rows": len(X_batch), "y_rows": len(y_batch)})

            for x_row, y_val in zip(X_batch, y_batch):
                label = int(y_val)
                if label in collected and len(collected[label]) < n_samples:
                    collected[label].append(list(x_row) + [label])
                else:
                    ignored_rows += 1

            iteration += 1

        accepted_before_padding = {int(cls): len(collected[cls]) for cls in unique_classes}
        for cls in unique_classes:
            shortfall = n_samples - len(collected[cls])
            if shortfall > 0 and len(collected[cls]) > 0:
                import warnings as _w
                _w.warn(
                    f"TabPFGen: class {cls} only generated {len(collected[cls])}/{n_samples} "
                    f"samples after {max_iterations} iterations. "
                    f"Filling remainder by resampling existing rows.",
                    UserWarning, stacklevel=2,
                )
                existing = collected[cls].copy()
                while len(collected[cls]) < n_samples:
                    collected[cls].append(existing[len(collected[cls]) % len(existing)])

        with cache_path.open("wb") as f:
            pickle.dump(collected, f)

        rows = []
        for cls in unique_classes:
            rows.extend(collected[cls][:n_samples])

        df_all = pd.DataFrame(rows, columns=feature_cols + ["__label__"])
        self.sampling_observations = {
            "policy": "existing deterministic cycling to fill nonempty classes; unchanged",
            "cache_reused": False, "requested_per_class": n_samples,
            "iterations": iteration, "max_iterations": max_iterations,
            "raw_library_rows": raw_rows, "batch_sizes": batch_sizes,
            "ignored_unknown_label_or_full_quota_rows": ignored_rows,
            "accepted_before_padding": accepted_before_padding,
            "padding_copies": {int(cls): len(collected[cls]) - accepted_before_padding[int(cls)] for cls in unique_classes},
            "returned_per_class": {int(cls): len(collected[cls]) for cls in unique_classes},
            "returned_rows": len(df_all),
        }
        audit_path.write_text(json.dumps(self.sampling_observations, indent=2) + "\n")
        return df_all
