"""TabuLa-based generator."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.generators.base import BaseGenerator

_TABULA_REPO = "https://github.com/zhao-zilong/Tabula"
_TABULA_DIR = Path(__file__).resolve().parent.parent.joinpath("models").joinpath("tabula")
_PARTITIONED_DATASETS = {"cervical_cancer", "kidney_disease", "T1D_child"}
_LOGGER = logging.getLogger(__name__)


def _ensure_tabula() -> None:
    if not _TABULA_DIR.exists():
        subprocess.run(
            ["git", "clone", _TABULA_REPO, str(_TABULA_DIR)],
            check=True,
        )
    if str(_TABULA_DIR) not in sys.path:
        sys.path.insert(0, str(_TABULA_DIR))


def _save_tabula(model, model_dir: Path) -> None:
    """Save a TabuLa model while skipping non-serializable attributes."""
    import numpy as np

    model_dir.mkdir(parents=True, exist_ok=True)

    attributes = model.__dict__.copy()
    for key in ["tokenizer", "model", "config"]:
        attributes.pop(key, None)

    if isinstance(attributes.get("conditional_col_dist"), np.ndarray):
        attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

    safe_attributes = {}
    for key, value in attributes.items():
        try:
            json.dumps(value)
            safe_attributes[key] = value
        except (TypeError, ValueError):
            pass

    with model_dir.joinpath("config.json").open("w", encoding="utf-8") as f:
        json.dump(safe_attributes, f)

    torch.save(model.model.state_dict(), str(model_dir.joinpath("model.pt")))


def _load_tabula(model_dir: Path, llm: str):
    """Load a TabuLa model from its saved config and weights."""
    _ensure_tabula()
    from tabula.tabula import Tabula

    config_path = model_dir.joinpath("config.json")
    model_pt = model_dir.joinpath("model.pt")

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found in {model_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = Tabula(llm=llm)

    for key, value in config.items():
        try:
            setattr(model, key, value)
        except AttributeError:
            pass

    buffer = io.BytesIO(model_pt.read_bytes())
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state_dict)
    model.model.config.pad_token_id = model.tokenizer.eos_token_id

    return model


def _compute_max_length(df: pd.DataFrame, margin: int = 20) -> int:
    """Estimate a suitable max_length from a sample of dataset rows."""
    _ensure_tabula()
    from tabula.tabula import Tabula

    tokenizer = Tabula(llm="distilgpt2").tokenizer
    sample = df.head(min(10, len(df)))
    lengths: list[int] = []

    for _, row in sample.iterrows():
        text = ", ".join(f"{col} is {val}" for col, val in row.items())
        tokens = tokenizer(text, return_tensors="pt")
        lengths.append(tokens["input_ids"].shape[1])

    estimated = max(lengths) + margin
    return max(80, min(estimated, 300))


def _resolve_dataset_name(model_dir: Path) -> str | None:
    """Infer the dataset name from the persisted model directory path."""
    for current in [model_dir, *model_dir.parents]:
        if current.name.startswith("seed_"):
            return current.parent.name
    return None


def _build_feature_partitions(columns: list[str], max_columns: int) -> list[list[str]]:
    """Split feature columns into contiguous partitions."""
    return [columns[idx:idx + max_columns] for idx in range(0, len(columns), max_columns)]


def _save_partition_plan(model_dir: Path, partitions: list[list[str]], all_columns: list[str]) -> None:
    """Persist the partition layout used for one fitted model."""
    with model_dir.joinpath("partition_plan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "partitioned",
                "all_columns": all_columns,
                "feature_partitions": partitions,
            },
            f,
        )


def _load_partition_plan(model_dir: Path) -> dict | None:
    """Load the partition layout for a partitioned TabuLa model."""
    plan_path = model_dir.joinpath("partition_plan.json")
    if not plan_path.exists():
        return None
    with plan_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fit_single_tabula(df: pd.DataFrame, model_dir: Path, llm: str, batch_size: int, epochs: int) -> None:
    """Fit and persist one TabuLa model for a given column subset."""
    _ensure_tabula()
    from tabula.tabula import Tabula

    model_dir.mkdir(parents=True, exist_ok=True)
    model = Tabula(
        llm=llm,
        batch_size=batch_size,
        epochs=epochs,
    )
    model.fit(df)
    _save_tabula(model, model_dir)

    estimated_max_length = _compute_max_length(df)
    _LOGGER.info(
        "[tabula] Estimated max_length=%d for model_dir=%s",
        estimated_max_length,
        model_dir,
    )
    with model_dir.joinpath("sampling_config.json").open("w", encoding="utf-8") as f:
        json.dump({"estimated_max_length": estimated_max_length}, f)


def _sample_in_batches(
    model,
    n_samples: int,
    batch_size: int,
    device: str,
    max_length: int,
    temperature: float,
    top_k: int,
) -> pd.DataFrame:
    """Generate samples in smaller batches and log progress after each batch."""
    batches: list[pd.DataFrame] = []
    generated = 0
    batch_idx = 0

    while generated < n_samples:
        batch_idx += 1
        current_batch_size = min(batch_size, n_samples - generated)
        batch_start = time.perf_counter()
        _LOGGER.warning(
            "[tabula] Starting batch %d with %d requested rows",
            batch_idx,
            current_batch_size,
        )
        batch_df = model.sample(
            current_batch_size,
            device=device,
            max_length=max_length,
            temperature=temperature,
            k=top_k,
        )
        if batch_df is None or batch_df.empty:
            raise RuntimeError(
                f"TabuLa returned an empty batch at batch {batch_idx} "
                f"with requested size {current_batch_size}."
            )
        batch_elapsed = time.perf_counter() - batch_start
        batches.append(batch_df)
        generated += len(batch_df)
        _LOGGER.warning(
            "[tabula] Finished batch %d in %.2fs; progress=%d/%d rows",
            batch_idx,
            batch_elapsed,
            generated,
            n_samples,
        )

    if not batches:
        return pd.DataFrame()

    return pd.concat(batches, ignore_index=True).iloc[:n_samples].reset_index(drop=True)


def _sample_from_saved_model(
    model_dir: Path,
    llm: str,
    n_samples: int,
    batch_size: int,
    device: str,
    max_length_default: int,
    temperature: float,
    top_k: int,
    force_fixed_max_length: bool,
) -> pd.DataFrame:
    """Load one persisted TabuLa model and sample from it."""
    sampling_config_path = model_dir.joinpath("sampling_config.json")
    if sampling_config_path.exists():
        with sampling_config_path.open("r", encoding="utf-8") as f:
            sampling_config = json.load(f)
        max_length = sampling_config.get("estimated_max_length", max_length_default)
    else:
        max_length = max_length_default

    if force_fixed_max_length:
        max_length = max_length_default

    _LOGGER.warning(
        "[tabula] Sampling with device=%s, n_samples=%d, batch_size=%d, max_length=%d, force_fixed_max_length=%s, model_dir=%s",
        device,
        n_samples,
        batch_size,
        max_length,
        force_fixed_max_length,
        model_dir,
    )

    load_start = time.perf_counter()
    model = _load_tabula(model_dir, llm=llm)
    load_elapsed = time.perf_counter() - load_start
    _LOGGER.warning("[tabula] Model loaded in %.2fs from %s", load_elapsed, model_dir)

    return _sample_in_batches(
        model=model,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )


class TabulaGenerator(BaseGenerator):
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        import warnings

        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")
        warnings.filterwarnings("ignore", message=".*loss_type=None.*")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = _resolve_dataset_name(model_dir)
        partition_max_columns = int(self.params.get("partition_max_columns", 10))
        use_partitioned_mode = (
            dataset_name in _PARTITIONED_DATASETS
            and len(df.columns) > partition_max_columns
        )

        if use_partitioned_mode:
            partitions = _build_feature_partitions(df.columns.tolist(), partition_max_columns)
            _save_partition_plan(model_dir, partitions, df.columns.tolist())
            _LOGGER.warning(
                "[tabula] Using partitioned mode for dataset=%s with %d partitions",
                dataset_name,
                len(partitions),
            )
            for partition_idx, columns in enumerate(partitions):
                partition_dir = model_dir.joinpath(f"partition_{partition_idx:02d}")
                _LOGGER.warning(
                    "[tabula] Fitting partition %d/%d with %d columns",
                    partition_idx + 1,
                    len(partitions),
                    len(columns),
                )
                _fit_single_tabula(
                    df=df.loc[:, columns].copy(),
                    model_dir=partition_dir,
                    llm=self.params["llm"],
                    batch_size=self.params["batch_size"],
                    epochs=self.params["epochs"],
                )
            return

        _fit_single_tabula(
            df=df,
            model_dir=model_dir,
            llm=self.params["llm"],
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
        )

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        import warnings

        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")

        cuda_device = self.params.get("cuda_device", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        if torch.cuda.is_available():
            device = f"cuda:{cuda_device}" if cuda_device.isdigit() else "cuda"
        else:
            device = "cpu"

        sample_batch_size = int(self.params.get("sample_batch_size", 100))
        sample_batch_size = max(1, min(sample_batch_size, n_samples))
        temperature = self.params.get("temperature", 0.7)
        top_k = self.params.get("top_k", 100)
        force_fixed_max_length = bool(self.params.get("force_fixed_max_length", False))
        max_length_default = int(self.params.get("max_length_sample", 200))

        partition_plan = _load_partition_plan(model_dir)
        if partition_plan is not None and partition_plan.get("mode") == "partitioned":
            sampled_partitions: list[pd.DataFrame] = []
            feature_partitions = partition_plan["feature_partitions"]
            _LOGGER.warning(
                "[tabula] Sampling in partitioned mode with %d partitions",
                len(feature_partitions),
            )
            for partition_idx, columns in enumerate(feature_partitions):
                partition_dir = model_dir.joinpath(f"partition_{partition_idx:02d}")
                _LOGGER.warning(
                    "[tabula] Sampling partition %d/%d with %d columns",
                    partition_idx + 1,
                    len(feature_partitions),
                    len(columns),
                )
                partition_df = _sample_from_saved_model(
                    model_dir=partition_dir,
                    llm=self.params["llm"],
                    n_samples=n_samples,
                    batch_size=sample_batch_size,
                    device=device,
                    max_length_default=max_length_default,
                    temperature=temperature,
                    top_k=top_k,
                    force_fixed_max_length=force_fixed_max_length,
                )
                missing_columns = [col for col in columns if col not in partition_df.columns]
                if missing_columns:
                    raise RuntimeError(
                        f"Partition {partition_idx:02d} is missing generated columns: {missing_columns}"
                    )
                sampled_partitions.append(partition_df.loc[:, columns].reset_index(drop=True))

            combined = pd.concat(sampled_partitions, axis=1)
            ordered_columns = partition_plan.get("all_columns", combined.columns.tolist())
            return combined.loc[:, ordered_columns]

        return _sample_from_saved_model(
            model_dir=model_dir,
            llm=self.params["llm"],
            n_samples=n_samples,
            batch_size=sample_batch_size,
            device=device,
            max_length_default=max_length_default,
            temperature=temperature,
            top_k=top_k,
            force_fixed_max_length=force_fixed_max_length,
        )
