"""TabuLa-based generator."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from src.generators.base import BaseGenerator

_TABULA_REPO = "https://github.com/zhao-zilong/Tabula"
_TABULA_DIR  = Path(__file__).resolve().parent.parent.joinpath("models").joinpath("tabula")


def _ensure_tabula():
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
    for k, v in attributes.items():
        try:
            json.dumps(v)
            safe_attributes[k] = v
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
    model_pt    = model_dir.joinpath("model.pt")

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

    buffer     = io.BytesIO(model_pt.read_bytes())
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state_dict)

    model.model.config.pad_token_id = model.tokenizer.eos_token_id

    return model


def _compute_max_length(df: pd.DataFrame, margin: int = 20) -> int:
    """Estimate a suitable max_length from a sample of dataset rows."""
    _ensure_tabula()
    from tabula.tabula import Tabula  

    tokenizer = Tabula(llm="distilgpt2").tokenizer
    sample    = df.head(min(10, len(df)))
    lengths   = []

    for _, row in sample.iterrows():
        text   = ", ".join(f"{col} is {val}" for col, val in row.items())
        tokens = tokenizer(text, return_tensors="pt")
        lengths.append(tokens["input_ids"].shape[1])

    estimated = max(lengths) + margin
    return max(80, min(estimated, 300))


class TabulaGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        import warnings  
        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")
        warnings.filterwarnings("ignore", message=".*loss_type=None.*")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        _ensure_tabula()
        from tabula.tabula import Tabula  

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model = Tabula(
            llm=self.params["llm"],
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
        )
        model.fit(df)
        _save_tabula(model, model_dir)

        estimated_max_length = _compute_max_length(df)
        with model_dir.joinpath("sampling_config.json").open("w") as f:
            json.dump({"estimated_max_length": estimated_max_length}, f)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        import warnings  
        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")

        cuda_device = self.params.get("cuda_device", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        import torch  
        if torch.cuda.is_available():
            device = f"cuda:{cuda_device}" if cuda_device.isdigit() else "cuda"
        else:
            device = "cpu"

        sampling_config_path = model_dir.joinpath("sampling_config.json")
        if sampling_config_path.exists():
            with sampling_config_path.open("r") as f:
                sampling_config = json.load(f)
            max_length = sampling_config.get(
                "estimated_max_length",
                self.params.get("max_length_sample", 200)
            )
        else:
            max_length = self.params.get("max_length_sample", 200)

        model = _load_tabula(model_dir, llm=self.params["llm"])
        return model.sample(
            n_samples,
            device=device,
            max_length=max_length,
            temperature=self.params.get("temperature", 0.7),
            k=self.params.get("top_k", 100),
        )
