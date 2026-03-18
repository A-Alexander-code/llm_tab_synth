"""Helpers to isolate per-model training auxiliary artifacts."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from config.const import MODELS_DIR, TRAIN_MODEL_INFO_DIR


def get_training_info_dir(model_dir: Path) -> Path:
    try:
        relative_model_dir = model_dir.resolve().relative_to(MODELS_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"model_dir must live under {MODELS_DIR}, got {model_dir}"
        ) from exc

    return TRAIN_MODEL_INFO_DIR.joinpath(relative_model_dir)


@contextmanager
def training_workspace(model_dir: Path):
    workspace_dir = get_training_info_dir(model_dir)

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    os.chdir(workspace_dir)
    try:
        yield workspace_dir
    finally:
        os.chdir(previous_cwd)
