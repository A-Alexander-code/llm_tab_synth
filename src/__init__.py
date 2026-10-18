"""Public utilities loaded lazily so data preparation needs no model stack."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "io_utils": ["load_dataset", "save_dataframe", "load_params", "save_json", "load_json"],
    "hashing": ["hash_dataframe", "hash_params", "compute_experiment_hash"],
    "logging_utils": ["setup_logging"],
    "sampling": ["split_by_class", "apply_imbalance_ratio", "train_test_split_by_class"],
    "model_registry": ["is_model_trained", "register_trained_model", "get_model_metadata"],
    "timing": ["Timer", "save_timing", "load_timing"],
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name: str) -> Any:
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"src.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
