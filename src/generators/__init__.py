"""Registry of available synthetic data generators and factory function."""

from src.io_utils import (
    load_dataset,
    save_dataframe,
    load_params,
    save_json,
    load_json,
)
from src.hashing import (
    hash_dataframe,
    hash_params,
    compute_experiment_hash,
)
from src.logging_utils import (
    setup_logging,
)
from src.data_splitter import (
    split_and_save,
    load_split,
    is_split_ready,
    reconstruct_dataframe,
)
from src.model_registry import (
    is_model_trained,
    register_trained_model,
    get_model_metadata,
)
from src.timing import (
    Timer,
    save_timing,
    load_timing,
)

__all__ = [
    # io
    "load_dataset",
    "save_dataframe",
    "load_params",
    "save_json",
    "load_json",
    # hashing
    "hash_dataframe",
    "hash_params",
    "compute_experiment_hash",
    # logging
    "setup_logging",
    # data splitting
    "split_and_save",
    "load_split",
    "is_split_ready",
    "reconstruct_dataframe",
    # model registry
    "is_model_trained",
    "register_trained_model",
    "get_model_metadata",
    # timing
    "Timer",
    "save_timing",
    "load_timing",
]