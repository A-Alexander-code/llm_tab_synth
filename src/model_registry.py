"""
Management of trained model registry.
Verifies if a model has been trained using:
  1. Hash of dataset + parameters (integrity)
  2. Model file existence (physical presence)
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.io_utils import load_json, save_json

logger = logging.getLogger(__name__)

_HASH_FILENAME = "model_hash.json"


def is_model_trained(
    model_class_dir: Path,
    experiment_hash: str,
) -> bool:
    """
    Check if a model has been trained for a given class.

    Verifies: (1) directory and model file exist, (2) experiment hash matches.

    Args:
        model_class_dir: Path to model class directory (e.g., models/great/dataset/seed_42/class_0/)
        experiment_hash: Combined hash of dataset + parameters

    Returns:
        True if model exists and hash matches, False otherwise.
    """
    hash_file = model_class_dir.joinpath(_HASH_FILENAME)

    if not model_class_dir.exists() or not hash_file.exists():
        logger.debug("Model not found at: %s", model_class_dir)
        return False

    try:
        metadata = load_json(hash_file)
        stored_hash = metadata.get("experiment_hash", "")
        if stored_hash == experiment_hash:
            logger.info("Model already trained (hash match): %s", model_class_dir)
            return True
        else:
            logger.warning(
                "Hash mismatch for %s. Expected: %s | Found: %s. "
                "Model will be retrained.",
                model_class_dir, experiment_hash, stored_hash,
            )
            return False
    except Exception as exc:
        logger.warning("Could not read hash file %s: %s", hash_file, exc)
        return False


def register_trained_model(
    model_class_dir: Path,
    experiment_hash: str,
    metadata: dict | None = None,
) -> None:
    """
    Register a trained model by saving its hash and metadata.

    Args:
        model_class_dir: Path to model class directory
        experiment_hash: Combined hash of dataset + parameters
        metadata: Optional additional information (model, dataset, seed, etc.)
    """
    hash_file = model_class_dir.joinpath(_HASH_FILENAME)
    record = {
        "experiment_hash": experiment_hash,
        **(metadata or {}),
    }
    save_json(record, hash_file)
    logger.debug("Model registered at: %s", model_class_dir)


def get_model_metadata(model_class_dir: Path) -> dict:
    """
    Return the registered metadata of a trained model.

    Args:
        model_class_dir: Path to model class directory

    Returns:
        Dictionary with metadata or empty dict if not found.
    """
    hash_file = model_class_dir.joinpath(_HASH_FILENAME)
    if not hash_file.exists():
        return {}
    try:
        return load_json(hash_file)
    except Exception:
        return {}