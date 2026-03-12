"""Stratified train/test split with metadata preservation.

Saves: X_train, X_test, y_train, y_test (.npy) and feature_meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    test_size: float,
    seed: int,
) -> dict:
    """Stratified split with explicit class distribution preservation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    np.save(output_dir.joinpath("X_train.npy"), X_train)
    np.save(output_dir.joinpath("X_test.npy"),  X_test)
    np.save(output_dir.joinpath("y_train.npy"), y_train)
    np.save(output_dir.joinpath("y_test.npy"),  y_test)

    original_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    train_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    test_counts = {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
    original_ir = _compute_ir(original_counts)
    train_ir = _compute_ir(train_counts)
    test_ir = _compute_ir(test_counts)

    meta = {
        "dataset": {
            "n_samples_original": int(len(df)),
            "n_features": int(len(feature_cols)),
            "target_column": target_col,
            "feature_names": feature_cols,
            "feature_dtypes": {col: str(df[col].dtype) for col in feature_cols},
        },
        "split": {
            "seed": seed,
            "test_size": test_size,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "class_distribution": {
            "original": {
                "counts": original_counts,
                "imbalance_ratio": original_ir,
            },
            "train": {
                "counts": train_counts,
                "imbalance_ratio": train_ir,
            },
            "test": {
                "counts": test_counts,
                "imbalance_ratio": test_ir,
            },
        },
    }

    meta_path = output_dir.joinpath("feature_meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def load_split(
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load saved split from processed directory."""
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "feature_meta.json"]
    for fname in required:
        if not processed_dir.joinpath(fname).exists():
            raise FileNotFoundError(
                f"Missing file in processed dir: {processed_dir.joinpath(fname)}"
            )

    X_train = np.load(processed_dir.joinpath("X_train.npy"))
    X_test  = np.load(processed_dir.joinpath("X_test.npy"))
    y_train = np.load(processed_dir.joinpath("y_train.npy"))
    y_test  = np.load(processed_dir.joinpath("y_test.npy"))

    with processed_dir.joinpath("feature_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return X_train, X_test, y_train, y_test, meta


def is_split_ready(processed_dir: Path) -> bool:
    """Check if all required split files exist."""
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "feature_meta.json"]
    return all(processed_dir.joinpath(f).exists() for f in required)


def reconstruct_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
) -> pd.DataFrame:
    """Reconstruct DataFrame from arrays and metadata."""
    feature_names = meta["dataset"]["feature_names"]
    target_col    = meta["dataset"]["target_column"]

    df = pd.DataFrame(X, columns=feature_names)
    df[target_col] = y
    return df


def _compute_ir(counts: dict[int, int]) -> float:
    """Compute imbalance ratio: min_class / max_class."""
    if len(counts) < 2:
        return 1.0
    values = list(counts.values())
    return round(min(values) / max(values), 4)