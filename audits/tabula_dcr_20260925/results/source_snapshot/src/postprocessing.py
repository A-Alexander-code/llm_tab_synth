"""Typed postprocessing utilities for synthetic tabular samples."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DatasetTypeSpec:
    dataset: str
    numeric_int: list[str] = field(default_factory=list)
    numeric_float: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    categorical_values: dict[str, list] = field(default_factory=dict)
    target: str = "class"

def load_type_spec(
    dataset: str,
    description_path: Path,
    raw_data_path: Path | None = None,
) -> DatasetTypeSpec:
    """Load dataset type metadata and infer categorical values when possible."""
    if not description_path.exists():
        logger.warning(
            f"datasets_description.json not found at {description_path}. "
            f"Postprocessing will be skipped for '{dataset}'."
        )
        return DatasetTypeSpec(dataset=dataset)

    with description_path.open("r", encoding="utf-8") as f:
        desc = json.load(f)

    if dataset not in desc:
        logger.warning(
            f"Dataset '{dataset}' not found in datasets_description.json. "
            f"Postprocessing will be skipped."
        )
        return DatasetTypeSpec(dataset=dataset)

    d = desc[dataset]

    spec = DatasetTypeSpec(
        dataset=dataset,
        numeric_int=d.get("numeric_int", []),
        numeric_float=d.get("numeric_float", []),
        binary=d.get("binary", []),
        categorical=d.get("categorical", []),
        target=d.get("target", "class"),
    )

    if spec.categorical and raw_data_path is not None and raw_data_path.exists():
        spec.categorical_values = _infer_categorical_values(
            spec.categorical, raw_data_path, spec.target
        )
    elif spec.categorical and (raw_data_path is None or not raw_data_path.exists()):
        logger.warning(
            f"Categorical columns found for '{dataset}' but raw data not available "
            f"at {raw_data_path}. Categorical postprocessing will use nearest-int rounding."
        )

    return spec


def _infer_categorical_values(
    categorical_cols: list[str],
    raw_data_path: Path,
    target: str,
) -> dict[str, list]:
    """Read sorted unique values for each categorical column from raw data."""
    try:
        raw_df = pd.read_csv(raw_data_path)
    except Exception as e:
        logger.error(f"Could not read raw data from {raw_data_path}: {e}")
        return {}

    cat_values: dict[str, list] = {}
    for col in categorical_cols:
        if col not in raw_df.columns:
            logger.warning(f"Categorical column '{col}' not found in raw data. Skipping.")
            continue
        unique_vals = sorted(raw_df[col].dropna().unique().tolist())
        cat_values[col] = unique_vals
        logger.info(f"  Categorical '{col}': valid values = {unique_vals}")

    return cat_values

def _process_numeric_int(series: pd.Series) -> pd.Series:
    """Round to the nearest integer and clip at zero."""
    return series.round(0).clip(lower=0).astype(float)


def _process_binary(series: pd.Series) -> pd.Series:
    """Round to binary bounds."""
    return series.round(0).clip(lower=0, upper=1).astype(float)


def _process_categorical(series: pd.Series, valid_values: list) -> pd.Series:
    """Map each value to the nearest valid category value."""
    if not valid_values:
        return series.round(0).astype(float)

    valid_arr = np.array(valid_values, dtype=float)

    def _nearest(val):
        if pd.isna(val):
            return valid_arr[0]
        dists = np.abs(valid_arr - float(val))
        return valid_arr[np.argmin(dists)]

    return series.apply(_nearest).astype(float)

def postprocess_synthetic(
    synth_df: pd.DataFrame,
    spec: DatasetTypeSpec,
) -> pd.DataFrame:
    """Apply type-aware postprocessing to a synthetic samples DataFrame."""
    if spec is None:
        return synth_df

    df = synth_df.copy()
    n_modified = 0

    for col in spec.numeric_int:
        if col in df.columns:
            before = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = _process_numeric_int(df[col])
            n_modified += 1
            logger.debug(f"  [int]   {col}: {before} → int (clip ≥ 0)")

    for col in spec.numeric_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_modified += 1
            logger.debug(f"  [float] {col}: kept as float")

    for col in spec.binary:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = _process_binary(df[col])
            n_modified += 1
            logger.debug(f"  [bin]   {col}: → {{0, 1}}")

    for col in spec.categorical:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid = spec.categorical_values.get(col, [])
            df[col] = _process_categorical(df[col], valid)
            n_modified += 1
            logger.debug(f"  [cat]   {col}: mapped to nearest of {valid}")

    logger.info(
        f"Postprocessing applied to '{spec.dataset}': "
        f"{n_modified} columns processed."
    )

    return df

def postprocess_summary(spec: DatasetTypeSpec) -> str:
    """Return a compact summary for logging."""
    return (
        f"int={len(spec.numeric_int)}, "
        f"float={len(spec.numeric_float)}, "
        f"binary={len(spec.binary)}, "
        f"categorical={len(spec.categorical)}"
        + (
            f" {list(spec.categorical_values.keys())}"
            if spec.categorical_values else ""
        )
    )
