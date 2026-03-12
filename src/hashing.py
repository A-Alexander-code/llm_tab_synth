"""
Functions to generate hashes that uniquely identify a dataset + parameter configuration.
Used by the model registry to detect if a model has already been trained.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Compute SHA-256 hash of a DataFrame.
    Columns are sorted so the hash is independent of column order.
    """
    df_sorted = df[sorted(df.columns)].reset_index(drop=True)
    content   = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_params(params: dict) -> str:
    """
    Compute SHA-256 hash of a dictionary of parameters.
    Sort keys to ensure order-independence.
    """
    content = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_experiment_hash(df: pd.DataFrame, params: dict) -> str:
    """
    Compute combined hash of dataset + parameters.
    Uniquely identifies a training experiment.
    """
    combined = hash_dataframe(df) + hash_params(params)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()