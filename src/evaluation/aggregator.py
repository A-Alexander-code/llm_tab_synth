"""Helpers to store raw metrics and build aggregated evaluation summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_RAW_COLUMNS = [
    "dataset", "model", "seed", "output_ir",
    "block", "metric", "classifier", "value",
]

_AGG_COLUMNS = [
    "dataset", "model", "output_ir",
    "block", "metric", "classifier",
    "mean", "std", "min", "max", "n_runs",
]


def build_rows(metric_rows: list[dict[str, Any]], 
               dataset: str, 
               model: str, 
               seed: int, 
               output_ir: float,) -> list[dict[str, Any]]:
    """Attach experiment identifiers to metric rows."""
    enriched = []
    for row in metric_rows:
        enriched.append({
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "output_ir": output_ir,
            "block": row.get("block", ""),
            "metric": row.get("metric", ""),
            "classifier": row.get("classifier", None),
            "value": row.get("value", None),
        })
    return enriched


def save_combination_csv(rows: list[dict[str, Any]], 
                         output_path: Path,) -> None:
    """Save raw metrics for one dataset/model/seed/IR combination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    df.to_csv(output_path, index=False)


def append_to_master(rows: list[dict[str, Any]], 
                     master_path: Path,) -> None:
    """Append rows to the dataset-level master CSV."""
    master_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    write_header = not master_path.exists()
    df.to_csv(master_path, mode="a", header=write_header, index=False)


def build_aggregated(master_path: Path) -> pd.DataFrame:
    """Aggregate numeric metric values from the master CSV."""
    if not master_path.exists():
        return pd.DataFrame(columns=_AGG_COLUMNS)

    df = pd.read_csv(master_path)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    group_cols = ["dataset", "model", "output_ir", "block", "metric", "classifier"]

    agg = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            n_runs="count",
        )
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def save_aggregated(master_path: Path, output_path: Path) -> pd.DataFrame:
    """Build and save the aggregated metrics CSV."""
    agg = build_aggregated(master_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    return agg
