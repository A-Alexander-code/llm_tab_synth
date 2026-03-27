"""Detection helpers for constant and near-constant columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_STD_THRESHOLD = 1e-12
_FREQ_THRESHOLD = 0.99
_BLOCK_DEGENERATE = "diagnostic_degenerate"


@dataclass
class ColumnStats:
    name: str
    nunique: int
    std: float
    max_freq: float


@dataclass
class DegenerateReport:
    constant: list[str]
    near_constant: list[str]
    stats: list[ColumnStats]

    @property
    def all_degenerate(self) -> list[str]:
        return list(dict.fromkeys(self.constant + self.near_constant))


@dataclass
class ValidColumnResult:
    valid_cols: list[str]
    excluded_cols: list[str]
    origin: dict[str, str]  
    real_report: DegenerateReport
    synth_report: DegenerateReport


def detect_degenerate_columns(df: pd.DataFrame) -> DegenerateReport:
    """Classify numeric columns as constant or near-constant."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    constant: list[str] = []
    near_constant: list[str] = []
    stats: list[ColumnStats] = []

    for col in numeric_cols:
        series = df[col].dropna()
        nunique = int(series.nunique())
        std = float(series.std()) if len(series) > 1 else 0.0
        max_freq = float(series.value_counts(normalize=True).iloc[0]) if len(series) > 0 else 1.0

        stats.append(ColumnStats(name=col, nunique=nunique, std=std, max_freq=max_freq))

        if nunique <= 1:
            constant.append(col)
        elif std < _STD_THRESHOLD or max_freq > _FREQ_THRESHOLD:
            near_constant.append(col)

    return DegenerateReport(constant=constant, near_constant=near_constant, stats=stats)


def get_valid_columns_for_statistical_metrics(real_df:  pd.DataFrame, 
                                              synth_df: pd.DataFrame,) -> ValidColumnResult:
    """Exclude columns that are degenerate in either dataset."""
    real_report = detect_degenerate_columns(real_df)
    synth_report = detect_degenerate_columns(synth_df)

    real_deg = set(real_report.all_degenerate)
    synth_deg = set(synth_report.all_degenerate)
    all_deg = real_deg | synth_deg

    origin: dict[str, str] = {}
    for col in all_deg:
        if col in real_deg and col in synth_deg:
            origin[col] = "both"
        elif col in real_deg:
            origin[col] = "real"
        else:
            origin[col] = "synthetic"

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in numeric_cols if c in synth_df.columns and c not in all_deg]
    excluded = [c for c in numeric_cols if c in all_deg]

    return ValidColumnResult(valid_cols=valid_cols, 
                             excluded_cols=excluded, 
                             origin=origin, 
                             real_report=real_report, 
                             synth_report=synth_report,)


def compute_degenerate_diagnostics(synth_df: pd.DataFrame, 
                                   real_df: pd.DataFrame, 
                                   vcr: ValidColumnResult | None = None,) -> list[dict[str, Any]]:
    """Compute diagnostic metrics for degenerate-column coverage."""
    if vcr is None:
        vcr = get_valid_columns_for_statistical_metrics(real_df, synth_df)

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    total = max(len(numeric_cols), 1)

    n_const_synth = len(vcr.synth_report.constant)
    n_nconst_synth = len(vcr.synth_report.near_constant)
    n_const_real = len(vcr.real_report.constant)
    n_nconst_real = len(vcr.real_report.near_constant)
    n_excluded = len(vcr.excluded_cols)
    n_valid = len(vcr.valid_cols)

    def _row(metric: str, value: float) -> dict[str, Any]:
        return {"block": _BLOCK_DEGENERATE, "metric": metric, "classifier": None,
                "value": value, "status": "ok", "reason": None}

    return [
        _row("synth_constant_cols", float(n_const_synth)),
        _row("synth_near_constant_cols", float(n_nconst_synth)),
        _row("real_constant_cols", float(n_const_real)),
        _row("real_near_constant_cols", float(n_nconst_real)),
        _row("excluded_cols_count", float(n_excluded)),
        _row("cfr_synth", n_const_synth / total),
        _row("ncfr_synth", n_nconst_synth / total),
        _row("vfr", n_valid / total),
    ]


def build_degenerate_audit_rows(vcr: ValidColumnResult, 
                                dataset: str, 
                                model: str, 
                                seed: int,  
                                output_ir: float,) -> list[dict[str, Any]]:
    """Build audit rows for the per-combination degenerate report."""
    real_stats = {s.name: s for s in vcr.real_report.stats}
    synth_stats = {s.name: s for s in vcr.synth_report.stats}
    all_cols = set(real_stats) | set(synth_stats)
    rows = []

    for col in sorted(all_cols):
        rs = real_stats.get(col)
        ss = synth_stats.get(col)

        if col in vcr.real_report.constant or col in vcr.synth_report.constant:
            issue_type = "constant"
        elif col in vcr.real_report.near_constant or col in vcr.synth_report.near_constant:
            issue_type = "near_constant"
        else:
            continue  

        rows.append({
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "output_ir": output_ir,
            "feature": col,
            "source": vcr.origin.get(col, "unknown"),
            "issue_type": issue_type,
            "nunique_real": rs.nunique if rs else None,
            "nunique_synth": ss.nunique if ss else None,
            "std_real": rs.std if rs else None,
            "std_synth": ss.std if ss else None,
            "excluded_from_metrics": col in vcr.excluded_cols,
        })

    return rows
