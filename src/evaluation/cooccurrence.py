"""Extended co-occurrence fidelity metrics for synthetic tabular data."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

_BLOCK = "utility_cooccurrence"
_METRICS = [
    "cooc_l1_mean",
    "cooc_jsd_mean",
    "cooc_l2_frobenius_normed",
    "cooc_max_abs_diff",
    "cooc_top10_pair_overlap",
    "cooc_spearman_pairwise_strength",
]


class NotComputableError(RuntimeError):
    """Raised when a co-occurrence metric cannot be computed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _ok(metric: str, value: float) -> dict[str, Any]:
    return {
        "block": _BLOCK,
        "metric": metric,
        "classifier": None,
        "value": value,
        "status": "ok",
        "reason": None,
    }


def _not_computable(metric: str, reason: str) -> dict[str, Any]:
    return {
        "block": _BLOCK,
        "metric": metric,
        "classifier": None,
        "value": float("nan"),
        "status": "not_computable",
        "reason": reason,
    }


def _common_columns(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
) -> list[str]:
    return [
        column
        for column in real_train.columns
        if column != target_column and column in synthetic.columns
    ]


def _quantile_edges(series: pd.Series, n_bins: int) -> np.ndarray | None:
    clean = series.dropna().astype(float)
    if clean.empty:
        return None
    edges = np.unique(np.quantile(clean.values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 1:
        return None
    return edges


def _discretize_numeric(series: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"q{idx}" for idx in range(len(edges) - 1)]
    discretized = pd.cut(
        series.astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    return discretized.astype(object).where(pd.notna(discretized), "NA")


def _group_categorical(
    real_series: pd.Series,
    synth_series: pd.Series,
    max_categories: int,
) -> tuple[pd.Series, pd.Series] | None:
    real_values = real_series.astype(object).where(pd.notna(real_series), "NA").astype(str)
    synth_values = synth_series.astype(object).where(pd.notna(synth_series), "NA").astype(str)

    if real_values.nunique(dropna=False) <= 1:
        return None

    if real_values.nunique(dropna=False) > max_categories:
        top_categories = set(real_values.value_counts().head(max_categories - 1).index.tolist())
        real_values = real_values.where(real_values.isin(top_categories), "OTHER")
        synth_values = synth_values.where(synth_values.isin(top_categories), "OTHER")

    if real_values.nunique(dropna=False) <= 1:
        return None
    return real_values, synth_values


def discretize_for_cooccurrence(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str = "class",
    numeric_strategy: str = "quantile",
    n_bins: int = 4,
    max_categories: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discretize aligned columns using bins learned from the real training data."""
    if numeric_strategy != "quantile":
        raise ValueError(f"Unsupported numeric_strategy: {numeric_strategy}")
    if real_train.empty or synthetic.empty:
        raise NotComputableError("empty_input")

    common_columns = _common_columns(real_train, synthetic, target_column)
    if not common_columns:
        raise NotComputableError("no_common_columns")

    real_common = real_train.loc[:, common_columns].copy()
    synth_common = synthetic.loc[:, common_columns].copy()

    real_discrete: dict[str, pd.Series] = {}
    synth_discrete: dict[str, pd.Series] = {}

    for column in common_columns:
        if pd.api.types.is_numeric_dtype(real_common[column]):
            edges = _quantile_edges(real_common[column], n_bins)
            if edges is None:
                continue
            real_col = _discretize_numeric(real_common[column], edges)
            synth_col = _discretize_numeric(synth_common[column], edges)
            if real_col.nunique(dropna=False) <= 1:
                continue
            real_discrete[column] = real_col
            synth_discrete[column] = synth_col
        else:
            grouped = _group_categorical(
                real_series=real_common[column],
                synth_series=synth_common[column],
                max_categories=max_categories,
            )
            if grouped is None:
                continue
            real_col, synth_col = grouped
            real_discrete[column] = real_col
            synth_discrete[column] = synth_col

    if len(real_discrete) < 2:
        raise NotComputableError("insufficient_valid_features")

    return pd.DataFrame(real_discrete), pd.DataFrame(synth_discrete)


def compute_pairwise_joint_distributions(
    df_discrete: pd.DataFrame,
) -> dict[tuple[str, str], dict[tuple[str, str], float]]:
    """Compute normalized pairwise joint distributions for a discrete dataframe."""
    pairwise: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for left_column, right_column in combinations(df_discrete.columns.tolist(), 2):
        counts = (
            df_discrete.groupby([left_column, right_column], dropna=False)
            .size()
            .astype(float)
        )
        total = float(counts.sum())
        if total <= 0:
            continue
        pairwise[(left_column, right_column)] = {
            (str(left_value), str(right_value)): float(count / total)
            for (left_value, right_value), count in counts.items()
        }
    return pairwise


def _pair_vectors(
    real_dist: dict[tuple[str, str], float],
    synth_dist: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    support = sorted(set(real_dist) | set(synth_dist))
    real_vector = np.asarray([real_dist.get(key, 0.0) for key in support], dtype=float)
    synth_vector = np.asarray([synth_dist.get(key, 0.0) for key in support], dtype=float)
    return real_vector, synth_vector, support


def compute_cooccurrence_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str = "class",
    numeric_strategy: str = "quantile",
    n_bins: int = 4,
    top_k: int = 10,
) -> dict[str, float]:
    """Compare pairwise joint distributions between real and synthetic data."""
    real_discrete, synth_discrete = discretize_for_cooccurrence(
        real_train=real_train,
        synthetic=synthetic,
        target_column=target_column,
        numeric_strategy=numeric_strategy,
        n_bins=n_bins,
    )
    real_pairwise = compute_pairwise_joint_distributions(real_discrete)
    synth_pairwise = compute_pairwise_joint_distributions(synth_discrete)

    if not real_pairwise:
        raise NotComputableError("insufficient_valid_features")

    l1_values: list[float] = []
    jsd_values: list[float] = []
    l2_values: list[float] = []
    max_abs_values: list[float] = []
    real_strengths: list[float] = []
    synth_strengths: list[float] = []

    real_global_counter: Counter[tuple[str, str, str, str]] = Counter()
    synth_global_counter: Counter[tuple[str, str, str, str]] = Counter()

    for pair in sorted(real_pairwise):
        real_dist = real_pairwise[pair]
        synth_dist = synth_pairwise.get(pair, {})
        real_vector, synth_vector, support = _pair_vectors(real_dist, synth_dist)
        diff = real_vector - synth_vector

        l1_values.append(float(np.mean(np.abs(diff))))
        jsd_values.append(float(jensenshannon(real_vector, synth_vector)))
        l2_values.append(float(np.linalg.norm(diff, ord=2)))
        max_abs_values.append(float(np.max(np.abs(diff))))
        real_strengths.append(float(np.sum(real_vector ** 2)))
        synth_strengths.append(float(np.sum(synth_vector ** 2)))

        for key, real_value, synth_value in zip(support, real_vector, synth_vector, strict=True):
            real_global_counter[(pair[0], pair[1], key[0], key[1])] = real_value
            synth_global_counter[(pair[0], pair[1], key[0], key[1])] = synth_value

    top_k_real = {
        item for item, _ in real_global_counter.most_common(top_k)
    }
    top_k_synth = {
        item for item, _ in synth_global_counter.most_common(top_k)
    }
    overlap_denom = max(min(len(top_k_real), len(top_k_synth), top_k), 1)
    top_k_overlap = len(top_k_real & top_k_synth) / overlap_denom

    spearman_value = float("nan")
    if len(real_strengths) >= 2:
        correlation = spearmanr(real_strengths, synth_strengths).correlation
        if correlation is not None and np.isfinite(correlation):
            spearman_value = float(correlation)

    return {
        "cooc_l1_mean": float(np.mean(l1_values)),
        "cooc_jsd_mean": float(np.mean(jsd_values)),
        "cooc_l2_frobenius_normed": float(
            np.linalg.norm(np.asarray(l2_values, dtype=float), ord=2) / np.sqrt(len(l2_values))
        ),
        "cooc_max_abs_diff": float(np.max(max_abs_values)),
        "cooc_top10_pair_overlap": float(top_k_overlap),
        "cooc_spearman_pairwise_strength": spearman_value,
    }


def evaluate_cooccurrence_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    target_column: str = "class",
    numeric_strategy: str = "quantile",
    n_bins: int = 4,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return evaluator-style rows for co-occurrence fidelity metrics."""
    del dataset, model, seed, output_ir

    try:
        metrics = compute_cooccurrence_metrics(
            real_train=real_train,
            synthetic=synthetic,
            target_column=target_column,
            numeric_strategy=numeric_strategy,
            n_bins=n_bins,
            top_k=top_k,
        )
    except NotComputableError as exc:
        return [_not_computable(metric, exc.reason) for metric in _METRICS]

    rows: list[dict[str, Any]] = []
    for metric_name in _METRICS:
        value = metrics.get(metric_name, float("nan"))
        if np.isfinite(value):
            rows.append(_ok(metric_name, float(value)))
        else:
            rows.append(_not_computable(metric_name, "insufficient_pairwise_rankings"))
    return rows
