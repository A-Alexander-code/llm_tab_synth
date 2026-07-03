"""Extended privacy diagnostics based on synthetic output similarity patterns."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

_BLOCK_RECORD = "privacy_record_similarity"
_BLOCK_NGRAM = "privacy_ngram_overlap"
_BLOCK_SEMANTIC = "privacy_semantic_similarity"
_BLOCK_MEMRISK = "privacy_memorization_proxy"

_RECORD_METRICS = [
    "record_sim_max_mean",
    "record_sim_max_median",
    "record_sim_max_p95",
    "record_sim_max_p99",
    "record_sim_high_ratio_090",
    "record_sim_high_ratio_095",
    "record_sim_high_ratio_099",
    "record_sim_exactish_ratio",
]

_SEMANTIC_METRICS = [
    "semantic_sim_max_mean",
    "semantic_sim_max_median",
    "semantic_sim_max_p95",
    "semantic_sim_max_p99",
    "semantic_sim_high_ratio_090",
    "semantic_sim_high_ratio_095",
    "semantic_sim_high_ratio_099",
]

_MEMRISK_METRICS = [
    "memrisk_output_score",
    "memrisk_components_available",
]


class NotComputableError(RuntimeError):
    """Raised when a metric cannot be computed for a well-defined reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CanonicalData:
    """Canonical representation of aligned real and synthetic rows."""

    real_tokens: list[list[str]]
    synth_tokens: list[list[str]]


def _ok(block: str, metric: str, value: float, classifier: str | None = None) -> dict[str, Any]:
    return {
        "block": block,
        "metric": metric,
        "classifier": classifier,
        "value": value,
        "status": "ok",
        "reason": None,
    }


def _not_computable(
    block: str,
    metric: str,
    reason: str,
    classifier: str | None = None,
) -> dict[str, Any]:
    return {
        "block": block,
        "metric": metric,
        "classifier": classifier,
        "value": float("nan"),
        "status": "not_computable",
        "reason": reason,
    }


def _expected_ngram_metrics(n_values: tuple[int, ...]) -> list[str]:
    metrics: list[str] = []
    suffixes = [
        "overlap_mean",
        "overlap_median",
        "overlap_p95",
        "train_vocab_reuse_ratio",
        "synthetic_vocab_seen_ratio",
        "rare_reuse_ratio",
        "rare_reuse_per_record_mean",
    ]
    for n_value in n_values:
        metrics.extend([f"ngram{n_value}_{suffix}" for suffix in suffixes])
    return metrics


def _select_common_columns(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
) -> list[str]:
    return [
        column
        for column in real_train.columns
        if column != target_column and column in synthetic.columns
    ]


def _validated_common_frames(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if real_train.empty or synthetic.empty:
        raise NotComputableError("empty_input")

    common_columns = _select_common_columns(real_train, synthetic, target_column)
    if not common_columns:
        raise NotComputableError("no_common_columns")

    real_common = real_train.loc[:, common_columns].copy()
    synth_common = synthetic.loc[:, common_columns].copy()
    return real_common, synth_common, common_columns


def _format_scalar(value: Any, decimals: int) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        rounded = round(float(value), decimals)
        return str(rounded)
    return str(value)


def _quantile_edges(series: pd.Series, n_bins: int) -> np.ndarray | None:
    clean = series.dropna().astype(float)
    if clean.empty:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(clean.values, quantiles))
    if len(edges) <= 1:
        return None
    return edges


def _bin_numeric_series(series: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"bin_{idx:02d}" for idx in range(len(edges) - 1)]
    binned = pd.cut(
        series.astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    return binned.astype(object).where(pd.notna(binned), "NA")


def canonicalize_tabular_rows(
    df: pd.DataFrame,
    *,
    target_column: str = "class",
    decimals: int = 6,
    numeric_strategy: str = "round",
    binning_reference: pd.DataFrame | None = None,
    n_bins: int = 10,
) -> list[list[str]]:
    """Convert aligned tabular rows into deterministic feature=value tokens."""
    if numeric_strategy not in {"round", "quantile_bin"}:
        raise ValueError(f"Unsupported numeric_strategy: {numeric_strategy}")

    feature_columns = [column for column in df.columns if column != target_column]
    work_df = df.loc[:, feature_columns].copy()
    reference_df = (
        binning_reference.loc[:, feature_columns].copy()
        if binning_reference is not None
        else work_df.copy()
    )

    rows: list[list[str]] = []
    numeric_columns = set(work_df.select_dtypes(include=[np.number]).columns.tolist())

    cached_bins: dict[str, np.ndarray | None] = {}
    if numeric_strategy == "quantile_bin":
        for column in feature_columns:
            if column in numeric_columns:
                cached_bins[column] = _quantile_edges(reference_df[column], n_bins)

    for _, row in work_df.iterrows():
        tokens: list[str] = []
        for column in feature_columns:
            value = row[column]
            if column in numeric_columns:
                if pd.isna(value):
                    token_value = "NA"
                elif numeric_strategy == "round":
                    token_value = _format_scalar(value, decimals)
                else:
                    edges = cached_bins.get(column)
                    if edges is None:
                        token_value = _format_scalar(value, decimals)
                    else:
                        cut_result = _bin_numeric_series(pd.Series([value]), edges).iloc[0]
                        token_value = str(cut_result)
            else:
                token_value = "NA" if pd.isna(value) else str(value)
            tokens.append(f"{column}={token_value}")
        rows.append(tokens)

    return rows


def rows_to_strings(tokenized_rows: list[list[str]]) -> list[str]:
    """Join tokenized rows into canonical tabular strings."""
    return ["|".join(tokens) for tokens in tokenized_rows]


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _build_canonical_data(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str,
    numeric_strategy: str,
    decimals: int,
    n_bins: int,
) -> CanonicalData:
    real_common, synth_common, common_columns = _validated_common_frames(
        real_train=real_train,
        synthetic=synthetic,
        target_column=target_column,
    )

    real_aligned = real_common.loc[:, common_columns].copy()
    synth_aligned = synth_common.loc[:, common_columns].copy()

    real_tokens = canonicalize_tabular_rows(
        real_aligned,
        target_column=target_column,
        decimals=decimals,
        numeric_strategy=numeric_strategy,
        binning_reference=real_aligned,
        n_bins=n_bins,
    )
    synth_tokens = canonicalize_tabular_rows(
        synth_aligned,
        target_column=target_column,
        decimals=decimals,
        numeric_strategy=numeric_strategy,
        binning_reference=real_aligned,
        n_bins=n_bins,
    )
    return CanonicalData(real_tokens=real_tokens, synth_tokens=synth_tokens)


def _max_jaccard_scores(
    real_token_rows: list[list[str]],
    synth_token_rows: list[list[str]],
) -> np.ndarray:
    real_sets = [set(tokens) for tokens in real_token_rows]
    synth_sets = [set(tokens) for tokens in synth_token_rows]

    token_index: dict[str, list[int]] = defaultdict(list)
    for row_idx, token_set in enumerate(real_sets):
        for token in token_set:
            token_index[token].append(row_idx)

    scores: list[float] = []
    real_lengths = [len(token_set) for token_set in real_sets]

    for synth_set in synth_sets:
        if not synth_set:
            scores.append(0.0)
            continue

        intersections: dict[int, int] = defaultdict(int)
        for token in synth_set:
            for row_idx in token_index.get(token, []):
                intersections[row_idx] += 1

        if not intersections:
            scores.append(0.0)
            continue

        synth_size = len(synth_set)
        max_score = 0.0
        for row_idx, overlap in intersections.items():
            union = synth_size + real_lengths[row_idx] - overlap
            if union <= 0:
                continue
            score = overlap / union
            if score > max_score:
                max_score = score
        scores.append(float(max_score))

    return np.asarray(scores, dtype=float)


def compute_record_similarity_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str = "class",
    decimals: int = 6,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Compute max token-level Jaccard similarity against real training rows."""
    canonical = _build_canonical_data(
        real_train=real_train,
        synthetic=synthetic,
        target_column=target_column,
        numeric_strategy="round",
        decimals=decimals,
        n_bins=10,
    )
    scores = _max_jaccard_scores(canonical.real_tokens, canonical.synth_tokens)

    metrics: dict[str, float] = {
        "record_sim_max_mean": float(np.mean(scores)),
        "record_sim_max_median": float(np.median(scores)),
        "record_sim_max_p95": _percentile(scores, 95),
        "record_sim_max_p99": _percentile(scores, 99),
        "record_sim_exactish_ratio": float(np.mean(scores >= 0.999)),
    }
    for threshold in thresholds:
        metrics[f"record_sim_high_ratio_{int(round(threshold * 100)):03d}"] = float(
            np.mean(scores >= threshold)
        )
    return metrics


def extract_row_ngrams(
    tokenized_row: list[str],
    *,
    n: int = 2,
    mode: str = "combinations",
    max_ngrams_per_row: int | None = None,
) -> set[tuple[str, ...]]:
    """Extract deterministic tabular n-grams from a tokenized row."""
    if n <= 0:
        raise ValueError("n must be positive")
    if mode != "combinations":
        raise ValueError(f"Unsupported mode: {mode}")
    if len(tokenized_row) < n:
        return set()

    ngrams = list(combinations(tokenized_row, n))
    if max_ngrams_per_row is not None:
        ngrams = ngrams[:max_ngrams_per_row]
    return set(ngrams)


def compute_ngram_overlap_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str = "class",
    n_values: tuple[int, ...] = (2, 3),
    numeric_strategy: str = "quantile_bin",
    n_bins: int = 10,
    rare_max_count: int = 2,
) -> dict[str, float]:
    """Measure train/synthetic overlap over tabular token n-grams."""
    canonical = _build_canonical_data(
        real_train=real_train,
        synthetic=synthetic,
        target_column=target_column,
        numeric_strategy=numeric_strategy,
        decimals=6,
        n_bins=n_bins,
    )

    metrics: dict[str, float] = {}
    for n_value in n_values:
        real_row_ngrams = [extract_row_ngrams(tokens, n=n_value) for tokens in canonical.real_tokens]
        synth_row_ngrams = [extract_row_ngrams(tokens, n=n_value) for tokens in canonical.synth_tokens]

        if not any(real_row_ngrams) or not any(synth_row_ngrams):
            raise NotComputableError("insufficient_features_for_ngrams")

        real_counter = Counter(
            ngram
            for row_ngrams in real_row_ngrams
            for ngram in row_ngrams
        )
        synth_counter = Counter(
            ngram
            for row_ngrams in synth_row_ngrams
            for ngram in row_ngrams
        )

        real_vocab = set(real_counter)
        synth_vocab = set(synth_counter)
        rare_vocab = {ngram for ngram, count in real_counter.items() if count <= rare_max_count}

        

    return metrics

