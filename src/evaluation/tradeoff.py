"""Derived privacy-utility tradeoff metrics from existing evaluation rows."""

from __future__ import annotations

from typing import Any

import numpy as np

_BLOCK = "privacy_utility_tradeoff"
_PRIMARY_METRICS = [
    "tradeoff_utility_score",
    "tradeoff_privacy_score",
    "tradeoff_balance_score",
    "tradeoff_min_score",
]


def _ok(metric: str, value: float, classifier: str | None = None) -> dict[str, Any]:
    return {
        "block": _BLOCK,
        "metric": metric,
        "classifier": classifier,
        "value": value,
        "status": "ok",
        "reason": None,
    }


def _not_computable(
    metric: str,
    reason: str,
    classifier: str | None = None,
) -> dict[str, Any]:
    return {
        "block": _BLOCK,
        "metric": metric,
        "classifier": classifier,
        "value": float("nan"),
        "status": "not_computable",
        "reason": reason,
    }


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_classifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return str(value)


def _metric_map(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, str | None], float]:
    metric_values: dict[tuple[str, str | None], float] = {}
    for row in metric_rows:
        if row.get("status") != "ok":
            continue
        value = _safe_float(row.get("value"))
        if value is None:
            continue
        metric_values[(str(row.get("metric")), _normalize_classifier(row.get("classifier")))] = value
    return metric_values


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _collect_utility_components(
    metric_values: dict[tuple[str, str | None], float],
    classifier: str | None,
) -> list[float]:
    ratio_suffixes = ["aucroc", "auprc", "f1"]
    classifiers = {row_classifier for _, row_classifier in metric_values if row_classifier is not None}

    if classifier is not None:
        classifiers = {_normalize_classifier(classifier)}

    components: list[float] = []
    for row_classifier in sorted(classifiers):
        if row_classifier is None:
            continue
        for suffix in ratio_suffixes:
            trtr = metric_values.get((f"trtr_{suffix}", row_classifier))
            tstr = metric_values.get((f"tstr_{suffix}", row_classifier))
            if trtr is None or tstr is None or trtr <= 0:
                continue
            components.append(_clip01(tstr / trtr))

    return components


def _collect_privacy_components(metric_values: dict[tuple[str, str | None], float]) -> tuple[list[float], bool]:
    components: list[float] = []
    has_base_component = False

    exact_ratio = metric_values.get(("exact_matches_ratio", None))
    if exact_ratio is not None:
        components.append(_clip01(1.0 - exact_ratio))
        has_base_component = True

    dcr_p5 = metric_values.get(("dcr_p5", None))
    dcr_holdout_p5 = metric_values.get(("dcr_holdout_p5", None))
    if dcr_p5 is not None and dcr_holdout_p5 is not None and dcr_holdout_p5 > 0:
        components.append(_clip01(dcr_p5 / dcr_holdout_p5))
        has_base_component = True

    for risk_metric in [
        "record_sim_high_ratio_095",
        "semantic_sim_high_ratio_095",
        "memrisk_output_score",
        "ngram2_rare_reuse_ratio",
        "ngram3_rare_reuse_ratio",
    ]:
        risk_value = metric_values.get((risk_metric, None))
        if risk_value is None:
            continue
        components.append(_clip01(1.0 - risk_value))

    return components, has_base_component


def compute_tradeoff_metrics_from_rows(
    metric_rows: list[dict[str, Any]],
    *,
    utility_weight: float = 0.5,
    privacy_weight: float = 0.5,
    classifier: str | None = None,
) -> list[dict[str, Any]]:
    """Compute fixed-range privacy/utility tradeoff rows from metric rows."""
    if utility_weight < 0 or privacy_weight < 0:
        raise ValueError("utility_weight and privacy_weight must be non-negative")
    if utility_weight + privacy_weight <= 0:
        raise ValueError("utility_weight + privacy_weight must be > 0")

    metric_values = _metric_map(metric_rows)
    utility_components = _collect_utility_components(metric_values, classifier)
    privacy_components, has_base_privacy = _collect_privacy_components(metric_values)

    if not utility_components or not has_base_privacy or not privacy_components:
        return [
            _not_computable(metric_name, "missing_base_metrics", classifier)
            for metric_name in _PRIMARY_METRICS
        ]

    utility_score = float(np.mean(utility_components))
    privacy_score = float(np.mean(privacy_components))
    balance_score = float(
        (utility_weight * utility_score + privacy_weight * privacy_score)
        / (utility_weight + privacy_weight)
    )
    min_score = float(min(utility_score, privacy_score))

    return [
        _ok("tradeoff_utility_score", utility_score, classifier),
        _ok("tradeoff_privacy_score", privacy_score, classifier),
        _ok("tradeoff_balance_score", balance_score, classifier),
        _ok("tradeoff_min_score", min_score, classifier),
    ]
