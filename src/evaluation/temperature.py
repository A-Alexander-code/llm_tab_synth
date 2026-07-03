"""Sensitivity metrics across sampling temperatures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_BLOCK = "privacy_temperature_sensitivity"


def _ok(
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    metric: str,
    classifier: str | None,
    value: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "output_ir": output_ir,
        "block": _BLOCK,
        "metric": metric,
        "classifier": classifier,
        "value": value,
        "status": "ok",
        "reason": None,
    }


def _not_computable(
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    metric: str,
    classifier: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "output_ir": output_ir,
        "block": _BLOCK,
        "metric": metric,
        "classifier": classifier,
        "value": float("nan"),
        "status": "not_computable",
        "reason": reason,
    }


def _tracked_classifier_label(metric_name: str, classifier: str | None) -> str | None:
    if metric_name == "memrisk_output_score":
        return classifier
    if classifier:
        return f"{metric_name}|{classifier}"
    return metric_name


def _metric_names_for_tracked(metric_name: str) -> list[str]:
    if metric_name == "memrisk_output_score":
        return [
            "temp_memrisk_range",
            "temp_memrisk_slope",
            "temp_memrisk_max",
            "temp_memrisk_min",
        ]
    return [
        "temp_metric_range",
        "temp_metric_std",
        "temp_metric_cv",
        "temp_metric_slope",
    ]


def _build_not_rows(
    group: pd.Series,
    metric_name: str,
    classifier: str | None,
    reason: str,
) -> list[dict[str, Any]]:
    return [
        _not_computable(
            dataset=str(group["dataset"]),
            model=str(group["model"]),
            seed=int(group["seed"]),
            output_ir=float(group["output_ir"]),
            metric=temp_metric,
            classifier=_tracked_classifier_label(metric_name, classifier),
            reason=reason,
        )
        for temp_metric in _metric_names_for_tracked(metric_name)
    ]


def compute_temperature_sensitivity_from_aggregated(
    aggregated_df: pd.DataFrame,
    *,
    metrics_to_track: list[str],
    temperatures: list[float],
) -> pd.DataFrame:
    """Compute fixed rows summarizing metric sensitivity to sampling temperature."""
    if aggregated_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "seed",
                "output_ir",
                "block",
                "metric",
                "classifier",
                "value",
                "status",
                "reason",
            ]
        )
    if "temperature" not in aggregated_df.columns:
        raise ValueError("aggregated_df must contain a temperature column")

    work_df = aggregated_df.copy()
    if "status" not in work_df.columns:
        work_df["status"] = "ok"
    if "reason" not in work_df.columns:
        work_df["reason"] = None
    if "classifier" not in work_df.columns:
        work_df["classifier"] = None

    grouped = work_df.groupby(
        ["dataset", "model", "seed", "output_ir", "metric", "classifier"],
        dropna=False,
    )

    rows: list[dict[str, Any]] = []
    requested_temperatures = [float(value) for value in temperatures]
    requested_set = set(requested_temperatures)

    for group_key, group_df in grouped:
        dataset, model, seed, output_ir, metric_name, classifier = group_key
        if metric_name not in metrics_to_track:
            continue

        group_head = pd.Series(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "output_ir": output_ir,
            }
        )
        metric_names = _metric_names_for_tracked(str(metric_name))
        label = _tracked_classifier_label(str(metric_name), classifier)

        available_temperatures = set(group_df["temperature"].astype(float).tolist())
        missing_temperatures = [
            temperature
            for temperature in requested_temperatures
            if temperature not in available_temperatures
        ]
        if missing_temperatures:
            reason = f"missing_temperature_samples:{missing_temperatures[0]}"
            rows.extend(_build_not_rows(group_head, str(metric_name), classifier, reason))
            continue

        requested_rows = group_df[group_df["temperature"].astype(float).isin(requested_set)].copy()
        requested_rows["value"] = pd.to_numeric(requested_rows["value"], errors="coerce")
        if requested_rows["value"].isna().all():
            rows.extend(_build_not_rows(group_head, str(metric_name), classifier, "all_nan_values"))
            continue

        valid_rows = requested_rows[requested_rows["status"] == "ok"].copy()
        valid_rows["value"] = pd.to_numeric(valid_rows["value"], errors="coerce")
        valid_rows = valid_rows.dropna(subset=["value"])

        if len(valid_rows) < 2:
            rows.extend(_build_not_rows(group_head, str(metric_name), classifier, "insufficient_temperatures"))
            continue

        valid_rows = valid_rows.sort_values("temperature")
        values = valid_rows["value"].to_numpy(dtype=float)
        temps = valid_rows["temperature"].to_numpy(dtype=float)

        if metric_name == "memrisk_output_score":
            slope = float(np.polyfit(temps, values, 1)[0])
            rows.extend(
                [
                    _ok(dataset, model, int(seed), float(output_ir), "temp_memrisk_range", label, float(np.max(values) - np.min(values))),
                    _ok(dataset, model, int(seed), float(output_ir), "temp_memrisk_slope", label, slope),
                    _ok(dataset, model, int(seed), float(output_ir), "temp_memrisk_max", label, float(np.max(values))),
                    _ok(dataset, model, int(seed), float(output_ir), "temp_memrisk_min", label, float(np.min(values))),
                ]
            )
            continue

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        slope = float(np.polyfit(temps, values, 1)[0])

        rows.append(
            _ok(dataset, model, int(seed), float(output_ir), metric_names[0], label, float(np.max(values) - np.min(values)))
        )
        rows.append(
            _ok(dataset, model, int(seed), float(output_ir), metric_names[1], label, std_value)
        )
        if np.isclose(mean_value, 0.0):
            rows.append(
                _not_computable(
                    dataset,
                    model,
                    int(seed),
                    float(output_ir),
                    metric_names[2],
                    label,
                    "division_by_zero",
                )
            )
        else:
            rows.append(
                _ok(dataset, model, int(seed), float(output_ir), metric_names[2], label, float(std_value / abs(mean_value)))
            )
        rows.append(
            _ok(dataset, model, int(seed), float(output_ir), metric_names[3], label, slope)
        )

    return pd.DataFrame(rows)
