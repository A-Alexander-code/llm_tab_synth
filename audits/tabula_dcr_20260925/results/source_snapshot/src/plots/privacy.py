import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Optional

from src.plots.common import draw_family_bands, finite_values, style_boxplot
from src.plots.data import require_columns
from src.plots.styles import (
    MODEL_COLORS,
    MODEL_FAMILIES,
    MODEL_LABELS,
    dataset_label,
)

PANEL_SPECS = (
    {
        "title": "DCR mean / holdout\nmean (→ 1)",
        "metric": "ratio_dcr_mean",
        "reference": 1.0,
        "yscale": "log",
    },
    {
        "title": "MIA AUC (→ 0.5)",
        "metric": "mia_auc",
        "reference": 0.5,
        "yscale": "linear",
    },
    {
        "title": "MIA advantage (↓)",
        "metric": "mia_advantage",
        "reference": 0.0,
        "yscale": "linear",
    },
    {
        "title": "NNAA (→ 0.5)",
        "metric": "nnaa",
        "reference": 0.5,
        "yscale": "linear",
    },
    {
        "title": "k-anonymity k=3 ratio (↑)",
        "metric": "kanon_k3_ratio",
        "reference": None,
        "yscale": "linear",
    },
    {
        "title": "k-anonymity k=5 ratio (↑)",
        "metric": "kanon_k5_ratio",
        "reference": None,
        "yscale": "linear",
    },
)

FONT_SIZES = {
    "default": 14,
    "suptitle": 19,
    "panel_title": 17,
    "dataset_label": 11,
    "x_tick": 14,
    "y_tick": 14,
    "model_legend": 15,
    "model_legend_title": 16,
    "family_legend": 14,
    "family_legend_title": 15,
    "reference_legend": 14,
}


def build_privacy_table(
    raw_data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
) -> pd.DataFrame:
    require_columns(
        raw_data,
        ("dataset", "model", "output_ir", "block", "metric", "mean"),
        "privacy plot",
    )

    data = raw_data[
        raw_data["dataset"].isin(datasets)
        & raw_data["model"].isin(model_order)
    ].copy()
    index_columns = ("dataset", "model", "output_ir")

    dcr = _pivot_metrics(
        data,
        "privacy_dcr",
        ("dcr_mean", "dcr_holdout_mean"),
        index_columns,
    )
    dcr["ratio_dcr_mean"] = (
        dcr["dcr_mean"] / dcr["dcr_holdout_mean"]
    ).replace([np.inf, -np.inf], np.nan)

    mia = _pivot_metrics(
        data,
        "privacy_mia",
        ("mia_auc", "mia_advantage"),
        index_columns,
    )
    nnaa = _pivot_metrics(
        data,
        "privacy_nnaa",
        ("nnaa",),
        index_columns,
    )
    kanon = _pivot_metrics(
        data,
        "privacy_kanonymity",
        ("kanon_k3_ratio", "kanon_k5_ratio"),
        index_columns,
    )

    privacy = dcr[list(index_columns) + ["ratio_dcr_mean"]]
    privacy = privacy.merge(
        mia[list(index_columns) + ["mia_auc", "mia_advantage"]],
        on=list(index_columns),
        how="outer",
    )
    privacy = privacy.merge(
        nnaa[list(index_columns) + ["nnaa"]],
        on=list(index_columns),
        how="outer",
    )
    privacy = privacy.merge(
        kanon[
            list(index_columns)
            + ["kanon_k3_ratio", "kanon_k5_ratio"]
        ],
        on=list(index_columns),
        how="outer",
    )

    if privacy.empty:
        raise ValueError(
            "No privacy metrics were found for the selected datasets."
        )

    privacy["dataset"] = pd.Categorical(
        privacy["dataset"],
        categories=datasets,
        ordered=True,
    )
    privacy["model"] = pd.Categorical(
        privacy["model"],
        categories=model_order,
        ordered=True,
    )

    return privacy.sort_values(["dataset", "model", "output_ir"])


def privacy_output_basename() -> str:
    return "figure_privacy_profiles_taxonomy"


def create_privacy_figure(
    data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
    include_title: bool,
) -> Figure:
    datasets_to_plot = tuple(
        dataset
        for dataset in datasets
        if dataset in data["dataset"].dropna().astype(str).unique()
    )

    if not datasets_to_plot:
        raise ValueError("No valid datasets were found after filtering.")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["default"],
            "font.family": "sans-serif",
            "axes.titlesize": FONT_SIZES["panel_title"],
            "axes.labelsize": FONT_SIZES["dataset_label"],
            "xtick.labelsize": FONT_SIZES["x_tick"],
            "ytick.labelsize": FONT_SIZES["y_tick"],
            "legend.fontsize": FONT_SIZES["model_legend"],
            "legend.title_fontsize": FONT_SIZES["model_legend_title"],
        }
    )

    figure, axes = plt.subplots(
        len(datasets_to_plot),
        len(PANEL_SPECS),
        figsize=(22, max(9, 2.85 * len(datasets_to_plot))),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    positions = np.arange(len(model_order))

    for row_index, dataset in enumerate(datasets_to_plot):
        for column_index, panel in enumerate(PANEL_SPECS):
            ax = axes[row_index, column_index]
            subset = data[data["dataset"] == dataset]
            _plot_panel(ax, subset, panel, positions, model_order)
            ax.tick_params(axis="y", labelsize=FONT_SIZES["y_tick"])

            if row_index == 0:
                ax.set_title(
                    panel["title"],
                    fontsize=FONT_SIZES["panel_title"],
                    loc="left",
                )

            if column_index == 0:
                ax.set_ylabel(
                    dataset_label(dataset),
                    fontsize=FONT_SIZES["dataset_label"],
                )

            if row_index == len(datasets_to_plot) - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels(
                    [MODEL_LABELS[model] for model in model_order],
                    rotation=60,
                    ha="right",
                    fontsize=FONT_SIZES["x_tick"],
                )
            else:
                ax.tick_params(axis="x", labelbottom=False)

    model_handles = [
        Patch(
            facecolor=MODEL_COLORS[model],
            edgecolor="black",
            label=MODEL_LABELS[model],
            alpha=0.90,
        )
        for model in model_order
    ]
    family_handles = [
        Patch(
            facecolor=family["color"],
            edgecolor="none",
            label=family["label"],
            alpha=0.75,
        )
        for family in MODEL_FAMILIES
    ]
    reference_handle = [
        Line2D(
            [0],
            [0],
            color="#4A4A4A",
            linestyle="--",
            linewidth=0.9,
            label="Reference value",
        )
    ]
    model_legend = figure.legend(
        handles=model_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=True,
        fontsize=FONT_SIZES["model_legend"],
        title="Generative model color code",
        title_fontsize=FONT_SIZES["model_legend_title"],
        bbox_to_anchor=(0.5, 0.07),
        borderpad=0.8,
        columnspacing=1.4,
        handlelength=1.6,
    )
    family_legend = figure.legend(
        handles=family_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZES["family_legend"],
        title="Taxonomic family bands",
        title_fontsize=FONT_SIZES["family_legend_title"],
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.6,
        handlelength=1.8,
    )
    figure.legend(
        handles=reference_handle,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=FONT_SIZES["reference_legend"],
        bbox_to_anchor=(0.5, -0.005),
        handlelength=1.8,
    )
    figure.add_artist(model_legend)
    figure.add_artist(family_legend)

    if include_title:
        figure.suptitle(
            "Privacy evaluation profiles by dataset",
            fontsize=FONT_SIZES["suptitle"],
            y=0.995,
        )

    figure.tight_layout(rect=[0, 0.15, 1, 0.975])
    return figure


def _pivot_metrics(
    data: pd.DataFrame,
    block: str,
    metrics: tuple[str, ...],
    index_columns: tuple[str, ...],
) -> pd.DataFrame:
    subset = data[
        (data["block"] == block)
        & data["metric"].isin(metrics)
    ]

    if subset.empty:
        return pd.DataFrame(columns=list(index_columns) + list(metrics))

    pivoted = (
        subset.pivot_table(
            index=list(index_columns),
            columns="metric",
            values="mean",
            aggfunc="mean",
            observed=False,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for metric in metrics:
        if metric not in pivoted.columns:
            pivoted[metric] = np.nan

    return pivoted


def _plot_panel(
    ax,
    subset: pd.DataFrame,
    panel: dict,
    positions: np.ndarray,
    model_order: tuple[str, ...],
) -> None:
    draw_family_bands(ax, model_order)
    values = [
        _boxplot_values(subset, model, panel["metric"])
        for model in model_order
    ]
    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    style_boxplot(boxplot, model_order)

    if panel["reference"] is not None:
        ax.axhline(
            panel["reference"],
            color="#4A4A4A",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )

    y_min, y_max = _subplot_limits(
        subset,
        panel["metric"],
        panel["reference"],
        panel["yscale"],
    )
    ax.set_yscale(panel["yscale"])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, len(model_order) - 0.5)
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.65,
        zorder=2,
    )


def _boxplot_values(
    subset: pd.DataFrame,
    model: str,
    metric: str,
) -> np.ndarray:
    values = finite_values(subset[subset["model"] == model][metric])
    return values if len(values) else np.array([np.nan])


def _subplot_limits(
    subset: pd.DataFrame,
    metric: str,
    reference: Optional[float],
    yscale: str,
) -> tuple[float, float]:
    values = finite_values(subset[metric])

    if reference is not None:
        values = np.concatenate([values, np.array([reference])])

    if yscale == "log":
        values = values[values > 0]

    if not len(values):
        return (1e-3, 1.0) if yscale == "log" else (0.0, 1.0)

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    if np.isclose(y_min, y_max):
        margin = 0.01 if y_max == 0 else abs(y_max) * 0.01
        y_min -= margin
        y_max += margin

    if yscale == "log":
        y_min *= 0.95
        y_max *= 1.05
    else:
        y_min = max(0.0, y_min) if y_min >= 0 else y_min
        y_max = min(1.0, y_max) if y_max <= 1.0 else y_max

    return y_min, y_max
