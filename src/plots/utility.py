import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.plots.common import draw_family_bands, finite_values, style_boxplot
from src.plots.data import require_columns
from src.plots.styles import (
    MODEL_COLORS,
    MODEL_FAMILIES,
    MODEL_LABELS,
    classifier_label,
    dataset_label,
)

METRICS = ("aucroc", "auprc", "f1")

METRIC_LABELS = {
    "aucroc": "AUC-ROC",
    "auprc": "AUC-PRC",
    "f1": "F1-score",
}

METRIC_DIRECTIONS = {
    "aucroc": "↑",
    "auprc": "↑",
    "f1": "↑",
}

FONT_SIZES = {
    "default": 14,
    "suptitle": 19,
    "subplot_title": 17,
    "dataset_label": 14,
    "x_tick": 14,
    "y_tick": 14,
    "model_legend": 15,
    "model_legend_title": 16,
    "family_legend": 14,
    "family_legend_title": 15,
}


def build_utility_table(
    raw_data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
    classifier: str,
) -> pd.DataFrame:
    require_columns(
        raw_data,
        (
            "dataset",
            "block",
            "classifier",
            "model",
            "metric",
            "mean",
            "output_ir",
        ),
        "utility plot",
    )

    valid_metrics = [f"tstr_{metric}" for metric in METRICS]
    data = raw_data[
        raw_data["dataset"].isin(datasets)
        & (raw_data["block"] == "utility_ml")
        & (raw_data["classifier"] == classifier)
        & raw_data["model"].isin(model_order)
        & raw_data["metric"].isin(valid_metrics)
    ].copy()

    if data.empty:
        raise ValueError(
            "No utility metrics were found for the selected parameters."
        )

    data["metric_base"] = data["metric"].str.replace(
        r"^tstr_",
        "",
        regex=True,
    )
    data["dataset"] = pd.Categorical(
        data["dataset"],
        categories=datasets,
        ordered=True,
    )
    data["model"] = pd.Categorical(
        data["model"],
        categories=model_order,
        ordered=True,
    )
    data["metric_base"] = pd.Categorical(
        data["metric_base"],
        categories=METRICS,
        ordered=True,
    )

    return data.sort_values(
        ["dataset", "metric_base", "model", "output_ir"]
    )


def utility_output_basename(classifier: str) -> str:
    return f"figure_utility_profiles_tstr_only_taxonomy_{classifier}"


def create_utility_figure(
    data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
    classifier: str,
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
            "axes.titlesize": FONT_SIZES["subplot_title"],
            "axes.labelsize": FONT_SIZES["dataset_label"],
            "xtick.labelsize": FONT_SIZES["x_tick"],
            "ytick.labelsize": FONT_SIZES["y_tick"],
            "legend.fontsize": FONT_SIZES["model_legend"],
            "legend.title_fontsize": FONT_SIZES["model_legend_title"],
        }
    )

    figure, axes = plt.subplots(
        len(datasets_to_plot),
        len(METRICS),
        figsize=(15, max(8, 3.0 * len(datasets_to_plot))),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    positions = np.arange(len(model_order))

    for row_index, dataset in enumerate(datasets_to_plot):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            subset = data[
                (data["dataset"] == dataset)
                & (data["metric_base"] == metric)
            ]

            draw_family_bands(ax, model_order)
            values = [
                _boxplot_values(subset, model)
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

            y_min, y_max = _subplot_limits(subset)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(-0.5, len(model_order) - 0.5)
            ax.grid(
                axis="y",
                linestyle=":",
                linewidth=0.5,
                alpha=0.65,
                zorder=2,
            )
            ax.tick_params(
                axis="y",
                labelsize=FONT_SIZES["y_tick"],
            )

            if row_index == 0:
                ax.set_title(
                    f"{METRIC_LABELS[metric]} "
                    f"({METRIC_DIRECTIONS[metric]})",
                    fontsize=FONT_SIZES["subplot_title"],
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
    model_legend = figure.legend(
        handles=model_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=True,
        fontsize=FONT_SIZES["model_legend"],
        title="Generative model color code",
        title_fontsize=FONT_SIZES["model_legend_title"],
        bbox_to_anchor=(0.5, 0.035),
        borderpad=0.8,
        columnspacing=1.4,
        handlelength=1.6,
    )
    figure.legend(
        handles=family_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZES["family_legend"],
        title="Taxonomic family bands",
        title_fontsize=FONT_SIZES["family_legend_title"],
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.6,
        handlelength=1.8,
    )
    figure.add_artist(model_legend)

    if include_title:
        figure.suptitle(
            "Downstream utility profiles by dataset | "
            f"Classifier: {classifier_label(classifier)}",
            fontsize=FONT_SIZES["suptitle"],
            y=0.995,
        )

    figure.tight_layout(rect=[0, 0.13, 1, 0.975])
    return figure


def _boxplot_values(subset: pd.DataFrame, model: str) -> np.ndarray:
    values = finite_values(subset[subset["model"] == model]["mean"])
    return values if len(values) else np.array([np.nan])


def _subplot_limits(subset: pd.DataFrame) -> tuple[float, float]:
    values = finite_values(subset["mean"])

    if not len(values):
        return 0.0, 1.0

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    if np.isclose(y_min, y_max):
        margin = 0.01 if y_max == 0 else abs(y_max) * 0.01
        y_min = max(0.0, y_min - margin)
        y_max = min(1.0, y_max + margin)

    return y_min, y_max
