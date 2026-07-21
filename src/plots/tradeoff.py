import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import Optional

from src.plots.common import finite_values
from src.plots.data import require_columns
from src.plots.styles import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    classifier_label,
    dataset_label,
)

FIGURE_WIDTH = 20
ROW_HEIGHT = 2.85
COLUMN_SPACING = 0.35
PANEL_TITLE_PAD = 10
PANEL_TITLE_LINE_SPACING = 1.15
KIDNEY_X_PADDING_RATIO = 0.05
MODEL_LEGEND_Y = 0.035
REFERENCE_LEGEND_Y = -0.010
BOTTOM_LAYOUT_MARGIN = 0.15
TOP_LAYOUT_MARGIN = 0.94

DYNAMIC_Y_AXIS_SETTINGS = {
    "mia_advantage": {
        "padding_ratio": 0.06,
        "below_zero_margin": 0.01,
    },
    "kanon_k5_ratio": {
        "padding_ratio": 0.06,
        "below_zero_margin": 0.01,
    },
}

PANEL_SPECS = (
    {
        "title": "MIA AUC (→ 0.5)\nvs. TSTR AUC-ROC (↑)",
        "xmetric": "tstr_aucroc",
        "trtr_metric": "trtr_aucroc",
        "ymetric": "mia_auc",
        "xlabel": "TSTR AUC-ROC (↑)",
        "ylabel": "MIA AUC (→ 0.5)",
        "y_reference": 0.5,
    },
    {
        "title": "NNAA (→ 0.5)\nvs. TSTR AUC-ROC (↑)",
        "xmetric": "tstr_aucroc",
        "trtr_metric": "trtr_aucroc",
        "ymetric": "nnaa",
        "xlabel": "TSTR AUC-ROC (↑)",
        "ylabel": "NNAA (→ 0.5)",
        "y_reference": 0.5,
    },
    {
        "title": "MIA advantage (↓)\nvs. TSTR AUC-PRC (↑)",
        "xmetric": "tstr_auprc",
        "trtr_metric": "trtr_auprc",
        "ymetric": "mia_advantage",
        "xlabel": "TSTR AUC-PRC (↑)",
        "ylabel": "MIA advantage (↓)",
        "y_reference": 0.0,
    },
    {
        "title": "k-anonymity k=5 ratio (↑)\nvs. TSTR AUC-PRC (↑)",
        "xmetric": "tstr_auprc",
        "trtr_metric": "trtr_auprc",
        "ymetric": "kanon_k5_ratio",
        "xlabel": "TSTR AUC-PRC (↑)",
        "ylabel": "k-anonymity k=5 ratio (↑)",
        "y_reference": 1.0,
    },
)

FONT_SIZES = {
    "default": 14,
    "suptitle": 19,
    "panel_title": 17,
    "dataset_label": 11,
    "x_label": 14,
    "x_tick": 14,
    "y_tick": 14,
    "model_legend": 15,
    "model_legend_title": 16,
    "reference_legend": 14,
}


def build_tradeoff_table(
    raw_data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
    classifier: str,
    output_ir: float,
) -> pd.DataFrame:
    require_columns(
        raw_data,
        (
            "dataset",
            "model",
            "output_ir",
            "block",
            "classifier",
            "metric",
            "mean",
        ),
        "tradeoff plot",
    )

    data = raw_data[
        raw_data["dataset"].isin(datasets)
        & raw_data["model"].isin(model_order)
        & np.isclose(raw_data["output_ir"].astype(float), output_ir)
    ].copy()
    index_columns = ("dataset", "model", "output_ir")

    utility = _pivot_metrics(
        data[
            (data["block"] == "utility_ml")
            & (data["classifier"] == classifier)
        ],
        ("tstr_aucroc", "tstr_auprc", "trtr_aucroc", "trtr_auprc"),
        index_columns,
    )
    mia = _pivot_metrics(
        data[data["block"] == "privacy_mia"],
        ("mia_auc", "mia_advantage"),
        index_columns,
    )
    nnaa = _pivot_metrics(
        data[data["block"] == "privacy_nnaa"],
        ("nnaa",),
        index_columns,
    )
    kanon = _pivot_metrics(
        data[data["block"] == "privacy_kanonymity"],
        ("kanon_k5_ratio",),
        index_columns,
    )

    relationships = utility.merge(
        mia,
        on=list(index_columns),
        how="outer",
    )
    relationships = relationships.merge(
        nnaa,
        on=list(index_columns),
        how="outer",
    )
    relationships = relationships.merge(
        kanon,
        on=list(index_columns),
        how="outer",
    )

    if relationships.empty:
        raise ValueError(
            "No tradeoff metrics were found for the selected parameters."
        )

    relationships["dataset"] = pd.Categorical(
        relationships["dataset"],
        categories=datasets,
        ordered=True,
    )
    relationships["model"] = pd.Categorical(
        relationships["model"],
        categories=model_order,
        ordered=True,
    )

    return relationships.sort_values(["dataset", "model", "output_ir"])


def tradeoff_output_basename(classifier: str, output_ir: float) -> str:
    ir_text = str(output_ir).replace(".", "_")
    return f"figure_trade_off_privacy_utility_ir{ir_text}_{classifier}"


def create_tradeoff_figure(
    data: pd.DataFrame,
    datasets: tuple[str, ...],
    model_order: tuple[str, ...],
    classifier: str,
    output_ir: float,
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
            "axes.labelsize": FONT_SIZES["x_label"],
            "xtick.labelsize": FONT_SIZES["x_tick"],
            "ytick.labelsize": FONT_SIZES["y_tick"],
            "legend.fontsize": FONT_SIZES["model_legend"],
            "legend.title_fontsize": FONT_SIZES["model_legend_title"],
        }
    )

    figure, axes = plt.subplots(
        len(datasets_to_plot),
        len(PANEL_SPECS),
        figsize=(
            FIGURE_WIDTH,
            max(9, ROW_HEIGHT * len(datasets_to_plot)),
        ),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for row_index, dataset in enumerate(datasets_to_plot):
        for column_index, panel in enumerate(PANEL_SPECS):
            ax = axes[row_index, column_index]
            subset = data[data["dataset"] == dataset]
            _plot_panel(ax, subset, panel, model_order)

            if dataset == "kidney_disease":
                _expand_x_axis(ax, KIDNEY_X_PADDING_RATIO)

            if row_index == 0:
                ax.set_title(
                    panel["title"],
                    fontsize=FONT_SIZES["panel_title"],
                    loc="center",
                    pad=PANEL_TITLE_PAD,
                    linespacing=PANEL_TITLE_LINE_SPACING,
                )

            if row_index == len(datasets_to_plot) - 1:
                ax.set_xlabel(
                    panel["xlabel"],
                    fontsize=FONT_SIZES["x_label"],
                )

            if column_index == 0:
                ax.set_ylabel(
                    dataset_label(dataset),
                    fontsize=FONT_SIZES["dataset_label"],
                )
            else:
                ax.set_ylabel("")

            ax.tick_params(axis="x", labelsize=FONT_SIZES["x_tick"])
            ax.tick_params(axis="y", labelsize=FONT_SIZES["y_tick"])

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7.5,
            label=MODEL_LABELS[model],
        )
        for model in model_order
    ]
    reference_handles = [
        Line2D(
            [0],
            [0],
            color="#4A4A4A",
            linestyle="--",
            linewidth=0.9,
            label="TRTR utility baseline",
        ),
        Line2D(
            [0],
            [0],
            color="#7A7A7A",
            linestyle=":",
            linewidth=1.0,
            label="Privacy reference value",
        ),
    ]
    model_legend = figure.legend(
        handles=model_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=True,
        fontsize=FONT_SIZES["model_legend"],
        title="Generative model color and marker code",
        title_fontsize=FONT_SIZES["model_legend_title"],
        bbox_to_anchor=(0.5, MODEL_LEGEND_Y),
        borderpad=0.8,
        columnspacing=1.4,
        handlelength=1.6,
    )
    figure.legend(
        handles=reference_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZES["reference_legend"],
        bbox_to_anchor=(0.5, REFERENCE_LEGEND_Y),
        columnspacing=1.8,
        handlelength=2.2,
    )
    figure.add_artist(model_legend)

    if include_title:
        figure.suptitle(
            "Trade-off between downstream utility and privacy metrics"
            f" | IR = {output_ir}"
            f" | Classifier: {classifier_label(classifier)}",
            fontsize=FONT_SIZES["suptitle"],
            y=0.995,
        )

    figure.tight_layout(
        rect=[
            0,
            BOTTOM_LAYOUT_MARGIN,
            1,
            TOP_LAYOUT_MARGIN,
        ]
    )
    figure.subplots_adjust(wspace=COLUMN_SPACING)
    return figure


def _pivot_metrics(
    data: pd.DataFrame,
    metrics: tuple[str, ...],
    index_columns: tuple[str, ...],
) -> pd.DataFrame:
    subset = data[data["metric"].isin(metrics)]

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
    model_order: tuple[str, ...],
) -> None:
    x_column = panel["xmetric"]
    y_column = panel["ymetric"]
    plot_data = subset.dropna(subset=[x_column, y_column]).copy()

    for model in model_order:
        model_data = plot_data[plot_data["model"] == model]

        if model_data.empty:
            continue

        ax.scatter(
            model_data[x_column],
            model_data[y_column],
            s=58,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.90,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    x_reference = _trtr_baseline(subset, panel["trtr_metric"])

    if np.isfinite(x_reference):
        ax.axvline(
            x_reference,
            color="#4A4A4A",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

    if panel["y_reference"] is not None:
        ax.axhline(
            panel["y_reference"],
            color="#7A7A7A",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
        )

    x_min, x_max = _axis_limits(
        plot_data[x_column],
        reference_values=[x_reference],
        lower_bound=0.0,
        upper_bound=1.0,
    )

    if y_column in DYNAMIC_Y_AXIS_SETTINGS:
        settings = DYNAMIC_Y_AXIS_SETTINGS[y_column]
        y_min, y_max = _dynamic_y_limits(
            plot_data[y_column],
            settings["padding_ratio"],
            settings["below_zero_margin"],
        )
    else:
        y_min, y_max = _axis_limits(
            plot_data[y_column],
            reference_values=[panel["y_reference"]],
            lower_bound=0.0,
            upper_bound=1.0,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(
        linestyle=":",
        linewidth=0.5,
        alpha=0.65,
        zorder=0,
    )


def _trtr_baseline(subset: pd.DataFrame, metric: str) -> float:
    values = finite_values(subset[metric])
    return float(np.mean(values)) if len(values) else np.nan


def _axis_limits(
    values,
    reference_values=None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> tuple[float, float]:
    finite = finite_values(values)

    if reference_values is not None:
        references = finite_values(reference_values)
        finite = np.concatenate([finite, references])

    if not len(finite):
        return 0.0, 1.0

    value_min = float(np.min(finite))
    value_max = float(np.max(finite))

    if np.isclose(value_min, value_max):
        margin = 0.01 if value_max == 0 else abs(value_max) * 0.01
        value_min -= margin
        value_max += margin

    span = value_max - value_min
    padding = 0.04 * span if span > 0 else 0.01
    value_min -= padding
    value_max += padding

    if lower_bound is not None:
        value_min = max(lower_bound, value_min)

    if upper_bound is not None:
        value_max = min(upper_bound, value_max)

    return value_min, value_max


def _dynamic_y_limits(
    values,
    padding_ratio: float,
    below_zero_margin: float,
) -> tuple[float, float]:
    finite = finite_values(values)

    if not len(finite):
        return -below_zero_margin, 1.0

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    span = data_max - data_min

    if np.isclose(span, 0.0):
        span = max(abs(data_max), 0.01)

    padding = padding_ratio * span
    y_min = min(data_min - padding, -below_zero_margin)
    y_max = data_max + padding

    if np.isclose(y_min, y_max):
        y_max = y_min + 0.01

    return y_min, y_max


def _expand_x_axis(ax, padding_ratio: float) -> None:
    x_min, x_max = ax.get_xlim()
    span = x_max - x_min

    if not np.isfinite(span) or span <= 0:
        return

    padding = padding_ratio * span
    ax.set_xlim(x_min - padding, x_max + padding)
