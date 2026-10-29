"""Privacy figures ported from notebook cell 4.

Seed validation, summaries, axes and layout follow plotting_results_ver4.ipynb.
No notebook execution or notebook dependency is required.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from src.plots.data import require_columns
from src.plots.styles import DATASET_LABELS, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS


def run(datasets, classifiers, project_root, png_dpi=300, include_title=False):
    """Generate the notebook figures for the selected datasets and classifiers."""

    PROJECT_ROOT = project_root
    CSV_PATH = PROJECT_ROOT / "results" / "results_long.csv"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "privacy_profiles_final"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    Y_PADDING_RATIO = 0.20
    MIN_Y_PADDING = 0.05
    SMALL_METRIC_PADDING = 0.008

    DATASETS = list(datasets)
    dataset_labels = {d: DATASET_LABELS.get(d, d.replace("_", " ").title()) for d in DATASETS}





    MODEL_FAMILIES = [
        {
            "label": "LLM-based generators",
            "models": ["great", "tabula", "realtabformer"],
            "color": "#DDEBFF",
        },
        {
            "label": "Comparison generators",
            "models": ["ctgan", "copulagan", "tvae", "gaussian_copula", "mostlyai"],
            "color": "#FFF1CC",
        },
    ]

    METRICS = {
        "mia_auroc": {
            "label": "MIA AUCROC",
            "component": "mia",
            "fold": "mean",
            "reference": 0.5,
        },
        "mia_tpr_target_fpr_1pct_calibrated": {
            "label": "MIA TPR@1% FPR",
            "component": "mia",
            "fold": "aggregate",
            "reference": None,
        },
        "mia_observed_fpr": {
            "label": "MIA Observed FPR",
            "component": "mia",
            "fold": "aggregate",
            "reference": 0.01,
        },
        "nnaa_overall": {
            "label": "NNAA",
            "component": "nnaa",
            "fold": None,
            "reference": 0.5,
        },
        "neighborhood_k5_distinct_groups_ratio": {
            "label": "Neighborhood-based k-Anonymity\n(k=5 distinct groups)",
            "component": "neighborhood",
            "fold": None,
            "reference": None,
        },
        "detectability_auroc": {
            "label": "Detectability AUCROC",
            "component": "detectability",
            "fold": "mean",
            "reference": 0.5,
        },
    }

    SMALL_METRICS = {
        "mia_observed_fpr",
        "mia_tpr_target_fpr_1pct_calibrated",
        "neighborhood_k5_distinct_groups_ratio",
    }

    LIMITED_METRICS = {
        "mia_tpr_target_fpr_1pct_calibrated",
        "mia_observed_fpr",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(CSV_PATH, low_memory=False)

    required_columns = {
        "dataset", "seed", "generador", "component",
        "fold", "metric", "value", "status"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fold"] = df["fold"].astype("string")
    df["status"] = df["status"].astype("string").str.lower()

    df = df.loc[
        df["dataset"].isin(DATASETS)
        & df["generador"].isin(MODEL_ORDER)
        & df["metric"].isin(METRICS)
    ].copy()

    metric_masks = []

    for metric, config in METRICS.items():

        mask = (
            df["metric"].eq(metric)
            & df["component"].eq(config["component"])
        )

        if config["fold"] is None:
            mask &= df["fold"].isna()
        else:
            mask &= df["fold"].eq(config["fold"]).fillna(False)

        metric_masks.append(mask)

    selection_mask = metric_masks[0]

    for mask in metric_masks[1:]:
        selection_mask |= mask

    df = df.loc[selection_mask].copy()

    valid_status = (
        df["status"].eq("ok")
        | (
            df["metric"].isin(LIMITED_METRICS)
            & df["status"].eq("insufficient_resolution")
        )
    )

    df = df.loc[valid_status].copy()

    invalid_values = (
        df["value"].isna()
        | ~np.isfinite(df["value"])
        | df["value"].lt(0)
        | df["value"].gt(1)
    )

    if invalid_values.any():
        raise ValueError(
            "Invalid metric values detected:\n"
            + df.loc[
                invalid_values,
                ["dataset", "seed", "generador", "metric", "value"]
            ].head(15).to_string(index=False)
        )

    duplicate_columns = ["dataset", "seed", "generador", "metric"]

    duplicates = df.duplicated(subset=duplicate_columns, keep=False)

    if duplicates.any():
        raise ValueError(
            "Duplicate privacy observations detected:\n"
            + df.loc[duplicates, duplicate_columns].head(15).to_string(index=False)
        )

    expected_index = pd.MultiIndex.from_product(
        [DATASETS, MODEL_ORDER, METRICS.keys()],
        names=["dataset", "generador", "metric"]
    )

    observed_index = pd.MultiIndex.from_frame(
        df[["dataset", "generador", "metric"]].drop_duplicates()
    )

    missing = expected_index.difference(observed_index)

    if len(missing):
        raise ValueError(
            f"Missing {len(missing)} dataset-generator-metric combinations:\n"
            + str(missing.tolist()[:15])
        )

    seed_counts = (
        df.groupby(["dataset", "generador", "metric"])["seed"].nunique()
    )

    if (seed_counts != 5).any():
        raise ValueError(
            "Some combinations do not contain five seeds:\n"
            + seed_counts[seed_counts != 5].to_string()
        )

    summary = (
        df.groupby(["dataset", "metric", "generador"])
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            minimum=("value", "min"),
            maximum=("value", "max"),
            n=("seed", "nunique"),
            limited_seeds=(
                "status",
                lambda x: x.eq("insufficient_resolution").sum()
            )
        )
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0)


    def get_metric_y_limits(data):

        limits = {}

        for metric, config in METRICS.items():

            values = data.loc[
                data["metric"].eq(metric),
                "value"
            ].to_numpy(dtype=float)

            reference = config["reference"]

            if reference is not None:
                values = np.append(values, reference)

            ymin = float(np.min(values))
            ymax = float(np.max(values))

            base_padding = (
                SMALL_METRIC_PADDING if metric in SMALL_METRICS
                else MIN_Y_PADDING
            )

            padding = max(base_padding, (ymax - ymin) * Y_PADDING_RATIO)

            limits[metric] = (ymin - padding, ymax + padding)

        return limits


    def build_box_stats(subset, model):

        row = subset.loc[subset["generador"].eq(model)]

        if row.empty:
            return {
                "med": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "whislo": np.nan,
                "whishi": np.nan,
                "fliers": [],
            }

        row = row.iloc[0]

        return {
            "med": row["mean"],
            "q1": max(row["mean"] - row["std"], row["minimum"]),
            "q3": min(row["mean"] + row["std"], row["maximum"]),
            "whislo": row["minimum"],
            "whishi": row["maximum"],
            "fliers": [],
        }


    def draw_family_bands(ax):

        for family in MODEL_FAMILIES:

            indices = [MODEL_ORDER.index(model) for model in family["models"]]

            ax.axvspan(
                min(indices) - 0.5,
                max(indices) + 0.5,
                facecolor=family["color"],
                alpha=0.65,
                zorder=0
            )

        ax.axvline(2.5, color="white", linewidth=2, zorder=1)


    def plot_privacy_profiles():

        y_limits = get_metric_y_limits(df)

        fig, axes = plt.subplots(
            len(DATASETS),
            len(METRICS),
            figsize=(27, 15),
            sharex=True,
            squeeze=False
        )

        positions = np.arange(len(MODEL_ORDER))

        for row_idx, dataset in enumerate(DATASETS):

            for col_idx, (metric, config) in enumerate(METRICS.items()):

                ax = axes[row_idx, col_idx]

                subset = summary.loc[
                    summary["dataset"].eq(dataset)
                    & summary["metric"].eq(metric)
                ]

                draw_family_bands(ax)

                stats = [build_box_stats(subset, model) for model in MODEL_ORDER]

                bp = ax.bxp(
                    stats,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False
                )

                for box, model in zip(bp["boxes"], MODEL_ORDER):
                    box.set(
                        facecolor=MODEL_COLORS[model],
                        edgecolor="black",
                        alpha=0.9,
                        linewidth=0.8
                    )

                for item in bp["whiskers"] + bp["caps"]:
                    item.set(color="black", linewidth=0.8)

                for item in bp["medians"]:
                    item.set(color="black", linewidth=1.2)

                reference = config["reference"]

                if reference is not None:
                    ax.axhline(
                        reference,
                        color="black",
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.9,
                        zorder=5
                    )

                limited = subset["limited_seeds"].gt(0).any()

                if limited and metric in LIMITED_METRICS:
                    ax.text(
                        0.97,
                        0.96,
                        "*",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=17,
                        fontweight="bold",
                        color="#9A3412",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.8,
                            "pad": 1.5
                        },
                        zorder=7
                    )

                ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
                ax.set_ylim(*y_limits[metric])

                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.set_yticks(
                    [t for t in ax.get_yticks() if -1e-9 <= t <= 1.0 + 1e-9]
                )
                ax.set_ylim(*y_limits[metric])

                if metric in SMALL_METRICS:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                else:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

                ax.grid(
                    axis="y",
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.65,
                    zorder=2
                )

                if row_idx == 0:
                    ax.set_title(config["label"], fontsize=15, pad=12)

                if col_idx == 0:
                    ax.set_ylabel(
                        dataset_labels[dataset],
                        fontsize=14,
                        labelpad=12
                    )

                if row_idx == len(DATASETS) - 1:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(
                        [MODEL_LABELS[m] for m in MODEL_ORDER],
                        rotation=60,
                        ha="right"
                    )
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        model_handles = [
            Patch(
                facecolor=MODEL_COLORS[model],
                edgecolor="black",
                label=MODEL_LABELS[model]
            )
            for model in MODEL_ORDER
        ]

        fig.legend(
            handles=model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.105),
            ncol=8,
            title="Generative model color code",
            frameon=True,
            fontsize=12,
            title_fontsize=13
        )

        family_handles = [
            Patch(
                facecolor=family["color"],
                edgecolor="none",
                alpha=0.75,
                label=family["label"]
            )
            for family in MODEL_FAMILIES
        ]

        fig.legend(
            handles=family_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=2,
            frameon=False,
            fontsize=12
        )

        reference_handles = [
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Dashed lines indicate metric-specific reference values"
            ),
            Line2D(
                [0], [0],
                color="#9A3412",
                linestyle="None",
                marker="$*$",
                markersize=13,
                label="Insufficient FPR calibration resolution"
            ),
        ]

        fig.legend(
            handles=reference_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.028),
            ncol=2,
            frameon=False,
            fontsize=11
        )

        fig.tight_layout(
            rect=[0.02, 0.15, 1, 0.97],
            w_pad=1.3,
            h_pad=1.5
        )

        output_name = "figure_3_privacy_evaluation"

        png_path = FIGURES_DIR / f"{output_name}.png"
        pdf_path = FIGURES_DIR / f"{output_name}.pdf"

        if include_title:
            fig.suptitle("Privacy", y=1.01)

        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

        plt.close(fig)

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        print("\nShared Y-axis limits by metric:")

        for metric, limits in y_limits.items():
            print(
                f"{METRICS[metric]['label'].replace(chr(10), ' ')}: "
                f"{limits[0]:.4f} – {limits[1]:.4f}"
            )

        print("\nInsufficient-resolution observations:")

        limited_summary = summary.loc[
            summary["limited_seeds"].gt(0),
            ["dataset", "metric", "generador", "limited_seeds"]
        ]

        print(
            limited_summary.groupby(["dataset", "metric"])["limited_seeds"]
            .agg(["min", "max"])
            .to_string()
        )


    plot_privacy_profiles()

# Compatibility for the existing A1 numerical audit; not used to draw figures.


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
