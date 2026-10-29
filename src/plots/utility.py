"""Utility figures ported from notebook cell 1.

Seed validation, summaries, axes and layout follow plotting_results_ver4.ipynb.
No notebook execution or notebook dependency is required.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.plots.styles import DATASET_LABELS, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS


def run(datasets, classifiers, project_root, png_dpi=300, include_title=False):
    """Generate the notebook figures for the selected datasets and classifiers."""

    PROJECT_ROOT = project_root
    CSV_PATH = PROJECT_ROOT / "results" / "results_long.csv"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "utility_profiles_final"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    Y_PADDING_RATIO = 0.20
    MIN_Y_PADDING = 0.05

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

    CLASSIFIERS = {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }

    METRICS = {
        "utility_auroc": "AUCROC",
        "utility_average_precision": "Average Precision",
        "utility_f1": "F1-score",
    }

    RATIO_METRICS = {
        "utility_auroc",
        "utility_average_precision",
        "utility_f1",
        "utility_ppv",
        "utility_sensitivity",
        "utility_specificity",
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

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "dataset", "seed", "generador", "component",
        "condition", "classifier", "metric", "value", "status"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Repair values corrupted by a thousands separator (0.688 -> 688)
    thousands_mask = (
        df["metric"].isin(RATIO_METRICS)
        & df["value"].between(1, 1000, inclusive="neither")
        & (df["value"] % 1 == 0)
    )

    if thousands_mask.any():
        print(f"Repairing {thousands_mask.sum()} values divided by 1000:")
        print(
            df.loc[
                thousands_mask,
                ["dataset", "seed", "generador", "classifier", "metric", "value"]
            ].to_string()
        )
        df.loc[thousands_mask, "value"] = df.loc[thousands_mask, "value"] / 1000

    df = df.loc[
        df["dataset"].isin(DATASETS)
        & df["classifier"].isin(CLASSIFIERS)
        & df["metric"].isin(METRICS)
        & df["status"].eq("ok")
        & (
            (
                df["component"].eq("utility_tstr")
                & df["condition"].eq("tstr")
                & df["generador"].isin(MODEL_ORDER)
            )
            |
            (
                df["component"].eq("trtr_original")
                & df["generador"].eq("real")
                & df["condition"].eq("trtr_original")
            )
        )
    ].copy()

    df = df.dropna(subset=["value"])

    out_of_range = (df["value"] < 0) | (df["value"] > 1)

    if out_of_range.any():
        print("Values outside [0, 1] excluded:")
        print(
            df.loc[
                out_of_range,
                ["dataset", "seed", "generador", "classifier", "metric", "value"]
            ].to_string()
        )
        df = df.loc[~out_of_range].copy()

    df["evaluation"] = np.where(df["generador"].eq("real"), "trtr", "tstr")

    duplicate_columns = [
        "dataset", "seed", "generador", "classifier", "metric", "evaluation"
    ]

    duplicates = df.duplicated(subset=duplicate_columns, keep=False)

    if duplicates.any():
        raise ValueError(
            "Duplicate utility observations detected:\n"
            f"{df.loc[duplicates, duplicate_columns].head(10)}"
        )

    if df.empty:
        raise ValueError("No valid utility results found.")

    trtr = df.loc[df["evaluation"].eq("trtr")].copy()
    tstr = df.loc[df["evaluation"].eq("tstr")].copy()

    summary = (
        tstr.groupby(["dataset", "classifier", "metric", "generador"])["value"]
        .agg(mean="mean", std="std", minimum="min", maximum="max", n="count")
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0)

    trtr_summary = (
        trtr.groupby(["dataset", "classifier", "metric"])["value"]
        .agg(mean="mean", n="count")
        .reset_index()
    )


    def get_classifier_y_limits(data, classifier):

        limits = {}

        for metric in METRICS:

            values = data.loc[
                data["classifier"].eq(classifier)
                & data["metric"].eq(metric),
                "value"
            ].to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if not len(values):
                raise ValueError(f"No values for {classifier} / {metric}")

            ymin = float(values.min())
            ymax = float(values.max())

            padding = max(MIN_Y_PADDING, (ymax - ymin) * Y_PADDING_RATIO)

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
                "fliers": []
            }

        row = row.iloc[0]

        return {
            "med": row["mean"],
            "q1": max(row["mean"] - row["std"], row["minimum"]),
            "q3": min(row["mean"] + row["std"], row["maximum"]),
            "whislo": row["minimum"],
            "whishi": row["maximum"],
            "fliers": []
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


    def plot_utility_profiles(classifier, classifier_label):

        y_limits = get_classifier_y_limits(df, classifier)

        fig, axes = plt.subplots(
            len(DATASETS),
            len(METRICS),
            figsize=(16, 14),
            sharex=True,
            squeeze=False
        )

        positions = np.arange(len(MODEL_ORDER))

        for row_idx, dataset in enumerate(DATASETS):

            for col_idx, (metric, metric_label) in enumerate(METRICS.items()):

                ax = axes[row_idx, col_idx]

                subset = summary.loc[
                    summary["dataset"].eq(dataset)
                    & summary["classifier"].eq(classifier)
                    & summary["metric"].eq(metric)
                ]

                reference = trtr_summary.loc[
                    trtr_summary["dataset"].eq(dataset)
                    & trtr_summary["classifier"].eq(classifier)
                    & trtr_summary["metric"].eq(metric)
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

                if not reference.empty:

                    trtr_mean = float(reference.iloc[0]["mean"])

                    ax.axhline(
                        trtr_mean,
                        color="black",
                        linestyle="--",
                        linewidth=1.4,
                        zorder=5
                    )

                    ax.text(
                        0.02,
                        0.03,
                        f"TRTR = {trtr_mean:.3f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.8,
                            "pad": 2
                        },
                        zorder=6
                    )

                ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
                ax.set_ylim(*y_limits[metric])
                ax.set_yticks(
                    [t for t in ax.get_yticks() if -1e-9 <= t <= 1.0 + 1e-9]
                )
                ax.set_ylim(*y_limits[metric])

                ax.grid(
                    axis="y",
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.65,
                    zorder=2
                )

                if row_idx == 0:
                    ax.set_title(f"{metric_label} (↑)")

                if col_idx == 0:
                    ax.set_ylabel(dataset_labels[dataset], labelpad=12)

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
            bbox_to_anchor=(0.5, 0.087),
            ncol=4,
            title="Generative model color code",
            frameon=True,
            fontsize=11
        )

        fig.legend(
            handles=[
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Dashed horizontal lines indicate the five-seed TRTR mean"
                )
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            frameon=False
        )

        family_handles = [
            Patch(
                facecolor=family["color"],
                edgecolor="none",
                label=family["label"],
                alpha=0.75
            )
            for family in MODEL_FAMILIES
        ]

        fig.legend(
            handles=family_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=2,
            frameon=False
        )

        fig.tight_layout(rect=[0.02, 0.15, 1, 0.965])

        output_name = f"utility_profiles_{classifier}"
        png_path = FIGURES_DIR / f"{output_name}.png"
        pdf_path = FIGURES_DIR / f"{output_name}.pdf"

        if include_title:
            fig.suptitle("Utility", y=1.01)

        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        print(f"\nY-axis limits for {classifier_label}:")

        for metric, limits in y_limits.items():
            print(f"  {METRICS[metric]}: {limits[0]:.3f} – {limits[1]:.3f}")


    for classifier, classifier_label in CLASSIFIERS.items():
        if classifier not in classifiers:
            continue
        plot_utility_profiles(classifier, classifier_label)
