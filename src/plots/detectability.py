"""Detectability figures ported from notebook cell 13.

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
from src.plots.styles import DATASET_LABELS, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS


def run(datasets, classifiers, project_root, png_dpi=300, include_title=False):
    """Generate the notebook figures for the selected datasets and classifiers."""

    PROJECT_ROOT = project_root
    CSV_PATH = PROJECT_ROOT / "results" / "results_long.csv"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary_final"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    Y_PADDING_RATIO = 0.20
    MIN_Y_PADDING = 0.05
    PMSE_MIN_PADDING = 0.025

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
        "detectability_auroc": {
            "label": "Detectability AUCROC",
            "reference": 0.5,
            "min_padding": MIN_Y_PADDING,
            "max_tick": 1.0,
        },
        "detectability_pmse_normalized": {
            "label": "Normalized pMSE",
            "reference": None,
            "min_padding": PMSE_MIN_PADDING,
            "max_tick": None,
        },
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(CSV_PATH, low_memory=False)

    required_columns = {
        "dataset", "seed", "generador", "component",
        "fold", "metric", "value", "status",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["fold"] = df["fold"].astype("string")
    df["status"] = df["status"].astype("string").str.lower()

    df = df.loc[
        df["dataset"].isin(DATASETS)
        & df["generador"].isin(MODEL_ORDER)
        & df["component"].eq("detectability")
        & df["metric"].isin(METRICS)
        & df["fold"].eq("mean").fillna(False)
        & df["status"].eq("ok")
    ].copy()

    if df.empty:
        raise ValueError("No valid detectability observations found.")

    invalid_values = (
        df["value"].isna()
        | ~np.isfinite(df["value"])
        | df["value"].lt(0)
    )

    invalid_auroc = df["metric"].eq("detectability_auroc") & df["value"].gt(1)

    if (invalid_values | invalid_auroc).any():
        raise ValueError(
            "Invalid detectability values:\n"
            + df.loc[
                invalid_values | invalid_auroc,
                ["dataset", "seed", "generador", "metric", "value"]
            ].head(15).to_string(index=False)
        )

    duplicate_columns = ["dataset", "seed", "generador", "metric"]

    duplicates = df.duplicated(subset=duplicate_columns, keep=False)

    if duplicates.any():
        raise ValueError(
            "Duplicate detectability observations:\n"
            + df.loc[duplicates, duplicate_columns].head(15).to_string(index=False)
        )

    expected_seeds = {13, 23, 42, 69, 88}

    expected_combinations = pd.MultiIndex.from_product(
        [DATASETS, MODEL_ORDER, METRICS.keys()],
        names=["dataset", "generador", "metric"]
    )

    observed_combinations = pd.MultiIndex.from_frame(
        df[["dataset", "generador", "metric"]].drop_duplicates()
    )

    missing_combinations = expected_combinations.difference(observed_combinations)

    if len(missing_combinations):
        raise ValueError(
            "Missing detectability combinations:\n"
            + str(missing_combinations.tolist()[:15])
        )

    seed_sets = (
        df.groupby(["dataset", "generador", "metric"])["seed"].agg(lambda x: set(x))
    )

    incomplete_seeds = seed_sets[seed_sets.apply(lambda x: x != expected_seeds)]

    if not incomplete_seeds.empty:
        raise ValueError(
            "Missing or unexpected experimental seeds:\n"
            + incomplete_seeds.to_string()
        )

    summary = (
        df.groupby(["dataset", "metric", "generador"])["value"]
        .agg(mean="mean", std="std", minimum="min", maximum="max", n="count")
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

            minimum = float(np.min(values))
            maximum = float(np.max(values))

            padding = max(
                config["min_padding"],
                (maximum - minimum) * Y_PADDING_RATIO
            )

            limits[metric] = (minimum - padding, maximum + padding)

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


    def plot_detectability():

        y_limits = get_metric_y_limits(df)

        fig, axes = plt.subplots(
            len(DATASETS),
            len(METRICS),
            figsize=(15, 14),
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
                        linewidth=0.85
                    )

                for item in bp["whiskers"] + bp["caps"]:
                    item.set(color="black", linewidth=0.85)

                for item in bp["medians"]:
                    item.set(color="black", linewidth=1.2)

                reference = config["reference"]

                if reference is not None:
                    ax.axhline(
                        reference,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        zorder=5
                    )

                ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
                ax.set_ylim(*y_limits[metric])

                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

                max_tick = config["max_tick"]

                ax.set_yticks([
                    t for t in ax.get_yticks()
                    if t >= -1e-9 and (max_tick is None or t <= max_tick + 1e-9)
                ])
                ax.set_ylim(*y_limits[metric])

                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

                ax.grid(
                    axis="y",
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.65,
                    zorder=2
                )

                if row_idx == 0:
                    ax.set_title(config["label"], fontsize=16, pad=12)

                if col_idx == 0:
                    ax.set_ylabel(
                        dataset_labels[dataset],
                        fontsize=14,
                        labelpad=15
                    )

                if row_idx == len(DATASETS) - 1:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(
                        [MODEL_LABELS[model] for model in MODEL_ORDER],
                        rotation=60,
                        ha="right",
                        fontsize=11
                    )
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        model_handles = [
            Patch(
                facecolor=MODEL_COLORS[model],
                edgecolor="black",
                label=MODEL_LABELS[model],
                alpha=0.9
            )
            for model in MODEL_ORDER
        ]

        fig.legend(
            handles=model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.085),
            ncol=4,
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
            bbox_to_anchor=(0.5, 0.038),
            ncol=2,
            frameon=False,
            fontsize=12
        )

        fig.legend(
            handles=[
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Dashed line: AUCROC = 0.5 (chance-level discrimination)"
                )
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            frameon=False,
            fontsize=11
        )

        fig.tight_layout(
            rect=[0.02, 0.15, 1, 0.98],
            w_pad=1.5,
            h_pad=1.5
        )

        output_name = "figure_S5_detectability_profiles"

        png_path = FIGURES_DIR / f"{output_name}.png"
        pdf_path = FIGURES_DIR / f"{output_name}.pdf"

        if include_title:
            fig.suptitle("Detectability", y=1.01)

        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

        plt.close(fig)

        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

        print("\nShared Y-axis limits by metric:")

        for metric, limits in y_limits.items():
            print(
                f"  {METRICS[metric]['label']}: "
                f"{limits[0]:.4f} – {limits[1]:.4f}"
            )

        print("\nObservations per metric:")
        print(df.groupby("metric")["value"].count().to_string())

        return summary


    detectability_summary = plot_detectability()
