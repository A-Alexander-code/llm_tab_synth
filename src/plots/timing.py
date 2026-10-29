"""Timing figures ported from notebook cell 17.

Seed validation, summaries, axes and layout follow plotting_results_ver4.ipynb.
No notebook execution or notebook dependency is required.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from src.plots.styles import DATASET_LABELS, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS


def run(datasets, classifiers, project_root, png_dpi=300, include_title=False):
    """Generate the notebook figures for the selected datasets and classifiers."""

    PROJECT_ROOT = project_root
    CSV_PATH = PROJECT_ROOT / "results" / "generation_timing.csv"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary_final"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    Y_PADDING_FACTOR = 1.6
    EXPECTED_SEEDS = {13, 23, 42, 69, 88}
    Y_AXIS_LABEL = "Time (seconds, log scale)"

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

    PHASES = {
        "fit": "Fitting Time",
        "sampling": "Sampling Time",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(CSV_PATH, low_memory=False)

    required_columns = {
        "generator", "dataset", "seed", "class",
        "phase", "status", "elapsed_seconds",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["seed"] = pd.to_numeric(df["seed"], errors="raise")
    df["class"] = pd.to_numeric(df["class"], errors="raise")
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="raise")

    df = df.loc[
        df["dataset"].isin(DATASETS)
        & df["generator"].isin(MODEL_ORDER)
        & df["phase"].isin(PHASES)
    ].copy()

    if df.empty:
        raise ValueError("No matching timing observations found.")

    if not df["status"].eq("completed").all():
        raise ValueError("Incomplete timing observations detected.")

    if (
        df["elapsed_seconds"].isna().any()
        or not np.isfinite(df["elapsed_seconds"]).all()
        or df["elapsed_seconds"].le(0).any()
    ):
        raise ValueError("Timing values must be finite and positive.")

    duplicate_keys = ["dataset", "generator", "seed", "class", "phase"]

    duplicates = df.duplicated(subset=duplicate_keys, keep=False)

    if duplicates.any():
        raise ValueError(
            "Duplicate timing observations detected:\n"
            + df.loc[duplicates, duplicate_keys].head(15).to_string(index=False)
        )

    expected_index = pd.MultiIndex.from_product(
        [DATASETS, MODEL_ORDER, sorted(EXPECTED_SEEDS), [0, 1], PHASES.keys()],
        names=duplicate_keys
    )

    observed_index = pd.MultiIndex.from_frame(df[duplicate_keys])

    missing = expected_index.difference(observed_index)
    unexpected = observed_index.difference(expected_index)

    if len(missing) or len(unexpected):
        raise ValueError(
            f"Incomplete timing structure. "
            f"Missing: {len(missing)}; unexpected: {len(unexpected)}."
        )

    seed_totals = (
        df.groupby(["dataset", "generator", "seed", "phase"], as_index=False)
        ["elapsed_seconds"]
        .sum()
        .rename(columns={"elapsed_seconds": "total_seconds"})
    )

    summary = (
        seed_totals.groupby(["dataset", "generator", "phase"], as_index=False)
        ["total_seconds"]
        .agg(mean="mean", std="std", minimum="min", maximum="max", n="count")
    )

    summary["std"] = summary["std"].fillna(0.0)

    if not summary["n"].eq(5).all():
        raise ValueError("Some timing groups do not contain five seeds.")


    def get_phase_limits(data):

        limits = {}

        for phase in PHASES:

            subset = data.loc[data["phase"].eq(phase)]

            lower = np.maximum(
                subset["minimum"].to_numpy(dtype=float),
                subset["mean"].to_numpy(dtype=float)
                - subset["std"].to_numpy(dtype=float)
            )

            upper = np.minimum(
                subset["maximum"].to_numpy(dtype=float),
                subset["mean"].to_numpy(dtype=float)
                + subset["std"].to_numpy(dtype=float)
            )

            limits[phase] = (
                float(np.min(lower)) / Y_PADDING_FACTOR,
                float(np.max(upper)) * Y_PADDING_FACTOR
            )

        return limits


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


    def plot_computational_time():

        y_limits = get_phase_limits(summary)

        fig, axes = plt.subplots(
            len(DATASETS),
            len(PHASES),
            figsize=(16, 14),
            sharex=True,
            squeeze=False
        )

        positions = np.arange(len(MODEL_ORDER))

        for row_idx, dataset in enumerate(DATASETS):

            for col_idx, (phase, phase_label) in enumerate(PHASES.items()):

                ax = axes[row_idx, col_idx]

                subset = summary.loc[
                    summary["dataset"].eq(dataset)
                    & summary["phase"].eq(phase)
                ]

                draw_family_bands(ax)

                for model_idx, model in enumerate(MODEL_ORDER):

                    row = subset.loc[subset["generator"].eq(model)]

                    if row.empty:
                        continue

                    row = row.iloc[0]

                    mean = float(row["mean"])
                    std = float(row["std"])
                    minimum = float(row["minimum"])
                    maximum = float(row["maximum"])

                    lower = max(minimum, mean - std)
                    upper = min(maximum, mean + std)

                    lower_error = max(0.0, mean - lower)
                    upper_error = max(0.0, upper - mean)

                    ax.errorbar(
                        model_idx,
                        mean,
                        yerr=np.array([[lower_error], [upper_error]]),
                        fmt="o",
                        markersize=8,
                        color=MODEL_COLORS[model],
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        ecolor=MODEL_COLORS[model],
                        elinewidth=1.5,
                        capsize=4,
                        capthick=1.2,
                        alpha=0.95,
                        zorder=5
                    )

                ax.set_yscale("log")
                ax.set_ylim(*y_limits[phase])
                ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)

                ax.yaxis.set_major_locator(LogLocator(base=10))

                ax.yaxis.set_minor_locator(
                    LogLocator(base=10, subs=np.arange(2, 10))
                )

                ax.yaxis.set_minor_formatter(NullFormatter())

                ax.grid(
                    axis="y",
                    which="major",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.65,
                    zorder=1
                )

                ax.grid(
                    axis="y",
                    which="minor",
                    linestyle=":",
                    linewidth=0.35,
                    alpha=0.25,
                    zorder=1
                )

                if row_idx == 0:
                    ax.set_title(phase_label, fontsize=16, pad=12)

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
                        ha="right",
                        fontsize=11
                    )
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        fig.tight_layout(
            rect=[0.04, 0.15, 1, 0.98],
            w_pad=1.5,
            h_pad=1.5
        )

        top = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0

        fig.text(
            0.012,
            (top + bottom) / 2,
            Y_AXIS_LABEL,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14
        )

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

        output_name = "figure_S6_computational_time"

        png_path = FIGURES_DIR / f"{output_name}.png"
        pdf_path = FIGURES_DIR / f"{output_name}.pdf"

        if include_title:
            fig.suptitle("Timing", y=1.01)

        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

        plt.close(fig)

        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

        print("\nShared logarithmic Y-axis limits:")

        for phase, limits in y_limits.items():
            print(
                f"  {PHASES[phase]}: "
                f"{limits[0]:.4f} – {limits[1]:.4f} seconds"
            )

        print("\nTiming summary:")

        print(
            summary[["dataset", "generator", "phase", "mean", "std"]]
            .round(3)
            .to_string(index=False)
        )

        return summary


    timing_summary = plot_computational_time()
