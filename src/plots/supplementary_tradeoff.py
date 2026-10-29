"""Supplementary Tradeoff figures ported from notebook cell 10.

Seed validation, summaries, axes and layout follow plotting_results_ver4.ipynb.
No notebook execution or notebook dependency is required.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from src.plots.styles import DATASET_LABELS, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS


def run(datasets, classifiers, project_root, png_dpi=300, include_title=False):
    """Generate the notebook figures for the selected datasets and classifiers."""

    PROJECT_ROOT = project_root
    CSV_PATH = PROJECT_ROOT / "results" / "results_long.csv"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "supplementary_final"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    CLASSIFIERS = {
        "random_forest": {"label": "Random Forest", "figure": "S3"},
        "logistic_regression": {"label": "Logistic Regression", "figure": "S4"},
    }

    CLASSIFIERS = {k: v for k, v in CLASSIFIERS.items() if k in classifiers}

    Y_PADDING_RATIO = 0.20
    MIN_PADDING = 0.03
    SMALL_METRIC_PADDING = 0.008

    DATASETS = list(datasets)
    dataset_labels = {d: DATASET_LABELS.get(d, d.replace("_", " ").title()) for d in DATASETS}





    MODEL_FAMILIES = [
        {
            "label": "LLM-based generators",
            "models": ["great", "tabula", "realtabformer"],
            "marker": "o",
        },
        {
            "label": "Comparison generators",
            "models": ["ctgan", "copulagan", "tvae", "gaussian_copula", "mostlyai"],
            "marker": "s",
        },
    ]

    PANELS = [
        {
            "x": "utility_auroc",
            "y": "mia_auroc",
            "component": "mia",
            "fold": "mean",
            "title": "Utility AUCROC vs. MIA AUCROC",
            "xlabel": "TSTR AUCROC",
            "ylabel": "MIA AUCROC",
            "reference": 0.5,
            "pad_lower": False,
        },
        {
            "x": "utility_auroc",
            "y": "nnaa_overall",
            "component": "nnaa",
            "fold": None,
            "title": "Utility AUCROC vs. NNAA",
            "xlabel": "TSTR AUCROC",
            "ylabel": "NNAA",
            "reference": 0.5,
            "pad_lower": False,
        },
        {
            "x": "utility_average_precision",
            "y": "mia_tpr_target_fpr_1pct_calibrated",
            "component": "mia",
            "fold": "aggregate",
            "title": "Average Precision vs. MIA TPR",
            "xlabel": "TSTR Average Precision",
            "ylabel": "MIA TPR@1% FPR",
            "reference": None,
            "pad_lower": True,
        },
        {
            "x": "utility_average_precision",
            "y": "neighborhood_k5_distinct_groups_ratio",
            "component": "neighborhood",
            "fold": None,
            "title": "Average Precision vs. Neighborhood-based\nk-Anonymity",
            "xlabel": "TSTR Average Precision",
            "ylabel": "k=5",
            "reference": None,
            "pad_lower": True,
        },
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(CSV_PATH, low_memory=False)

    required_columns = {
        "dataset", "seed", "generador", "component", "condition",
        "classifier", "fold", "metric", "value", "status",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["status"] = df["status"].astype("string").str.lower()
    df["fold"] = df["fold"].astype("string")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    utility_metrics = sorted({panel["x"] for panel in PANELS})
    privacy_metrics = sorted({panel["y"] for panel in PANELS})

    utility = df.loc[
        df["dataset"].isin(DATASETS)
        & df["generador"].isin(MODEL_ORDER)
        & df["classifier"].isin(CLASSIFIERS)
        & df["component"].eq("utility_tstr")
        & df["condition"].eq("tstr")
        & df["metric"].isin(utility_metrics)
        & df["fold"].isna()
        & df["status"].eq("ok")
    ].copy()

    baseline = df.loc[
        df["dataset"].isin(DATASETS)
        & df["generador"].eq("real")
        & df["classifier"].isin(CLASSIFIERS)
        & df["component"].eq("trtr_original")
        & df["condition"].eq("trtr_original")
        & df["metric"].isin(utility_metrics)
        & df["fold"].isna()
        & df["status"].eq("ok")
    ].copy()

    privacy_parts = []

    for panel in PANELS:

        part = df.loc[
            df["dataset"].isin(DATASETS)
            & df["generador"].isin(MODEL_ORDER)
            & df["metric"].eq(panel["y"])
            & df["component"].eq(panel["component"])
        ].copy()

        if panel["fold"] is None:
            part = part.loc[part["fold"].isna()]
        else:
            part = part.loc[part["fold"].eq(panel["fold"]).fillna(False)]

        if panel["y"] == "mia_tpr_target_fpr_1pct_calibrated":
            accepted_status = {"ok", "insufficient_resolution"}
        else:
            accepted_status = {"ok"}

        part = part.loc[part["status"].isin(accepted_status)]

        privacy_parts.append(part)

    privacy = pd.concat(privacy_parts, ignore_index=True)

    for name, data in {
        "utility": utility,
        "baseline": baseline,
        "privacy": privacy,
    }.items():

        invalid = (
            data["value"].isna()
            | ~np.isfinite(data["value"])
            | data["value"].lt(0)
            | data["value"].gt(1)
        )

        if invalid.any():
            raise ValueError(
                f"Invalid {name} values:\n"
                + data.loc[
                    invalid,
                    ["dataset", "seed", "generador", "metric", "value"]
                ].head(15).to_string(index=False)
            )

        duplicate_keys = ["dataset", "seed", "generador", "metric"]

        if name != "privacy":
            duplicate_keys.append("classifier")

        duplicated = data.duplicated(subset=duplicate_keys, keep=False)

        if duplicated.any():
            raise ValueError(
                f"Duplicate {name} observations:\n"
                + data.loc[duplicated, duplicate_keys].head(15).to_string(index=False)
            )


    def validate_seed_counts(data, classifier=None):

        subset = data.copy()

        if classifier is not None:
            subset = subset.loc[subset["classifier"].eq(classifier)]

        counts = subset.groupby(["dataset", "generador", "metric"])["seed"].nunique()

        if counts.empty or not counts.eq(5).all():
            raise ValueError(
                "Incomplete five-seed observations detected:\n"
                + counts[counts.ne(5)].to_string()
            )


    for classifier in CLASSIFIERS:
        validate_seed_counts(utility, classifier)

    validate_seed_counts(privacy)

    baseline_counts = (
        baseline.groupby(["dataset", "classifier", "metric"])["seed"].nunique()
    )

    if (
        len(baseline_counts)
        != len(DATASETS) * len(CLASSIFIERS) * len(utility_metrics)
        or not baseline_counts.eq(5).all()
    ):
        raise ValueError("Incomplete TRTR baseline observations.")


    def marker_for_model(model):

        for family in MODEL_FAMILIES:
            if model in family["models"]:
                return family["marker"]

        return "o"


    def pair_seed_observations(classifier, x_metric, y_metric):

        x = utility.loc[
            utility["classifier"].eq(classifier)
            & utility["metric"].eq(x_metric),
            ["dataset", "seed", "generador", "value"]
        ].rename(columns={"value": "x"})

        y = privacy.loc[
            privacy["metric"].eq(y_metric),
            ["dataset", "seed", "generador", "value", "status"]
        ].rename(columns={"value": "y", "status": "privacy_status"})

        paired = x.merge(
            y,
            on=["dataset", "seed", "generador"],
            how="inner",
            validate="one_to_one"
        )

        expected = pd.MultiIndex.from_product(
            [DATASETS, MODEL_ORDER],
            names=["dataset", "generador"]
        )

        observed = pd.MultiIndex.from_frame(
            paired[["dataset", "generador"]].drop_duplicates()
        )

        missing = expected.difference(observed)

        if len(missing):
            raise ValueError(f"Missing paired combinations: {missing.tolist()[:10]}")

        counts = paired.groupby(["dataset", "generador"])["seed"].nunique()

        if not counts.eq(5).all():
            raise ValueError(
                f"Incomplete paired seeds for {classifier}: {x_metric} / {y_metric}"
            )

        result = (
            paired.groupby(["dataset", "generador"])
            .agg(
                x_mean=("x", "mean"),
                x_std=("x", "std"),
                y_mean=("y", "mean"),
                y_std=("y", "std"),
                n=("seed", "nunique"),
                limited=(
                    "privacy_status",
                    lambda s: s.eq("insufficient_resolution").any()
                ),
            )
            .reset_index()
        )

        result[["x_std", "y_std"]] = result[["x_std", "y_std"]].fillna(0)

        return result


    def padded_limits(
        values,
        reference=None,
        small=False,
        errors=None,
        clip_lower=True
    ):

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if errors is not None:

            lower_errors, upper_errors = errors

            values = np.concatenate([
                values,
                np.asarray(lower_errors, dtype=float),
                np.asarray(upper_errors, dtype=float),
            ])

            values = values[np.isfinite(values)]

        if reference is not None:
            values = np.append(values, reference)

        if values.size == 0:
            raise ValueError("Cannot calculate axis limits from empty data.")

        minimum = float(values.min())
        maximum = float(values.max())

        padding = max(
            SMALL_METRIC_PADDING if small else MIN_PADDING,
            (maximum - minimum) * Y_PADDING_RATIO
        )

        lower = minimum - padding
        upper = min(1.0, maximum + padding)

        if clip_lower:
            lower = max(0.0, lower)

        return lower, upper


    def prepare_classifier_data(classifier):

        panel_summaries = []

        for panel_idx, panel in enumerate(PANELS):

            result = pair_seed_observations(classifier, panel["x"], panel["y"])
            result["panel"] = panel_idx
            panel_summaries.append(result)

        plot_data = pd.concat(panel_summaries, ignore_index=True)

        classifier_utility = utility.loc[utility["classifier"].eq(classifier)]
        classifier_baseline = baseline.loc[baseline["classifier"].eq(classifier)]

        baseline_summary = (
            classifier_baseline.groupby(["dataset", "metric"])["value"]
            .mean()
            .to_dict()
        )

        x_limits = {}

        for metric in utility_metrics:

            subset = plot_data.loc[
                plot_data["panel"].map(lambda idx: PANELS[int(idx)]["x"] == metric)
            ]

            values = np.concatenate([
                classifier_utility.loc[
                    classifier_utility["metric"].eq(metric), "value"
                ].to_numpy(dtype=float),
                classifier_baseline.loc[
                    classifier_baseline["metric"].eq(metric), "value"
                ].to_numpy(dtype=float),
            ])

            x_limits[metric] = padded_limits(
                values,
                errors=(
                    subset["x_mean"] - subset["x_std"],
                    subset["x_mean"] + subset["x_std"]
                )
            )

        y_limits = {}

        for panel_idx, panel in enumerate(PANELS):

            subset = plot_data.loc[plot_data["panel"].eq(panel_idx)]

            metric_values = privacy.loc[
                privacy["metric"].eq(panel["y"]), "value"
            ].to_numpy(dtype=float)

            y_limits[panel_idx] = padded_limits(
                metric_values,
                reference=panel["reference"],
                small=panel["y"] == "mia_tpr_target_fpr_1pct_calibrated",
                errors=(
                    subset["y_mean"] - subset["y_std"],
                    subset["y_mean"] + subset["y_std"]
                ),
                clip_lower=not panel["pad_lower"]
            )

        return plot_data, baseline_summary, x_limits, y_limits


    def plot_supplementary_tradeoff(classifier, config):

        plot_data, baseline_summary, x_limits, y_limits = (
            prepare_classifier_data(classifier)
        )

        fig, axes = plt.subplots(
            len(DATASETS),
            len(PANELS),
            figsize=(20, 16),
            squeeze=False
        )

        for row_idx, dataset in enumerate(DATASETS):

            for col_idx, panel in enumerate(PANELS):

                ax = axes[row_idx, col_idx]

                subset = plot_data.loc[
                    plot_data["dataset"].eq(dataset)
                    & plot_data["panel"].eq(col_idx)
                ]

                trtr_value = baseline_summary.get((dataset, panel["x"]), np.nan)

                if np.isfinite(trtr_value):
                    ax.axvline(
                        trtr_value,
                        color="black",
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=1
                    )

                if panel["reference"] is not None:
                    ax.axhline(
                        panel["reference"],
                        color="#666666",
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.85,
                        zorder=1
                    )

                for model in MODEL_ORDER:

                    row = subset.loc[subset["generador"].eq(model)]

                    if row.empty:
                        continue

                    row = row.iloc[0]

                    color = MODEL_COLORS[model]
                    marker = marker_for_model(model)

                    ax.errorbar(
                        row["x_mean"],
                        row["y_mean"],
                        xerr=row["x_std"],
                        yerr=row["y_std"],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.15,
                        capsize=3,
                        capthick=1,
                        alpha=0.65,
                        zorder=3
                    )

                    ax.scatter(
                        row["x_mean"],
                        row["y_mean"],
                        color=color,
                        marker=marker,
                        s=80,
                        edgecolor="black",
                        linewidth=0.7,
                        alpha=0.95,
                        zorder=5
                    )

                if (
                    panel["y"] == "mia_tpr_target_fpr_1pct_calibrated"
                    and subset["limited"].any()
                ):
                    ax.text(
                        0.97,
                        0.96,
                        "*",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=18,
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

                ax.set_xlim(*x_limits[panel["x"]])
                ax.set_ylim(*y_limits[col_idx])

                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                if panel["pad_lower"]:
                    ax.set_yticks(
                        [t for t in ax.get_yticks() if -1e-9 <= t <= 1.0 + 1e-9]
                    )
                    ax.set_ylim(*y_limits[col_idx])

                if panel["y"] == "mia_tpr_target_fpr_1pct_calibrated":
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

                ax.grid(
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.55,
                    zorder=0
                )

                if row_idx == 0:
                    ax.set_title(panel["title"], fontsize=14, pad=13)

                if col_idx == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        dataset_labels[dataset],
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=14,
                        fontweight="bold"
                    )

                ax.set_ylabel(panel["ylabel"])

                if row_idx == len(DATASETS) - 1:
                    ax.set_xlabel(panel["xlabel"])

        model_handles = [
            Line2D(
                [0], [0],
                marker=marker_for_model(model),
                linestyle="None",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor="black",
                markersize=9,
                label=MODEL_LABELS[model]
            )
            for model in MODEL_ORDER
        ]

        fig.legend(
            handles=model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.105),
            ncol=8,
            title="Synthetic data generators",
            frameon=True,
            fontsize=11,
            title_fontsize=12
        )

        family_handles = [
            Line2D(
                [0], [0],
                marker=family["marker"],
                linestyle="None",
                color="black",
                markerfacecolor="white",
                markersize=9,
                label=family["label"]
            )
            for family in MODEL_FAMILIES
        ]

        fig.legend(
            handles=family_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.072),
            ncol=2,
            frameon=False,
            fontsize=11
        )

        reference_handles = [
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Vertical dashed line: mean TRTR baseline"
            ),
            Line2D(
                [0], [0],
                color="#666666",
                linestyle="--",
                linewidth=1.5,
                label="Horizontal dashed line: metric-specific reference"
            ),
            Line2D(
                [0], [0],
                marker="$*$",
                linestyle="None",
                color="#9A3412",
                markersize=13,
                label="Insufficient MIA FPR calibration resolution"
            ),
        ]

        fig.legend(
            handles=reference_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=3,
            frameon=False,
            fontsize=10
        )

        fig.tight_layout(
            rect=[0.04, 0.15, 1, 0.97],
            w_pad=1.8,
            h_pad=2.0
        )

        output_name = (
            f"figure_{config['figure']}_privacy_utility_tradeoff_{classifier}"
        )

        png_path = FIGURES_DIR / f"{output_name}.png"
        pdf_path = FIGURES_DIR / f"{output_name}.pdf"

        if include_title:
            fig.suptitle("Supplementary Tradeoff", y=1.01)

        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

        plt.close(fig)

        print(f"\nFigure {config['figure']} — {config['label']}")
        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

        print("\nX-axis limits:")

        for metric, limits in x_limits.items():
            print(f"  {metric}: {limits[0]:.4f} – {limits[1]:.4f}")

        print("\nY-axis limits:")

        for panel_idx, panel in enumerate(PANELS):
            limits = y_limits[panel_idx]
            print(f"  {panel['ylabel']}: {limits[0]:.4f} – {limits[1]:.4f}")


    for classifier, config in CLASSIFIERS.items():
        plot_supplementary_tradeoff(classifier, config)
