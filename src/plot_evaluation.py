"""
plot_evaluation.py

Genera figuras para publicación científica:
  A. Scatter: Privacidad (dcr_mean) vs Utilidad (tstr_auc_ratio)
  B. Heatmaps: Matrices de correlación real vs modelos sintéticos
  C. Ridge plots: Distribuciones de densidad por variable

Estilo: fondo blanco, paleta accesible, tipografía limpia, 300 DPI.
Outputs en evaluation_results/plots/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─── Estilo global ────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.6,
    "grid.linewidth":     0.4,
    "grid.color":         "#CCCCCC",
    "grid.linestyle":     "--",
    "axes.grid":          False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "text.color":         "black",
    "axes.labelcolor":    "black",
    "xtick.color":        "black",
    "ytick.color":        "black",
})

# Paleta accesible para daltónicos (Wong 2011, 8 colores)
WONG_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# ─── Configuración ────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"
SYNTH_BASE = Path("synth_samples")
EVAL_DIR   = Path("evaluation_results")
OUTPUT_DIR = EVAL_DIR / "plots"
TARGET_COL = "Outcome"

RIDGE_VARS   = ["Glucose", "BMI"]
TOP_N_MODELS = 3
SAVE_FORMATS = ["png", "pdf"]
DPI          = 300

# Marcadores distintos por modelo para gráficos b/w imprimibles
MODEL_MARKERS = {
    "great":           "o",
    "tabula":          "s",
    "realtabformer":   "^",
    "ctgan":           "D",
    "tvae":            "v",
    "gaussian_copula": "P",
    "copulagan":       "X",
    "tabpfgen":        "h",
    "forestdiffusion": "*",
    "mostlyai":        "p",
}
DEFAULT_MARKER = "o"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    real      = pd.read_csv(INPUT_FILE)
    per_class = pd.read_csv(EVAL_DIR / "evaluation_per_class.csv")
    tstr      = pd.read_csv(EVAL_DIR / "evaluation_tstr.csv")
    return real, per_class, tstr


def load_synth(model: str, label: int) -> pd.DataFrame | None:
    p = SYNTH_BASE / model / f"class_{label}" / f"synthetic_class_{label}.csv"
    return pd.read_csv(p) if p.exists() else None


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def save_fig(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in SAVE_FORMATS:
        path = OUTPUT_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


def assign_colors_markers(models: list[str]):
    """Asigna colores de la paleta Wong y marcadores a cada modelo."""
    colors  = {}
    markers = {}
    for i, m in enumerate(models):
        colors[m]  = WONG_PALETTE[i % len(WONG_PALETTE)]
        markers[m] = MODEL_MARKERS.get(m, DEFAULT_MARKER)
    return colors, markers


# ─── A. Scatter: Privacidad vs Utilidad ───────────────────────────────────────

def plot_scatter_privacy_utility(per_class: pd.DataFrame, tstr: pd.DataFrame):
    print("A. Scatter: Privacidad vs Utilidad")

    models = sorted(per_class["model"].unique().tolist())
    colors, markers = assign_colors_markers(models)

    for label in [0, 1]:
        subset = per_class[per_class["class"] == label][["model", "dcr_mean"]].copy()
        merged = subset.merge(tstr[["model", "tstr_auc_ratio"]], on="model", how="inner")
        if merged.empty:
            print(f"  No data for class {label}")
            continue

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        for _, row in merged.iterrows():
            m = row["model"]
            ax.scatter(
                row["dcr_mean"], row["tstr_auc_ratio"],
                color=colors.get(m, WONG_PALETTE[0]),
                marker=markers.get(m, DEFAULT_MARKER),
                s=60, zorder=5, linewidths=0.5, edgecolors="black",
                label=m,
            )

        # Línea de referencia TRTR (utilidad = 1.0)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
                   label="TRTR baseline")

        # Mediana de DCR como referencia vertical
        xmed = merged["dcr_mean"].median()
        ax.axvline(xmed, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_xlabel(r"$\overline{DCR}$ (Privacy — higher is better)")
        ax.set_ylabel(r"$TSTR_{AUC}$ ratio (Utility — higher is better)")
        ax.set_title(f"Privacy–Utility Trade-off (Class {label})")

        # Leyenda fuera del área del gráfico
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=7.5,
        )

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        fig.tight_layout()
        save_fig(fig, f"A_scatter_privacy_utility_class{label}")
        plt.close(fig)


# ─── B. Heatmaps de correlación ───────────────────────────────────────────────

def plot_heatmaps(real: pd.DataFrame, per_class: pd.DataFrame):
    print("B. Heatmaps de correlación")
    feat = feature_cols(real)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for label in [0, 1]:
        real_class = real[real[TARGET_COL] == label][feat]
        corr_real  = real_class.corr()

        models  = sorted([p.name for p in SYNTH_BASE.iterdir() if p.is_dir()])
        n_total = len(models) + 1   # +1 para real
        ncols   = 3
        nrows   = (n_total + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.8, nrows * 3.5),
        )
        axes = axes.flatten()

        fig.suptitle(
            f"Pearson Correlation Matrices — Class {label}",
            fontsize=12, y=1.01,
        )

        def draw_hm(ax, corr, title, is_real=False):
            sns.heatmap(
                corr, ax=ax,
                cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f",
                annot_kws={"size": 5.5},
                linewidths=0.3, linecolor="#E0E0E0",
                cbar_kws={"shrink": 0.75, "aspect": 15},
                square=True,
            )
            weight = "bold" if is_real else "normal"
            prefix = "★ " if is_real else ""
            ax.set_title(f"{prefix}{title}", fontsize=8.5, fontweight=weight, pad=4)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_xlabel("")
            ax.set_ylabel("")

        draw_hm(axes[0], corr_real, "Real data", is_real=True)

        for idx, model in enumerate(models):
            synth = load_synth(model, label)
            if synth is None:
                axes[idx + 1].axis("off")
                continue
            synth_feat = [c for c in feat if c in synth.columns]
            corr_s = synth[synth_feat].corr().reindex(index=feat, columns=feat).fillna(0)

            row_eval = per_class[
                (per_class["model"] == model) & (per_class["class"] == label)
            ]
            dist = row_eval["corr_distance"].values[0] if not row_eval.empty else float("nan")
            draw_hm(axes[idx + 1], corr_s, f"{model}\n(d={dist:.3f})")

        for i in range(len(models) + 1, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        save_fig(fig, f"B_heatmaps_correlation_class{label}")
        plt.close(fig)


# ─── C. Ridge plots ───────────────────────────────────────────────────────────

def plot_ridge(real: pd.DataFrame, per_class: pd.DataFrame):
    print("C. Ridge plots de densidad")

    for label in [0, 1]:
        subset = per_class[per_class["class"] == label].copy()
        if subset.empty:
            continue

        top_models = subset.nsmallest(TOP_N_MODELS, "ks_stat_mean")["model"].tolist()
        real_class = real[real[TARGET_COL] == label]

        all_models_here = ["Real"] + top_models
        colors, _ = assign_colors_markers(top_models)
        colors["Real"] = "#000000"

        for var in RIDGE_VARS:
            if var not in real.columns:
                print(f"  Variable {var} not found, skipping")
                continue

            entries = [("Real", real_class[var].dropna().values)]
            for model in top_models:
                synth = load_synth(model, label)
                if synth is not None and var in synth.columns:
                    entries.append((model, synth[var].dropna().values))

            n = len(entries)
            # Altura proporcional al número de distribuciones
            fig, axes = plt.subplots(
                n, 1, figsize=(5.5, 1.4 * n + 0.8),
                sharex=True,
            )
            if n == 1:
                axes = [axes]

            fig.suptitle(
                f"Density Distribution: {var} (Class {label})",
                fontsize=11,
            )

            all_vals = np.concatenate([e[1] for e in entries])
            x_min = np.percentile(all_vals, 0.5)
            x_max = np.percentile(all_vals, 99.5)
            x_range = np.linspace(x_min, x_max, 500)

            for ax, (name, vals) in zip(axes, entries):
                color = colors.get(name, WONG_PALETTE[0])
                try:
                    kde     = gaussian_kde(vals, bw_method="scott")
                    density = kde(x_range)
                    ax.fill_between(x_range, density, alpha=0.25, color=color)
                    ax.plot(x_range, density, color=color, linewidth=1.2)
                except Exception:
                    ax.hist(vals, bins=20, density=True, color=color,
                            alpha=0.4, linewidth=0.6, edgecolor="black")

                # Mediana
                med = np.median(vals)
                ax.axvline(med, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.7)

                # Etiqueta del modelo a la izquierda
                ax.set_ylabel(name, fontsize=8, rotation=0,
                              labelpad=70, va="center")
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlim(x_min, x_max)

            axes[-1].set_xlabel(var)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.05)

            var_safe = var.lower().replace(" ", "_")
            save_fig(fig, f"C_ridge_{var_safe}_class{label}")
            plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading evaluation data...")
    real, per_class, tstr = load_data()

    plot_scatter_privacy_utility(per_class, tstr)
    plot_heatmaps(real, per_class)
    plot_ridge(real, per_class)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()