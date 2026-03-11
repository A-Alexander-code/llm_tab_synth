"""
plot_evaluation.py

Genera las siguientes gráficas por clase (0 y 1):
  A. Scatter: Privacidad (dcr_mean) vs Utilidad (tstr_auc_ratio)
  B. Heatmaps: Matriz de correlación real vs cada modelo
  C. Ridge plots: Distribución de densidad por variable crítica

Outputs en evaluation_results/plots/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─── Configuración ────────────────────────────────────────────────────────────

DATA_DIR     = Path("data")
INPUT_FILE   = DATA_DIR / "diabetes-retin.csv"
SYNTH_BASE   = Path("synth_samples")
EVAL_DIR     = Path("evaluation_results")
OUTPUT_DIR   = EVAL_DIR / "plots"
TARGET_COL   = "Outcome"

RIDGE_VARS   = ["Glucose", "BMI"]   # variables para ridge plot
TOP_N_MODELS = 3                     # modelos top para ridge plot

# Paleta de colores por modelo
MODEL_COLORS = {
    "great":          "#E63946",
    "tabula":         "#F4A261",
    "realtabformer":  "#2A9D8F",
    "ctgan":          "#457B9D",
    "tvae":           "#6A0572",
    "gaussian_copula":"#1D3557",
    "copulagan":      "#E9C46A",
    "tabpfgen":       "#264653",
    "forestdiffusion":"#A8DADC",
    "mostlyai":       "#E76F51",
}
DEFAULT_COLOR = "#888888"

SAVE_FORMATS = ["png", "pdf"]
DPI = 180


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    real   = pd.read_csv(INPUT_FILE)
    per_class = pd.read_csv(EVAL_DIR / "evaluation_per_class.csv")
    tstr   = pd.read_csv(EVAL_DIR / "evaluation_tstr.csv")
    return real, per_class, tstr


def load_synth(model: str, label: int) -> pd.DataFrame | None:
    p = SYNTH_BASE / model / f"class_{label}" / f"synthetic_class_{label}.csv"
    return pd.read_csv(p) if p.exists() else None


def feature_cols(df):
    return [c for c in df.columns if c != TARGET_COL]


def save_fig(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in SAVE_FORMATS:
        path = OUTPUT_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, DEFAULT_COLOR)


# ─── A. Scatter: Privacidad vs Utilidad ───────────────────────────────────────

def plot_scatter_privacy_utility(per_class: pd.DataFrame, tstr: pd.DataFrame):
    print("A. Scatter: Privacidad vs Utilidad")

    for label in [0, 1]:
        subset = per_class[per_class["class"] == label][["model", "dcr_mean"]].copy()
        merged = subset.merge(tstr[["model", "tstr_auc_ratio"]], on="model", how="inner")
        if merged.empty:
            print(f"  No data for class {label}")
            continue

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#0F1117")

        for _, row in merged.iterrows():
            color = model_color(row["model"])
            ax.scatter(row["dcr_mean"], row["tstr_auc_ratio"],
                       color=color, s=220, zorder=5, edgecolors="white", linewidths=0.8)
            ax.annotate(row["model"],
                        xy=(row["dcr_mean"], row["tstr_auc_ratio"]),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=9.5, color="white", fontweight="bold")

        # Líneas de referencia
        ax.axhline(1.0, color="#444", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(merged["dcr_mean"].median(), color="#444",
                   linestyle="--", linewidth=0.8, alpha=0.6)

        # Anotación de cuadrantes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmid = merged["dcr_mean"].median()
        ymid = 1.0
        ax.text(xmid + (xlim[1] - xmid) * 0.05,
                ylim[0] + (ylim[1] - ylim[0]) * 0.05,
                "Útil\nPoco privado", color="#888", fontsize=8, alpha=0.7)
        ax.text(xlim[0] + (xmid - xlim[0]) * 0.05,
                ylim[0] + (ylim[1] - ylim[0]) * 0.05,
                "No útil\nPrivado", color="#888", fontsize=8, alpha=0.7)
        ax.text(xmid + (xlim[1] - xmid) * 0.05,
                ymid + (ylim[1] - ymid) * 0.55,
                "★ Ideal\n(Útil + Privado)", color="#2A9D8F",
                fontsize=8.5, fontweight="bold", alpha=0.9)

        ax.set_xlabel("dcr_mean  (↑ más privado)", color="white", fontsize=11)
        ax.set_ylabel("tstr_auc_ratio  (↑ más útil)", color="white", fontsize=11)
        ax.set_title(f"Privacidad vs Utilidad — Clase {label}",
                     color="white", fontsize=14, pad=14)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        fig.tight_layout()
        save_fig(fig, f"A_scatter_privacy_utility_class{label}")
        plt.close(fig)


# ─── B. Heatmaps de correlación ───────────────────────────────────────────────

def plot_heatmaps(real: pd.DataFrame, per_class: pd.DataFrame):
    print("B. Heatmaps de correlación")
    feat = feature_cols(real)

    for label in [0, 1]:
        real_class = real[real[TARGET_COL] == label][feat]
        corr_real  = real_class.corr()

        models = sorted([p.name for p in SYNTH_BASE.iterdir() if p.is_dir()])
        n_models = len(models)
        ncols = 3
        nrows = (n_models + 1 + ncols - 1) // ncols  # +1 para el real

        fig = plt.figure(figsize=(ncols * 5.5, nrows * 4.8))
        fig.patch.set_facecolor("#0F1117")
        fig.suptitle(f"Matrices de Correlación — Clase {label}",
                     color="white", fontsize=15, y=1.01)

        axes = []
        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            axes.append(ax)

        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        vmin, vmax = -1, 1

        def draw_heatmap(ax, corr, title, highlight=False):
            ax.set_facecolor("#0F1117")
            sns.heatmap(
                corr, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", annot_kws={"size": 6.5},
                linewidths=0.3, linecolor="#1a1a2e",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title, color="#FFD700" if highlight else "white",
                         fontsize=10, fontweight="bold" if highlight else "normal")
            ax.tick_params(colors="white", labelsize=7)
            for _, spine in ax.spines.items():
                spine.set_visible(False)

        draw_heatmap(axes[0], corr_real, "REAL", highlight=True)

        for idx, model in enumerate(models):
            synth = load_synth(model, label)
            if synth is None:
                axes[idx + 1].axis("off")
                continue
            synth_feat = [c for c in feat if c in synth.columns]
            corr_s = synth[synth_feat].corr().reindex(index=feat, columns=feat).fillna(0)

            subset_row = per_class[
                (per_class["model"] == model) & (per_class["class"] == label)
            ]
            dist = subset_row["corr_distance"].values[0] if not subset_row.empty else float("nan")
            draw_heatmap(axes[idx + 1], corr_s,
                         f"{model}\n(corr_dist={dist:.3f})")

        # Ocultar ejes vacíos
        for i in range(len(models) + 1, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        save_fig(fig, f"B_heatmaps_correlation_class{label}")
        plt.close(fig)


# ─── C. Ridge plots ───────────────────────────────────────────────────────────

def plot_ridge(real: pd.DataFrame, per_class: pd.DataFrame):
    print("C. Ridge plots de densidad")

    # Determinar top N modelos por ks_stat_mean (menor = mejor fidelidad)
    for label in [0, 1]:
        subset = per_class[per_class["class"] == label].copy()
        if subset.empty:
            continue
        top_models = (
            subset.nsmallest(TOP_N_MODELS, "ks_stat_mean")["model"].tolist()
        )

        real_class = real[real[TARGET_COL] == label]

        for var in RIDGE_VARS:
            if var not in real.columns:
                print(f"  Variable {var} not found, skipping")
                continue

            entries = [("Real", real_class[var].dropna().values, "#FFFFFF")]
            for model in top_models:
                synth = load_synth(model, label)
                if synth is not None and var in synth.columns:
                    entries.append((model, synth[var].dropna().values,
                                    model_color(model)))

            n = len(entries)
            fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n + 1),
                                     sharex=True)
            if n == 1:
                axes = [axes]
            fig.patch.set_facecolor("#0F1117")
            fig.suptitle(
                f"Distribución de densidad: {var} — Clase {label}\n"
                f"(Top {TOP_N_MODELS} modelos por fidelidad vs Real)",
                color="white", fontsize=13, y=1.01
            )

            # Rango común
            all_vals = np.concatenate([e[1] for e in entries])
            x_min, x_max = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
            x_range = np.linspace(x_min, x_max, 400)

            for ax, (name, vals, color) in zip(axes, entries):
                ax.set_facecolor("#0F1117")
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    density = kde(x_range)
                    ax.fill_between(x_range, density, alpha=0.45, color=color)
                    ax.plot(x_range, density, color=color, linewidth=1.8)
                except Exception:
                    ax.hist(vals, bins=25, density=True,
                            color=color, alpha=0.5)

                ax.set_ylabel(name, color="white", fontsize=9,
                              rotation=0, labelpad=60, va="center")
                ax.tick_params(colors="white", labelsize=8)
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                ax.set_xlim(x_min, x_max)

            axes[-1].set_xlabel(var, color="white", fontsize=11)
            fig.tight_layout()
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