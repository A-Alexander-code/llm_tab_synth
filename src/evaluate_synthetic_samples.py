"""
evaluate_synthetic_samples.py

Evaluación completa de muestras sintéticas por modelo y clase:
  - Bloque 1: Privacidad (DCR, NNAA, MIA)
  - Bloque 2: Fidelidad estadística (KS-test, JS divergence, correlación)
  - Bloque 3: Utilidad (TSTR vs TRTR con XGBoost)

Outputs:
  evaluation_results/evaluation_results.csv   — tabla comparativa
  evaluation_results/evaluation_report.txt    — informe en texto plano
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Configuración ───────────────────────────────────────────────────────────

DATA_DIR        = Path("data")
INPUT_FILE      = DATA_DIR / "diabetes-retin.csv"
SYNTH_BASE      = Path("synth_samples")
OUTPUT_DIR      = Path("evaluation_results")
TARGET_COLUMN   = "Outcome"
RANDOM_STATE    = 42
N_BINS          = 20          # bins para JS divergence
NNAA_N_NEIGHBORS = 1          # vecinos para NNAA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Utilidades ──────────────────────────────────────────────────────────────

def load_real_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Real data loaded: %d rows, %d cols", *df.shape)
    return df


def load_synthetic(model: str, label: int) -> pd.DataFrame | None:
    path = SYNTH_BASE / model / f"class_{label}" / f"synthetic_class_{label}.csv"
    if not path.exists():
        logger.warning("Not found: %s", path)
        return None
    df = pd.read_csv(path)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]


def scale(real: np.ndarray, synth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    scaler.fit(real)
    return scaler.transform(real), scaler.transform(synth)


# ─── Bloque 1: Privacidad ────────────────────────────────────────────────────

def dcr(real: np.ndarray, synth: np.ndarray) -> dict:
    """Distance to Closest Record."""
    r, s = scale(real, synth)
    dists = cdist(s, r, metric="euclidean").min(axis=1)
    return {
        "dcr_mean":   float(np.mean(dists)),
        "dcr_median": float(np.median(dists)),
        "dcr_p5":     float(np.percentile(dists, 5)),   # percentil 5: casos más cercanos
    }


def nnaa(real: np.ndarray, synth: np.ndarray) -> dict:
    """
    Nearest Neighbor Adversarial Accuracy.
    Para cada muestra sintética: ¿su vecino más cercano es real o sintético?
    Valor ideal ~0.5 (indistinguible). Valores altos indican que los sintéticos
    están más cerca de otros sintéticos que de reales (baja privacidad/fidelidad).
    """
    r, s = scale(real, synth)
    n = min(len(r), len(s))
    r, s = r[:n], s[:n]

    combined = np.vstack([r, s])
    labels   = np.array([0] * n + [1] * n)   # 0=real, 1=synth

    dists = cdist(combined, combined, metric="euclidean")
    np.fill_diagonal(dists, np.inf)

    nn_labels = labels[dists.argmin(axis=1)]
    # Accuracy: fracción donde el vecino más cercano tiene la misma etiqueta
    nnaa_score = float(np.mean(nn_labels == labels))
    return {"nnaa": nnaa_score}


def membership_inference_attack(real: np.ndarray, synth: np.ndarray) -> dict:
    """
    Membership Inference Attack simplificado.
    Entrena un RF para distinguir real vs sintético usando solo features.
    AUC ~0.5 → buena privacidad. AUC alto → los sintéticos son distinguibles.
    """
    n = min(len(real), len(synth))
    r, s = scale(real[:n], synth[:n])

    X = np.vstack([r, s])
    y = np.array([0] * n + [1] * n)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    return {"mia_auc": float(auc)}


# ─── Bloque 2: Fidelidad estadística ─────────────────────────────────────────

def ks_tests(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: list[str]) -> dict:
    """KS-test por columna. p>0.05 → distribuciones no distinguibles."""
    results = {}
    stats, pvals = [], []
    for col in feature_cols:
        stat, pval = ks_2samp(real[col].dropna(), synth[col].dropna())
        stats.append(stat)
        pvals.append(pval)
    results["ks_stat_mean"]    = float(np.mean(stats))
    results["ks_pval_mean"]    = float(np.mean(pvals))
    results["ks_pval_pass_pct"] = float(np.mean([p > 0.05 for p in pvals]) * 100)
    return results


def js_divergence(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Jensen-Shannon divergence media por columna (0=idénticas, 1=opuestas)."""
    jsd_values = []
    for col in feature_cols:
        r_vals = real[col].dropna().values
        s_vals = synth[col].dropna().values
        all_vals = np.concatenate([r_vals, s_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), N_BINS + 1)
        p, _ = np.histogram(r_vals, bins=bins, density=True)
        q, _ = np.histogram(s_vals, bins=bins, density=True)
        p = p + 1e-10
        q = q + 1e-10
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())
        jsd_values.append(float(jsd))
    return {"jsd_mean": float(np.mean(jsd_values))}


def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Distancia entre matrices de correlación (norma de Frobenius normalizada)."""
    corr_r = real[feature_cols].corr().values
    corr_s = synth[feature_cols].corr().fillna(0).values
    diff   = corr_r - corr_s
    dist   = float(np.linalg.norm(diff, "fro") / len(feature_cols))
    return {"corr_distance": dist}


# ─── Bloque 3: Utilidad (TSTR vs TRTR) ───────────────────────────────────────

def tstr_vs_trtr(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Train on Synthetic Test on Real (TSTR) vs
    Train on Real Test on Real (TRTR).
    Usa el dataset real completo (ambas clases) para test.
    """
    # Dataset real completo para test
    X_real = real_df[feature_cols].values
    y_real = real_df[TARGET_COLUMN].values

    X_tr_real, X_te, y_tr_real, y_te = train_test_split(
        X_real, y_real, test_size=0.3, random_state=RANDOM_STATE, stratify=y_real
    )

    def train_eval(X_train, y_train, X_test, y_test):
        clf = XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        return {
            "auc":  float(roc_auc_score(y_test, proba)),
            "f1":   float(f1_score(y_test, preds, zero_division=0)),
        }

    # TRTR
    trtr = train_eval(X_tr_real, y_tr_real, X_te, y_te)

    # TSTR: combina sintéticos de ambas clases con el target correcto
    X_synth = synth_df[feature_cols].values
    y_synth = synth_df[TARGET_COLUMN].values

    if len(np.unique(y_synth)) < 2:
        return {
            "trtr_auc": trtr["auc"], "trtr_f1": trtr["f1"],
            "tstr_auc": None, "tstr_f1": None,
            "tstr_auc_ratio": None, "tstr_f1_ratio": None,
        }

    tstr = train_eval(X_synth, y_synth, X_te, y_te)

    return {
        "trtr_auc":       trtr["auc"],
        "trtr_f1":        trtr["f1"],
        "tstr_auc":       tstr["auc"],
        "tstr_f1":        tstr["f1"],
        "tstr_auc_ratio": float(tstr["auc"] / trtr["auc"]) if trtr["auc"] > 0 else None,
        "tstr_f1_ratio":  float(tstr["f1"] / trtr["f1"])   if trtr["f1"]  > 0 else None,
    }


# ─── Evaluación principal ────────────────────────────────────────────────────

def evaluate_model_class(
    model: str,
    label: int,
    real_class_df: pd.DataFrame,
    real_full_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    synth_df = load_synthetic(model, label)
    if synth_df is None:
        return None

    # Alinear columnas
    missing = [c for c in feature_cols if c not in synth_df.columns]
    if missing:
        logger.warning("Model %s class %d missing cols: %s", model, label, missing)
        return None

    synth_df = synth_df[feature_cols + [TARGET_COLUMN]].copy() if TARGET_COLUMN in synth_df.columns \
               else synth_df[feature_cols].copy()

    real_arr  = real_class_df[feature_cols].values.astype(float)
    synth_arr = synth_df[feature_cols].values.astype(float)

    result = {"model": model, "class": label, "n_synth": len(synth_df)}

    # Bloque 1: Privacidad
    result.update(dcr(real_arr, synth_arr))
    result.update(nnaa(real_arr, synth_arr))
    result.update(membership_inference_attack(real_arr, synth_arr))

    # Bloque 2: Fidelidad
    result.update(ks_tests(real_class_df[feature_cols], synth_df[feature_cols], feature_cols))
    result.update(js_divergence(real_class_df[feature_cols], synth_df[feature_cols], feature_cols))
    result.update(correlation_distance(real_class_df[feature_cols], synth_df[feature_cols], feature_cols))

    return result


def evaluate_tstr(
    model: str,
    real_full_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    """Combina sintéticos de ambas clases para TSTR."""
    dfs = []
    for label in [0, 1]:
        s = load_synthetic(model, label)
        if s is not None:
            if TARGET_COLUMN not in s.columns:
                s[TARGET_COLUMN] = label
            dfs.append(s)
    if len(dfs) < 2:
        return None
    synth_combined = pd.concat(dfs, ignore_index=True)
    tstr_result = tstr_vs_trtr(real_full_df, synth_combined, feature_cols)
    tstr_result["model"] = model
    return tstr_result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluation started")

    real_df      = load_real_data(INPUT_FILE)
    feature_cols = get_feature_cols(real_df)
    real_class0  = real_df[real_df[TARGET_COLUMN] == 0].reset_index(drop=True)
    real_class1  = real_df[real_df[TARGET_COLUMN] == 1].reset_index(drop=True)

    models = sorted([p.name for p in SYNTH_BASE.iterdir() if p.is_dir()])
    logger.info("Models found: %s", models)

    # ── Evaluación por clase ──────────────────────────────────────────────────
    rows_per_class = []
    for model in models:
        for label, real_class_df in [(0, real_class0), (1, real_class1)]:
            logger.info("Evaluating %s | class %d ...", model, label)
            result = evaluate_model_class(
                model, label, real_class_df, real_df, feature_cols
            )
            if result:
                rows_per_class.append(result)

    df_per_class = pd.DataFrame(rows_per_class)

    # ── TSTR (dataset completo) ───────────────────────────────────────────────
    tstr_rows = []
    for model in models:
        logger.info("TSTR %s ...", model)
        result = evaluate_tstr(model, real_df, feature_cols)
        if result:
            tstr_rows.append(result)

    df_tstr = pd.DataFrame(tstr_rows)

    # ── Guardar CSVs ─────────────────────────────────────────────────────────
    csv_class = OUTPUT_DIR / "evaluation_per_class.csv"
    csv_tstr  = OUTPUT_DIR / "evaluation_tstr.csv"
    df_per_class.to_csv(csv_class, index=False, float_format="%.4f")
    df_tstr.to_csv(csv_tstr,  index=False, float_format="%.4f")
    logger.info("CSVs saved to: %s", OUTPUT_DIR)

    # ── Generar informe en texto plano ────────────────────────────────────────
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("=" * 80 + "\n")
        f.write("INFORME DE EVALUACIÓN DE MUESTRAS SINTÉTICAS\n")
        f.write(f"Fecha: {ts}\n")
        f.write(f"Dataset real: {INPUT_FILE} ({len(real_df)} filas)\n")
        f.write("=" * 80 + "\n\n")

        f.write("MÉTRICAS INCLUIDAS\n")
        f.write("-" * 40 + "\n")
        f.write("PRIVACIDAD:\n")
        f.write("  dcr_mean      : distancia media al vecino real más cercano (↑ mejor)\n")
        f.write("  dcr_median    : distancia mediana al vecino real más cercano (↑ mejor)\n")
        f.write("  dcr_p5        : percentil 5 de DCR, casos más cercanos (↑ mejor)\n")
        f.write("  nnaa          : Nearest Neighbor Adversarial Accuracy (~0.5 ideal)\n")
        f.write("  mia_auc       : AUC del ataque de inferencia (~0.5 ideal)\n")
        f.write("FIDELIDAD:\n")
        f.write("  ks_stat_mean  : estadístico KS medio por columna (↓ mejor)\n")
        f.write("  ks_pval_mean  : p-valor KS medio (↑ mejor, >0.05 no distinguible)\n")
        f.write("  ks_pval_pass_pct: % columnas con p>0.05 (↑ mejor)\n")
        f.write("  jsd_mean      : JS divergence media por columna (↓ mejor, 0=idénticas)\n")
        f.write("  corr_distance : distancia entre matrices de correlación (↓ mejor)\n")
        f.write("UTILIDAD (solo tabla TSTR):\n")
        f.write("  trtr_auc/f1   : baseline real vs real\n")
        f.write("  tstr_auc/f1   : sintético vs real\n")
        f.write("  tstr_auc_ratio: tstr_auc / trtr_auc (↑ mejor, 1.0=perfecto)\n")
        f.write("  tstr_f1_ratio : tstr_f1 / trtr_f1   (↑ mejor, 1.0=perfecto)\n\n")

        for label in [0, 1]:
            f.write("=" * 80 + "\n")
            f.write(f"CLASE {label}\n")
            f.write("=" * 80 + "\n\n")

            subset = df_per_class[df_per_class["class"] == label].copy()
            if subset.empty:
                f.write("  Sin datos.\n\n")
                continue

            # Privacidad
            f.write("── PRIVACIDAD ──\n")
            priv_cols = ["model", "n_synth", "dcr_mean", "dcr_median", "dcr_p5", "nnaa", "mia_auc"]
            priv_cols = [c for c in priv_cols if c in subset.columns]
            f.write(subset[priv_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            f.write("\n\n")

            # Fidelidad
            f.write("── FIDELIDAD ESTADÍSTICA ──\n")
            fid_cols = ["model", "ks_stat_mean", "ks_pval_mean", "ks_pval_pass_pct", "jsd_mean", "corr_distance"]
            fid_cols = [c for c in fid_cols if c in subset.columns]
            f.write(subset[fid_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            f.write("\n\n")

        # TSTR
        f.write("=" * 80 + "\n")
        f.write("UTILIDAD — TSTR vs TRTR (dataset completo)\n")
        f.write("=" * 80 + "\n\n")
        if not df_tstr.empty:
            tstr_cols = ["model", "trtr_auc", "trtr_f1", "tstr_auc", "tstr_f1",
                         "tstr_auc_ratio", "tstr_f1_ratio"]
            tstr_cols = [c for c in tstr_cols if c in df_tstr.columns]
            f.write(df_tstr[tstr_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            f.write("\n\n")

            # Ranking por tstr_auc_ratio
            if "tstr_auc_ratio" in df_tstr.columns:
                ranked = df_tstr[["model", "tstr_auc_ratio", "tstr_f1_ratio"]].dropna()
                ranked = ranked.sort_values("tstr_auc_ratio", ascending=False).reset_index(drop=True)
                ranked.index += 1
                f.write("── RANKING POR UTILIDAD (tstr_auc_ratio) ──\n")
                f.write(ranked.to_string(float_format=lambda x: f"{x:.4f}"))
                f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("FIN DEL INFORME\n")
        f.write("=" * 80 + "\n")

    logger.info("Report saved to: %s", report_path)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()