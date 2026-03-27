"""Utility metrics for synthetic tabular data."""

from __future__ import annotations
 
import copy
import warnings
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

_BLOCK_UNIVARIATE = "utility_univariate"
_BLOCK_MULTIVARIATE = "utility_multivariate"
_BLOCK_DETECTABILITY = "utility_detectability"
_BLOCK_ML = "utility_ml"

_N_CV_FOLDS = 5

def _ok(block: str, metric: str, value: float, classifier: str | None = None) -> dict[str, Any]:
    return {"block": block, "metric": metric, "classifier": classifier,
            "value": value, "status": "ok", "reason": None}
 
 
def _not_computable(block: str, metric: str, reason: str, classifier: str | None = None) -> dict[str, Any]:
    return {"block": block, "metric": metric, "classifier": classifier,
            "value": float("nan"), "status": "not_computable", "reason": reason}
 
 
def _histogram_probs(a: np.ndarray, b: np.ndarray, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return p, q
 
 
def compute_univariate(real: pd.DataFrame, synth: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-column and mean univariate distance metrics."""
    rows = []
    numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    ks_vals, wd_vals, jsd_vals = [], [], []
 
    for col in numeric_cols:
        a = real[col].dropna().values.astype(float)
        b = synth[col].dropna().values.astype(float) if col in synth.columns else np.array([])
        if len(a) == 0 or len(b) == 0:
            continue
 
        ks_stat, _ = ks_2samp(a, b)
        wd = wasserstein_distance(a, b)
        p, q = _histogram_probs(a, b)
        jsd = float(jensenshannon(p, q))
 
        ks_vals.append(ks_stat)
        wd_vals.append(wd)
        jsd_vals.append(jsd)
 
        rows += [
            _ok(_BLOCK_UNIVARIATE, f"ks_{col}", ks_stat),
            _ok(_BLOCK_UNIVARIATE, f"wasserstein_{col}", wd),
            _ok(_BLOCK_UNIVARIATE, f"jsd_{col}", jsd),
        ]
 
    if ks_vals:
        rows += [
            _ok(_BLOCK_UNIVARIATE, "ks_mean", float(np.mean(ks_vals))),
            _ok(_BLOCK_UNIVARIATE, "wasserstein_mean", float(np.mean(wd_vals))),
            _ok(_BLOCK_UNIVARIATE, "jsd_mean", float(np.mean(jsd_vals))),
        ]
    return rows
 
 
def compute_multivariate(real: pd.DataFrame, synth: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute the Frobenius distance between filtered correlation matrices."""
    from src.evaluation.degenerate import get_valid_columns_for_statistical_metrics  # noqa
 
    vcr = get_valid_columns_for_statistical_metrics(real, synth)
    valid_cols = vcr.valid_cols
 
    if len(valid_cols) < 2:
        return [_not_computable(
            _BLOCK_MULTIVARIATE, "corr_frobenius",
            f"insufficient_valid_features (valid={len(valid_cols)}, "
            f"excluded={len(vcr.excluded_cols)})"
        )]
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_real = real[valid_cols].corr(method="pearson").fillna(0).values
        corr_synth = synth[valid_cols].corr(method="pearson").fillna(0).values
 
    frobenius = float(np.linalg.norm(corr_real - corr_synth, ord="fro"))
    return [_ok(_BLOCK_MULTIVARIATE, "corr_frobenius", frobenius)]
 
 
def compute_detectability(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 42) -> list[dict[str, Any]]:
    """Measure how well logistic regression separates real from synthetic data."""
    numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return [
            _not_computable(_BLOCK_DETECTABILITY, "propensity_auc", "no_numeric_columns"),
            _not_computable(_BLOCK_DETECTABILITY, "pmse", "no_numeric_columns"),
        ]
 
    real_feat = real[numeric_cols].fillna(0)
    synth_feat = synth[numeric_cols].fillna(0)
    n = min(len(real_feat), len(synth_feat))
    real_feat  = real_feat.sample(n=n, random_state=seed).reset_index(drop=True)
    synth_feat = synth_feat.sample(n=n, random_state=seed).reset_index(drop=True)
 
    X = pd.concat([real_feat, synth_feat], ignore_index=True).values
    y = np.array([1] * n + [0] * n)
 
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    skf = StratifiedKFold(n_splits=min(_N_CV_FOLDS, n // 2), shuffle=True, random_state=seed)
    aucs, pmses = [], []
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))
            pmses.append(float(np.mean((proba - 0.5) ** 2) / 0.25))
 
    return [
        _ok(_BLOCK_DETECTABILITY, "propensity_auc", float(np.mean(aucs))),
        _ok(_BLOCK_DETECTABILITY, "pmse", float(np.mean(pmses))),
    ]
 
 
def _get_classifiers(seed: int) -> dict[str, Any]:
    clfs: dict[str, Any] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
    }
    if _XGBOOST_AVAILABLE:
        clfs["xgboost"] = XGBClassifier(
            n_estimators=100, random_state=seed,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
        )
    return clfs
 
 
def _evaluate_classifier(clf, X_train: np.ndarray, y_train: np.ndarray, 
                         X_test:  np.ndarray, y_test:  np.ndarray, 
                         clf_name: str, 
                         mode: str, 
                         save_path: Path | None = None,) -> list[dict[str, Any]]:
    """Fit one classifier, optionally persist it, and return evaluation rows."""
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred  = clf.predict(X_test)
 
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(clf, f)
 
    return [
        _ok(_BLOCK_ML, f"{mode}_aucroc", float(roc_auc_score(y_test, proba)),            clf_name),
        _ok(_BLOCK_ML, f"{mode}_auprc", float(average_precision_score(y_test, proba)),  clf_name),
        _ok(_BLOCK_ML, f"{mode}_f1", float(f1_score(y_test, pred, zero_division=0)), clf_name),
    ]
 
 
def compute_ml_efficacy(real_train: pd.DataFrame, 
                        real_test: pd.DataFrame, 
                        synth: pd.DataFrame, 
                        target: str = "class", 
                        seed: int = 42, 
                        classifiers_dir: Path | None = None,) -> list[dict[str, Any]]:
    """Compute TRTR and TSTR metrics across the configured classifiers."""
    rows = []
 
    def _split(df: pd.DataFrame):
        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0).values
        y = df[target].values
        return X, y
 
    X_real_train, y_real_train = _split(real_train)
    X_real_test, y_real_test  = _split(real_test)
    X_synth, y_synth = _split(synth)
 
    clfs = _get_classifiers(seed)
 
    for clf_name, clf in clfs.items():
        tstr_save = (
            classifiers_dir.joinpath(f"{clf_name}_tstr.pkl")
            if classifiers_dir is not None else None
        )
        try:
            rows += _evaluate_classifier(copy.deepcopy(clf),
                X_real_train, y_real_train, X_real_test, y_real_test,
                clf_name, "trtr", save_path=None)
        except Exception as e:
            rows.append(_not_computable(_BLOCK_ML, f"trtr_{clf_name}", str(e), clf_name))
 
        try:
            rows += _evaluate_classifier(copy.deepcopy(clf),
                X_synth, y_synth, X_real_test, y_real_test,
                clf_name, "tstr", save_path=tstr_save)
        except Exception as e:
            rows.append(_not_computable(_BLOCK_ML, f"tstr_{clf_name}", str(e), clf_name))
 
    return rows
