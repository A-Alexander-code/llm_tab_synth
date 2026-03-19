"""Privacy metrics for synthetic tabular data."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

_BLOCK_EXACT = "privacy_exact"
_BLOCK_DCR = "privacy_dcr"
_BLOCK_MIA = "privacy_mia"
_BLOCK_ATTINF = "privacy_attinf"

def compute_exact_matches(synth: pd.DataFrame, 
                          real_train: pd.DataFrame,) -> list[dict[str, Any]]:
    """Count exact row matches between synthetic and training data."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    real_rounded = real_train[numeric_cols].round(6)
    synth_rounded = synth[numeric_cols].round(6) if set(numeric_cols).issubset(synth.columns) else pd.DataFrame()

    if synth_rounded.empty:
        return []

    real_set = set(map(tuple, real_rounded.values))
    n_exact = sum(1 for row in synth_rounded.itertuples(index=False) if tuple(row) in real_set)
    ratio = n_exact / max(len(synth_rounded), 1)

    return [
        {"block": _BLOCK_EXACT, "metric": "exact_matches_count", "classifier": None, "value": float(n_exact)},
        {"block": _BLOCK_EXACT, "metric": "exact_matches_ratio", "classifier": None, "value": ratio},
    ]

def compute_dcr(synth: pd.DataFrame, 
                real_train: pd.DataFrame, 
                n_neighbors: int = 1,) -> list[dict[str, Any]]:
    """Measure distances from synthetic rows to their closest training records."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in synth.columns]
    if not common_cols:
        return []

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_train[common_cols].fillna(0))
    synth_scaled = scaler.transform(synth[common_cols].fillna(0))

    tree = BallTree(real_scaled, metric="euclidean")
    dists, _ = tree.query(synth_scaled, k=n_neighbors)
    dists = dists[:, 0]

    return [
        {"block": _BLOCK_DCR, "metric": "dcr_mean",   "classifier": None, "value": float(np.mean(dists))},
        {"block": _BLOCK_DCR, "metric": "dcr_median", "classifier": None, "value": float(np.median(dists))},
        {"block": _BLOCK_DCR, "metric": "dcr_p5",     "classifier": None, "value": float(np.percentile(dists, 5))},
    ]


def compute_dcr_holdout(real_test: pd.DataFrame, 
                        real_train: pd.DataFrame,) -> list[dict[str, Any]]:
    """Measure holdout-to-train nearest-neighbor distances as a baseline."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols  = [c for c in numeric_cols if c in real_test.columns]
    if not common_cols:
        return []

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_train[common_cols].fillna(0))
    test_scaled = scaler.transform(real_test[common_cols].fillna(0))

    tree = BallTree(real_scaled, metric="euclidean")
    dists, _ = tree.query(test_scaled, k=1)
    dists = dists[:, 0]

    return [
        {"block": _BLOCK_DCR, "metric": "dcr_holdout_mean",   "classifier": None, "value": float(np.mean(dists))},
        {"block": _BLOCK_DCR, "metric": "dcr_holdout_median", "classifier": None, "value": float(np.median(dists))},
        {"block": _BLOCK_DCR, "metric": "dcr_holdout_p5",     "classifier": None, "value": float(np.percentile(dists, 5))},
    ]

def compute_mia(synth: pd.DataFrame, 
                real_train: pd.DataFrame, 
                real_test: pd.DataFrame, 
                seed: int = 42,) -> list[dict[str, Any]]:
    """Run a distance-based membership inference attack."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols  = [c for c in numeric_cols if c in synth.columns and c in real_test.columns]
    if not common_cols:
        return []

    scaler = StandardScaler()
    synth_scaled = scaler.fit_transform(synth[common_cols].fillna(0))
    tree = BallTree(synth_scaled, metric="euclidean")

    def _attack_features(df: pd.DataFrame) -> np.ndarray:
        X = scaler.transform(df[common_cols].fillna(0))
        dists, _ = tree.query(X, k=1)
        return dists

    n = min(len(real_train), len(real_test))
    train_feats = _attack_features(real_train.sample(n=n, random_state=seed))
    test_feats  = _attack_features(real_test.sample(n=n, random_state=seed))

    X_attack = np.vstack([train_feats, test_feats])
    y_attack = np.array([1] * n + [0] * n)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
    ])

    n_splits = min(5, n // 2)
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_proba = np.zeros(len(y_attack))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X_attack, y_attack):
            clf.fit(X_attack[train_idx], y_attack[train_idx])
            all_proba[test_idx] = clf.predict_proba(X_attack[test_idx])[:, 1]

    auc  = float(roc_auc_score(y_attack, all_proba))

    pred = (all_proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_attack == 1)).sum())
    fp = int(((pred == 1) & (y_attack == 0)).sum())
    fn = int(((pred == 0) & (y_attack == 1)).sum())
    tn = int(((pred == 0) & (y_attack == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    advantage = float(tpr - fpr)

    fpr_vals, tpr_vals, _ = roc_curve(y_attack, all_proba)
    mask = fpr_vals <= 0.1
    tpr_at_fpr01 = float(tpr_vals[mask].max()) if mask.any() else 0.0

    return [
        {"block": _BLOCK_MIA, "metric": "mia_auc",          "classifier": None, "value": auc},
        {"block": _BLOCK_MIA, "metric": "mia_advantage",    "classifier": None, "value": advantage},
        {"block": _BLOCK_MIA, "metric": "mia_tpr_at_fpr01", "classifier": None, "value": tpr_at_fpr01},
    ]

def _detect_sensitive_column(real_train: pd.DataFrame, 
                             target: str = "class",) -> str | None:
    """Return the non-target numeric column most correlated with the target."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if c != target]
    if not candidates or target not in real_train.columns:
        return None

    corrs = real_train[candidates].corrwith(real_train[target]).abs()
    return str(corrs.idxmax())


def compute_attribute_inference(synth: pd.DataFrame, 
                                real_train: pd.DataFrame, 
                                real_test: pd.DataFrame, 
                                target: str = "class", 
                                seed: int = 42,) -> list[dict[str, Any]]:
    """Measure attribute inference risk on the most target-correlated column."""
    sensitive_col = _detect_sensitive_column(real_train, target)
    if sensitive_col is None:
        return []

    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != sensitive_col and c in synth.columns]
    if not feature_cols:
        return []

    def _prepare(df: pd.DataFrame):
        X = df[feature_cols].fillna(0).values
        y = df[sensitive_col].fillna(0).values
        return X, y

    X_train, y_train = _prepare(real_train)
    X_test,  y_test = _prepare(real_test)
    X_synth, y_synth = _prepare(synth)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
    ])

    median_val = float(np.median(y_train))
    y_train_bin = (y_train > median_val).astype(int)
    y_test_bin = (y_test  > median_val).astype(int)
    y_synth_bin = (y_synth > median_val).astype(int)

    rows: list[dict[str, Any]] = [
        {"block": _BLOCK_ATTINF, "metric": "attinf_sensitive_col", "classifier": None, "value": sensitive_col},
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X_train, y_train_bin)
            from sklearn.metrics import balanced_accuracy_score
            acc_train = float(balanced_accuracy_score(y_train_bin, clf.predict(X_train)))
            acc_test = float(balanced_accuracy_score(y_test_bin,  clf.predict(X_test)))
            acc_synth = float(balanced_accuracy_score(y_synth_bin, clf.predict(X_synth)))

            rows += [
                {"block": _BLOCK_ATTINF, "metric": "attinf_acc_real_train", "classifier": None, "value": acc_train},
                {"block": _BLOCK_ATTINF, "metric": "attinf_acc_real_test",  "classifier": None, "value": acc_test},
                {"block": _BLOCK_ATTINF, "metric": "attinf_acc_synth",      "classifier": None, "value": acc_synth},
            ]
        except Exception as e:
            rows.append({"block": _BLOCK_ATTINF, "metric": "attinf_error", "classifier": None, "value": str(e)})

    return rows
