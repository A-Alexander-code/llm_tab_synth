"""Privacy metrics for synthetic tabular data."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.degenerate import get_valid_columns_for_statistical_metrics 

_BLOCK_EXACT = "privacy_exact"
_BLOCK_DCR = "privacy_dcr"
_BLOCK_MIA = "privacy_mia"
_BLOCK_ATTINF = "privacy_attinf"
_BLOCK_NNAA = "privacy_nnaa"
_BLOCK_KANON = "privacy_kanonymity"


def _ok(block, metric, value, classifier=None):
    return {"block": block, "metric": metric, "classifier": classifier,
            "value": value, "status": "ok", "reason": None}


def _not_computable(block, metric, reason, classifier=None):
    return {"block": block, "metric": metric, "classifier": classifier,
            "value": float("nan"), "status": "not_computable", "reason": reason}


def compute_exact_matches(synth: pd.DataFrame, real_train: pd.DataFrame) -> list[dict[str, Any]]:
    """Count exact rounded row matches between synthetic and training data."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return [_not_computable(_BLOCK_EXACT, "exact_matches_count", "no_numeric_columns"),
                _not_computable(_BLOCK_EXACT, "exact_matches_ratio", "no_numeric_columns")]

    real_rounded = real_train[numeric_cols].round(6)
    common = [c for c in numeric_cols if c in synth.columns]
    synth_rounded = synth[common].round(6) if common else pd.DataFrame()

    if synth_rounded.empty:
        return [_not_computable(_BLOCK_EXACT, "exact_matches_count", "no_common_columns"),
                _not_computable(_BLOCK_EXACT, "exact_matches_ratio", "no_common_columns")]

    real_set = set(map(tuple, real_rounded[common].values))
    n_exact = sum(1 for row in synth_rounded.itertuples(index=False) if tuple(row) in real_set)
    ratio = n_exact / max(len(synth_rounded), 1)

    return [
        _ok(_BLOCK_EXACT, "exact_matches_count", float(n_exact)),
        _ok(_BLOCK_EXACT, "exact_matches_ratio", ratio),
    ]


def compute_dcr(synth: pd.DataFrame, real_train: pd.DataFrame) -> list[dict[str, Any]]:
    """Measure distances from synthetic rows to their closest training records."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols  = [c for c in numeric_cols if c in synth.columns]
    if not common_cols:
        return [_not_computable(_BLOCK_DCR, m, "no_common_columns")
                for m in ("dcr_mean", "dcr_median", "dcr_p5")]

    scaler = StandardScaler()
    real_scaled  = scaler.fit_transform(real_train[common_cols].fillna(0))
    synth_scaled = scaler.transform(synth[common_cols].fillna(0))

    tree = BallTree(real_scaled, metric="euclidean")
    dists = tree.query(synth_scaled, k=1)[0][:, 0]

    return [
        _ok(_BLOCK_DCR, "dcr_mean", float(np.mean(dists))),
        _ok(_BLOCK_DCR, "dcr_median", float(np.median(dists))),
        _ok(_BLOCK_DCR, "dcr_p5", float(np.percentile(dists, 5))),
    ]


def compute_dcr_holdout(real_test: pd.DataFrame, real_train: pd.DataFrame) -> list[dict[str, Any]]:
    """Measure holdout-to-train nearest-neighbor distances as a baseline."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols  = [c for c in numeric_cols if c in real_test.columns]
    if not common_cols:
        return [_not_computable(_BLOCK_DCR, m, "no_common_columns")
                for m in ("dcr_holdout_mean", "dcr_holdout_median", "dcr_holdout_p5")]

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_train[common_cols].fillna(0))
    test_scaled = scaler.transform(real_test[common_cols].fillna(0))

    tree = BallTree(real_scaled, metric="euclidean")
    dists = tree.query(test_scaled, k=1)[0][:, 0]

    return [
        _ok(_BLOCK_DCR, "dcr_holdout_mean", float(np.mean(dists))),
        _ok(_BLOCK_DCR, "dcr_holdout_median", float(np.median(dists))),
        _ok(_BLOCK_DCR, "dcr_holdout_p5", float(np.percentile(dists, 5))),
    ]


def compute_mia(synth: pd.DataFrame, 
                real_train: pd.DataFrame,
                real_test: pd.DataFrame, 
                seed: int = 42,) -> list[dict[str, Any]]:
    """Run a distance-based membership inference attack."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols  = [c for c in numeric_cols if c in synth.columns and c in real_test.columns]
    if not common_cols:
        return [_not_computable(_BLOCK_MIA, m, "no_common_columns")
                for m in ("mia_auc", "mia_advantage", "mia_tpr_at_fpr01")]

    scaler = StandardScaler()
    synth_scaled = scaler.fit_transform(synth[common_cols].fillna(0))
    tree = BallTree(synth_scaled, metric="euclidean")

    def _attack_features(df: pd.DataFrame) -> np.ndarray:
        return tree.query(scaler.transform(df[common_cols].fillna(0)), k=1)[0]

    n = min(len(real_train), len(real_test))
    train_feats = _attack_features(real_train.sample(n=n, random_state=seed))
    test_feats  = _attack_features(real_test.sample(n=n, random_state=seed))

    X_attack = np.vstack([train_feats, test_feats])
    y_attack = np.array([1] * n + [0] * n)

    clf = Pipeline([("scaler", StandardScaler()),
                         ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1))])
    n_splits = min(5, n // 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_proba = np.zeros(len(y_attack))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X_attack, y_attack):
            clf.fit(X_attack[train_idx], y_attack[train_idx])
            all_proba[test_idx] = clf.predict_proba(X_attack[test_idx])[:, 1]

    auc = float(roc_auc_score(y_attack, all_proba))
    pred = (all_proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_attack == 1)).sum())
    fp = int(((pred == 1) & (y_attack == 0)).sum())
    fn = int(((pred == 0) & (y_attack == 1)).sum())
    tn = int(((pred == 0) & (y_attack == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    fpr_vals, tpr_vals, _ = roc_curve(y_attack, all_proba)
    mask = fpr_vals <= 0.1
    tpr_at_fpr01 = float(tpr_vals[mask].max()) if mask.any() else 0.0

    return [
        _ok(_BLOCK_MIA, "mia_auc", auc),
        _ok(_BLOCK_MIA, "mia_advantage", float(tpr - fpr)),
        _ok(_BLOCK_MIA, "mia_tpr_at_fpr01", tpr_at_fpr01),
    ]


def _detect_sensitive_column(real_train: pd.DataFrame, target: str = "class") -> str | None:
    """Return the non-target numeric column most correlated with the target."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    candidates   = [c for c in numeric_cols if c != target]
    if not candidates or target not in real_train.columns:
        return None
    return str(real_train[candidates].corrwith(real_train[target]).abs().idxmax())


def compute_attribute_inference(synth: pd.DataFrame, 
                                real_train: pd.DataFrame, 
                                real_test: pd.DataFrame, 
                                target: str = "class", 
                                seed: int = 42,) -> list[dict[str, Any]]:
    """Measure attribute inference risk on the most target-correlated column.""" 

    real_feat = real_train.drop(columns=[target], errors="ignore")
    synth_feat = synth.drop(columns=[target], errors="ignore")
    test_feat = real_test.drop(columns=[target], errors="ignore")

    vcr = get_valid_columns_for_statistical_metrics(real_feat, synth_feat)
    valid_cols = vcr.valid_cols

    if len(valid_cols) < 2:
        reason = (f"insufficient_valid_features (valid={len(valid_cols)}, "
                  f"excluded={len(vcr.excluded_cols)})")
        return [_not_computable(_BLOCK_ATTINF, m, reason)
                for m in ("attinf_sensitive_col", "attinf_acc_real_train",
                          "attinf_acc_real_test",  "attinf_acc_synth")]

    real_valid = real_train[valid_cols + [target]] if target in real_train.columns else real_train[valid_cols]
    sensitive_col = _detect_sensitive_column(real_valid, target)

    if sensitive_col is None:
        return [_not_computable(_BLOCK_ATTINF, "attinf_sensitive_col", "no_sensitive_col_detected")]

    feature_cols = [c for c in valid_cols if c != sensitive_col]
    if not feature_cols:
        return [_not_computable(_BLOCK_ATTINF, "attinf_sensitive_col",
                                "no_features_after_excluding_sensitive")]

    def _prepare(df):
        X = df[feature_cols].fillna(0).values if all(c in df.columns for c in feature_cols) else None
        y = df[sensitive_col].fillna(0).values if sensitive_col in df.columns else None
        return X, y

    X_train, y_train = _prepare(real_train)
    X_test, y_test = _prepare(real_test)
    X_synth, y_synth = _prepare(synth)

    if any(x is None for x in (X_train, y_train, X_test, y_test, X_synth, y_synth)):
        return [_not_computable(_BLOCK_ATTINF, "attinf_sensitive_col", "missing_columns")]

    median_val = float(np.median(y_train))
    y_train_bin = (y_train > median_val).astype(int)
    y_test_bin  = (y_test  > median_val).astype(int)
    y_synth_bin = (y_synth > median_val).astype(int)

    clf = Pipeline([("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1))])

    rows: list[dict[str, Any]] = [
        _ok(_BLOCK_ATTINF, "attinf_sensitive_col", sensitive_col),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X_train, y_train_bin)
            rows += [
                _ok(_BLOCK_ATTINF, "attinf_acc_real_train",
                    float(balanced_accuracy_score(y_train_bin, clf.predict(X_train)))),
                _ok(_BLOCK_ATTINF, "attinf_acc_real_test",
                    float(balanced_accuracy_score(y_test_bin,  clf.predict(X_test)))),
                _ok(_BLOCK_ATTINF, "attinf_acc_synth",
                    float(balanced_accuracy_score(y_synth_bin, clf.predict(X_synth)))),
            ]
        except Exception as e:
            rows.append(_not_computable(_BLOCK_ATTINF, "attinf_error", str(e)))

    return rows


def compute_nnaa(synth: pd.DataFrame, real_train: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute nearest-neighbor adversarial accuracy."""
    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in synth.columns]
    if not common_cols:
        return [_not_computable(_BLOCK_NNAA, m, "no_common_columns")
                for m in ("nnaa", "nnaa_acc_real", "nnaa_acc_synth")]

    scaler = StandardScaler()
    real_scaled  = scaler.fit_transform(real_train[common_cols].fillna(0))
    synth_scaled = scaler.transform(synth[common_cols].fillna(0))

    n_real = len(real_scaled)
    n_synth = len(synth_scaled)

    tree_real = BallTree(real_scaled, metric="euclidean")
    tree_synth = BallTree(synth_scaled, metric="euclidean")

    d_r_real = tree_real.query(real_scaled, k=min(2, n_real))[0][:, -1]
    d_s_real = tree_synth.query(real_scaled, k=1)[0][:, 0]
    acc_real = float(np.mean(d_r_real < d_s_real))

    d_s_synth = tree_synth.query(synth_scaled, k=min(2, n_synth))[0][:, -1]
    d_r_synth = tree_real.query(synth_scaled,  k=1)[0][:, 0]
    acc_synth = float(np.mean(d_s_synth < d_r_synth))

    nnaa = 0.5 * (acc_real + acc_synth)

    return [
        _ok(_BLOCK_NNAA, "nnaa", nnaa),
        _ok(_BLOCK_NNAA, "nnaa_acc_real", acc_real),
        _ok(_BLOCK_NNAA, "nnaa_acc_synth", acc_synth),
    ]


def compute_kanonymity(synth: pd.DataFrame, 
                       real_train: pd.DataFrame, 
                       real_test: pd.DataFrame, 
                       k_values: list[int] | None = None, 
                       epsilon_percentile: float = 25.0,) -> list[dict[str, Any]]:
    """Measure synthetic k-anonymity against the real training set."""
    if k_values is None:
        k_values = [3, 5]

    numeric_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in synth.columns and c in real_test.columns]
    if not common_cols:
        metrics = ["kanon_epsilon"] + [f"kanon_k{k}_{s}" for k in k_values for s in ("ratio", "count")]
        return [_not_computable(_BLOCK_KANON, m, "no_common_columns") for m in metrics]

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_train[common_cols].fillna(0))
    test_scaled = scaler.transform(real_test[common_cols].fillna(0))

    tree_real = BallTree(real_scaled, metric="euclidean")
    holdout_dists = tree_real.query(test_scaled, k=1)[0][:, 0]
    epsilon = float(np.percentile(holdout_dists, epsilon_percentile))
    if epsilon <= 0:
        epsilon = 1e-6

    synth_scaled = scaler.transform(synth[common_cols].fillna(0))
    neighbor_counts = tree_real.query_radius(synth_scaled, r=epsilon, count_only=True)

    rows: list[dict[str, Any]] = [_ok(_BLOCK_KANON, "kanon_epsilon", epsilon)]

    for k in k_values:
        is_kanon = neighbor_counts >= k
        kanon_count = int(is_kanon.sum())
        kanon_ratio = float(kanon_count / max(len(synth_scaled), 1))
        rows += [
            _ok(_BLOCK_KANON, f"kanon_k{k}_ratio", kanon_ratio),
            _ok(_BLOCK_KANON, f"kanon_k{k}_count", float(kanon_count)),
        ]

    return rows
