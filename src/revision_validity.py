"""Observational synthetic validity reports: no filtering or correction."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.revision_a1 import encode
from src.revision_a1_runtime import require


def table_record(frame):
    header = {"columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes], "rows": len(frame)}
    return {**header, "sha256": hashlib.sha256(encode(header) +
            pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()}


def training_reference(X_train, names, spec, fitted_categories):
    """Only prepared train and the explicit source schema; never infer clinical limits."""
    X = np.asarray(X_train)
    require(X.shape[1] == len(names) and np.isfinite(X).all(), "Invalid reference train")
    columns = {}
    for j, name in enumerate(names):
        kind = spec["types"][name]
        domain = [0, 1] if kind == "binary" else spec["categorical_domains"].get(name)
        columns[name] = {"type": kind, "empirical_min": float(X[:, j].min()), "empirical_max": float(X[:, j].max()),
                         "explicit_domain": domain,
                         "train_fractional_count_rows": (int((X[:, j] != np.trunc(X[:, j])).sum()) if kind == "count" else 0),
                         "observed_train_categories": ([float(v) for v in fitted_categories.get(name, [])]
                                                       if kind in {"binary", "categorical"} else None)}
    return {"columns": columns, "range_policy": "train empirical extrema; not clinical domain limits",
            "count_policy": "nonnegative integer semantic type; fractional imputation medians remain reported, not corrected",
            "train_sha256": table_record(pd.DataFrame(X, columns=names))}


def diagnose(frame, X_train, names, reference, *, requested, stage,
             discarded_observed=0, delivered=None, internal_observation=None):
    """Counts are at the wrapper boundary unless internal telemetry is available."""
    require(frame.columns.is_unique, "Duplicate column names cannot be diagnosed unambiguously")
    require(isinstance(requested, int) and requested > 0, "Invalid requested rows")
    missing_names = [n for n in names if n not in frame]
    extras = [n for n in frame if n not in names]
    columns = {}
    finite_rows = np.ones(len(frame), dtype=bool)
    row_issue = np.zeros(len(frame), dtype=bool)
    for name in names:
        if name not in frame:
            row_issue[:] = True
            continue
        values = frame[name]
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        missing = values.isna().to_numpy()
        nonnumeric = ~missing & np.isnan(numeric)
        infinity = np.isinf(numeric)
        finite = np.isfinite(numeric)
        rule = reference["columns"][name]
        kind = rule["type"]
        fractional = finite & (numeric != np.trunc(numeric)) if kind in {"count", "binary", "categorical"} else np.zeros(len(frame), bool)
        negative_count = finite & (numeric < 0) if kind == "count" else np.zeros(len(frame), bool)
        domain = rule["explicit_domain"]
        invalid_domain = finite & ~np.isin(numeric, domain) if domain is not None else np.zeros(len(frame), bool)
        observed = rule["observed_train_categories"]
        unseen = finite & ~np.isin(numeric, observed) if observed else np.zeros(len(frame), bool)
        outside = finite & ((numeric < rule["empirical_min"]) | (numeric > rule["empirical_max"]))
        flags = {"missing": missing, "infinite": infinity, "nonnumeric": nonnumeric,
                 "fractional_for_semantic_type": fractional, "negative_count": negative_count,
                 "invalid_explicit_domain": invalid_domain, "unseen_train_category": unseen,
                 "outside_train_empirical_range": outside}
        columns[name] = {k: {"count": int(mask.sum()), "row_positions": np.flatnonzero(mask).tolist()}
                         for k, mask in flags.items()}
        columns[name]["source_dtype"] = str(values.dtype)
        columns[name]["numeric_storage_dtype"] = bool(pd.api.types.is_numeric_dtype(values.dtype))
        columns[name]["nonfinite_numeric"] = int((~finite & ~nonnumeric).sum())
        row_issue |= missing | nonnumeric | infinity | fractional | negative_count | invalid_domain
        finite_rows &= finite
    if missing_names:
        finite_rows[:] = False
    finite_frame = frame.loc[finite_rows, names].apply(pd.to_numeric) if not missing_names else pd.DataFrame(columns=names)
    train_patterns = set(map(tuple, np.asarray(X_train).tolist()))
    matches = [i for i, row in zip(np.flatnonzero(finite_rows), finite_frame.itertuples(index=False, name=None))
               if tuple(row) in train_patterns]
    return {"schema_version": 1, "stage": stage, "requested_rows": requested,
            "produced_rows_observed": len(frame), "discarded_rows_observed": discarded_observed,
            "delivered_rows": delivered, "unobserved_internal_discarded_rows": None,
            "internal_observation": internal_observation,
            "missing_columns": missing_names, "extra_columns": extras, "columns_in_expected_order": list(frame.columns) == list(names),
            "columns": columns, "rows_with_domain_or_type_issues": int(row_issue.sum()),
            "synthetic_duplicate_rows_extra": int(frame.duplicated().sum()),
            "synthetic_rows_in_duplicate_patterns": int(frame.duplicated(keep=False).sum()),
            "finite_feature_duplicate_rows_extra": int(finite_frame.duplicated().sum()),
            "exact_train_match_rows": len(matches), "exact_train_match_positions": [int(i) for i in matches],
            "exact_match_policy": "all predictors in declared order; exact numeric equality; no rounding; IDs/target excluded",
            "empirical_range_is_domain_constraint": False, "diagnostic_action": "none",
            "input": table_record(frame)}
