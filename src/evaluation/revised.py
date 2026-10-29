"""Matched-train utility/privacy protocol, called only by main_evaluator.py.

Generation identities and inputs are read-only. Shared real partitions, attack
design, TRTR and selection evidence are persisted once, in a new evaluation tree.
Historical evaluators and generators retain their existing contracts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config.const import ROOT_DIR, REVISION_A1_SOURCE_FILES
from src.revision_a1 import digest, write_json
from src.revision_a1_runtime import (PINNED_RUN, RevisedRun, frame_record, read_json,
                                    real_numeric_matrix, require, runtime_record, value_hash)
from src.revision_utility import json_config
from src.evaluation.utility import _get_classifiers, binary_results, positive_probability
from src.evaluation.privacy import (MixedGeometry, calibrate_membership_threshold,
    distance_summary, explicit_attack_forest, grouped_rate_intervals, mixed_nnaa)

MODELS = ("ctgan", "copulagan", "tvae", "gaussian_copula", "mostlyai", "great", "realtabformer", "tabula")
SEEDS = (13, 23, 42, 69, 88)
QUOTAS = {"heart_framingham": (2877, 515), "diabetes_prima_indian": (400, 214),
          "diabetic_retinopathy": (432, 489), "cervical_cancer": (642, 44)}
SECRETS = {"heart_framingham": ("diabetes", ["age", "sex"], False),
           "diabetes_prima_indian": ("pregnancies", ["age", "bmi"], True),
           "cervical_cancer": ("STDs", ["Age", "Smokes"], False),
           "diabetic_retinopathy": ("ma1", ["macula_opticdisc_distance", "opticdisc_diameter"], True)}
LONG_COLUMNS = ("dataset", "seed", "generador", "evaluation_identity", "component", "condition",
    "classifier", "attacker", "attribute", "population", "class", "fold", "metric", "value", "unit",
    "n_rows", "n_train", "n_test", "n_members", "n_nonmembers", "tp", "fp", "tn", "fn",
    "coverage", "status", "reason", "evidence", "details")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    """Mutable execution state only; immutable configurations use write_json."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")


def result(metric, value, *, unit="ratio", status="ok", reason=None, **fields):
    if isinstance(value, np.generic):
        value = value.item()
    if value is not None and isinstance(value, (float, int)) and not np.isfinite(value):
        value, status, reason = None, "not_computable", reason or "nonfinite_result"
    return dict(metric=metric, value=value, unit=unit, status=status, reason=reason, **fields)


def strict_frame(frame, names, label):
    require(frame.columns.is_unique and list(frame) == names + ["class"], "Synthetic schema/order mismatch")
    real_numeric_matrix(frame)
    require(len(frame) > 0 and (frame["class"] == label).all(), "Synthetic label/empty pool mismatch")
    return frame


def read_pools(split, model, root, run):
    """Named pools with historical integrity plus explicitly bounded extension.

    No generation directory/code hash is reconstructed from the evaluator.
    Unknown extension events stay unknown and old manifests are never changed.
    """
    require(model in MODELS, "Generator excluded from this evaluation protocol")
    directory = Path(root) / split.dataset / f"seed_{split.seed}" / model
    identity = read_json(directory / "identity.json")
    manifest = read_json(directory / "samples_manifest.json")
    parameters = read_json(directory / "generator_parameters.json")
    require(manifest["identity"] == identity, "Historical identity/manifest mismatch")
    require(parameters == identity["generator_parameters"], "Historical parameter record mismatch")
    require((identity["run_id"], identity["dataset"], identity["seed"], identity["model"]) ==
            (PINNED_RUN, split.dataset, split.seed, model), "Historical source routing mismatch")
    require(identity["artifact_hashes"] == split.artifact_hashes, "Historical prepared-input hashes mismatch")
    require(identity["run_manifest_sha256"] == digest(run.directory / "run_manifest.json"),
            "Historical preparation manifest mismatch")
    require(identity["config"]["revision_a1"] == run.policy, "Historical preparation policy mismatch")
    require(set(manifest["files"]) == {"synthetic_class_0.csv", "synthetic_class_1.csv"},
            "Historical sample inventory mismatch")
    evidence = {"historical_identity": identity, "historical_identity_digest": value_hash(identity),
                "historical_files": {}, "pools": {}, "extensions": [],
                "historical_code_policy": "recorded provenance; not equated to current evaluation code"}
    for name in ("identity.json", "generator_parameters.json", "samples_manifest.json"):
        evidence["historical_files"][name] = {"path": str(directory / name), "sha256": digest(directory / name)}
    for name, expected in manifest.get("observability_files", {}).items():
        require(Path(name).name == name, "Unsafe historical observability filename")
        require(digest(directory / name) == expected, "Historical raw/diagnostic hash mismatch: " + name)
        evidence["historical_files"][name] = {"path": str(directory / name), "sha256": expected}
    pools = []
    for label in (0, 1):
        name = f"synthetic_class_{label}.csv"
        path = directory / name
        actual = digest(path)
        frame = strict_frame(pd.read_csv(path, float_precision="round_trip"), split.names, label)
        expected = manifest["files"][name]
        if actual != expected:
            require(split.dataset == "heart_framingham" and label == 0 and len(frame) == 2900,
                    "Unregistered pool content mismatch")
            backup = directory / "synthetic_class_0_original.csv"
            require(digest(backup) == expected, "Framingham original backup hash mismatch")
            original = strict_frame(pd.read_csv(backup, float_precision="round_trip"), split.names, label)
            require(len(original) == 700 and np.array_equal(real_numeric_matrix(frame.iloc[:700]),
                                                           real_numeric_matrix(original)),
                    "Framingham extension does not preserve original numeric prefix")
            evidence["extensions"].append({"class": 0, "pool_file": name, "pool_sha256": actual,
                "rows": len(frame), "original_rows": len(original), "additional_rows": len(frame)-len(original),
                "original_file": str(backup), "original_sha256": expected,
                "composition_evidence": "first 700 rows numerically identical to authenticated original backup",
                "generation_timestamp": None, "generation_call": None, "generation_rng_state": None,
                "historical_additional_model_binding": None,
                "provenance_status": "original block authenticated; additional generation event unknown",
                "old_manifest_coverage": "original 700 rows only; not rewritten"})
        evidence["pools"][str(label)] = {"path": str(path), "sha256": actual, "rows": len(frame),
                                         "historical_manifest_sha256": expected}
        pools.append(frame)
    return pools, evidence


def select_synthetic(pools, split, provenance):
    counts = tuple(int((split.y_train == k).sum()) for k in (0, 1))
    require(counts == QUOTAS[split.dataset], "Fixed real training counts differ from approved quotas")
    rng = np.random.default_rng(split.seed)
    frames, mapping = [], []
    for label, (pool, n) in enumerate(zip(pools, counts)):
        strict_frame(pool, split.names, label)
        require(len(pool) >= n, f"Synthetic class {label} shortfall: {len(pool)} < {n}; no padding")
        selected = rng.choice(len(pool), size=n, replace=False)
        for i in selected:
            mapping.append({"source_class": label, "source_row_index": int(i),
                            "source_file": provenance["pools"][str(label)]["path"],
                            "source_sha256": provenance["pools"][str(label)]["sha256"]})
        frames.append(pool.iloc[selected].copy())
    order = rng.permutation(sum(counts))
    combined = pd.concat(frames, ignore_index=True).iloc[order].reset_index(drop=True)
    selections = pd.DataFrame([{"selection_row": j, **mapping[i]} for j, i in enumerate(order)])
    require(not selections.duplicated(["source_file", "source_row_index"]).any(), "Repeated synthetic source row")
    return combined, selections


def real_groups(split):
    """Transitive union of original groups and exact PREPARED predictor equality."""
    rows, values = [], []
    for side in ("train", "test"):
        X, y = getattr(split, "X_" + side), getattr(split, "y_" + side)
        for i, record in enumerate(split.partition[side]):
            rows.append({**record, "side": side, "index": i, "clinical_class": int(y[i])})
            values.append(tuple(X[i].tolist()))
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen_origin, seen_value = {}, {}
    for i, (row, value) in enumerate(zip(rows, values)):
        for seen, key in ((seen_origin, row["group_id"]), (seen_value, value)):
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    output = {}
    for records in groups.values():
        key = value_hash(sorted(r["row_id"] for r in records))
        output[key] = [{**r, "attack_group": key} for r in records]
    return output


def choose_groups(items, target, rng):
    """Exact row quota with whole groups; local RNG, subset-sum predecessors."""
    require(target >= 0, "Negative grouped quota")
    if target == 0:
        return set()
    items = sorted(items)
    rng.shuffle(items)
    reachable, predecessors = 1, {}
    cap = (1 << (target + 1)) - 1
    for key, size in items:
        new = ((reachable << size) & cap) & ~reachable
        unseen = new
        while unseen:
            bit = unseen & -unseen
            total = bit.bit_length()-1
            predecessors[total] = (total-size, key)
            unseen -= bit
        reachable |= new
        if (reachable >> target) & 1:
            selected = set()
            while target:
                target, key = predecessors[target]
                selected.add(key)
            return selected
    raise ValueError("Whole-group quota unreachable; no row splitting or silent approximation")


def make_design(split):
    groups = real_groups(split)
    rng = random.Random(split.seed)
    labels, counts = {}, {}
    for key, rows in groups.items():
        clinical = {r["clinical_class"] for r in rows}
        require(len(clinical) == 1, "Mixed clinical labels in prepared/original group; cannot use classwise quota solver")
        labels[key] = next(iter(clinical))
        counts[key] = Counter(r["side"] for r in rows)
    G = set()
    for label in (0, 1):
        wanted = math.floor(.2 * int((split.y_train == label).sum()) + .5)
        G |= choose_groups([(k, c["train"]) for k, c in counts.items()
                            if labels[k] == label and c["train"] and not c["test"]], wanted, rng)
    blocks = [set(), set(), set()]
    quotas = []
    for label in (0, 1):
        H = {k for k, c in counts.items() if c["test"] and labels[k] == label}
        n = sum(counts[k]["test"] for k in H)
        q = [n//3 + int(i < n % 3) for i in range(3)]
        quotas.append(q)
        for i in range(3):
            selected = choose_groups([(k, counts[k]["test"]) for k in H], q[i], rng)
            blocks[i] |= selected
            H -= selected
        require(not H, "Unassigned holdout group")
    available = {k for k, c in counts.items() if c["train"] and not c["test"] and k not in G}
    for i in range(3):
        for label in (0, 1):
            forced = sum(counts[k]["train"] for k in blocks[i] if labels[k] == label)
            chosen = choose_groups([(k, counts[k]["train"]) for k in available if labels[k] == label],
                                   quotas[label][i]-forced, rng)
            blocks[i] |= chosen
            available -= chosen
        require(not blocks[i] & G, "Auxiliary/candidate group overlap")
    require(all(not blocks[i] & blocks[j] for i in range(3) for j in range(i)), "Attack block group overlap")
    g_rows = sorted((r for k in G for r in groups[k]), key=lambda r: r["index"])
    block_rows = [sorted((r for k in ks for r in groups[k]), key=lambda r: (r["side"], r["index"])) for ks in blocks]
    for rows in block_rows:
        require(Counter(r["clinical_class"] for r in rows if r["side"] == "train") ==
                Counter(r["clinical_class"] for r in rows if r["side"] == "test"), "Clinical membership imbalance")
    assignments = {r["row_id"]: r["attack_group"] for rs in groups.values() for r in rs}
    return {"seed": split.seed, "group_definition": "transitive original group OR exact prepared predictors; clinical target excluded",
            "G": g_rows, "blocks": block_rows, "row_attack_groups": assignments,
            "rotation": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            "G_selection": "20% by clinical class, half-up row quota; whole groups; exclude any group containing holdout",
            "candidate_selection": "balanced membership/class; whole groups; includes indistinguishable cross-membership groups"}


def observed_mask(split, side, fields):
    columns = [split.source.names.index(c) for c in fields]
    return np.asarray([all(split.source.X[r["source_position"]-1][j] is not None for j in columns)
                       for r in split.partition[side]], dtype=bool)


def input_integrity_record(split, run):
    directory = run.directory / split.dataset / f"seed_{split.seed}"
    return {"prepared_artifacts": {str(directory / name): expected for name, expected in split.artifact_hashes.items()},
            "source": {"path": str(REVISION_A1_SOURCE_FILES[split.dataset]),
                       "sha256": split.source.source_hash}}


def code_record():
    paths = [ROOT_DIR / "main_evaluator.py", ROOT_DIR / "src/revision_utility.py",
             ROOT_DIR / "src/revision_a1_runtime.py", ROOT_DIR / "src/revision_validity.py"]
    paths += sorted((ROOT_DIR / "src/evaluation").glob("*.py"))
    return {str(p.relative_to(ROOT_DIR)): digest(p) for p in paths}


class Results:
    """Append partial long-form results; one consolidated CSV, no ranking."""
    def __init__(self, root, context, directory):
        self.root, self.directory, self.context = Path(root), Path(directory), context
        self.rows = []

    def add(self, component, records, evidence):
        rows = []
        for record in records:
            extra = {k: v for k, v in record.items() if k not in LONG_COLUMNS}
            row = {**self.context, "component": component, "evidence": str(evidence),
                   **{k: v for k, v in record.items() if k in LONG_COLUMNS}}
            if extra:
                row["details"] = json.dumps(extra, allow_nan=False)
            rows.append(row)
        for path in (self.directory / "results_long.csv", self.root / "results_long.csv"):
            exists = path.exists()
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LONG_COLUMNS)
                if not exists:
                    writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
        self.rows.extend(rows)

    def phase(self, name, fn):
        directory = self.directory / name
        directory.mkdir(exist_ok=False)
        state = {"component": name, "status": "running", "start": now(), "warnings": [],
                 "context": self.context, "directory": str(directory)}
        atomic_json(directory / "execution.json", state)
        start = time.monotonic()
        records, caught = [], []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                records = fn(directory)
            state["status"] = ("failed" if any(r["status"] == "failed" for r in records) else
                               "completed_with_not_computable" if any(r["status"] == "not_computable" for r in records) else "completed")
        except Exception as exc:
            state.update(status="failed", error_type=type(exc).__name__, cause=str(exc), traceback=traceback.format_exc())
            records = [result(name + "_execution", None, status="failed", reason=f"{type(exc).__name__}: {exc}")]
        except BaseException as exc:
            state.update(status="interrupted", cause=type(exc).__name__)
            raise
        finally:
            import resource
            state.update(end=now(), elapsed_seconds=time.monotonic()-start,
                         warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                         process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            atomic_json(directory / "execution.json", state)
        self.add(name, records, directory / "execution.json")
        return state


def utility_condition(split, frame, condition, directory, *, save_estimator=False):
    """A single condition; TRTR called once in shared real output per split."""
    import pickle
    estimators = _get_classifiers(split.seed)
    outputs = []
    X, y = real_numeric_matrix(frame[split.names]), real_numeric_matrix(frame[["class"]])[:, 0].astype(int)
    require(tuple(int((y == k).sum()) for k in (0, 1)) == QUOTAS[split.dataset], "Utility training quotas mismatch")
    for name in ("logistic_regression", "random_forest", "xgboost"):
        d = directory / name
        d.mkdir()
        metadata = {"condition": condition, "classifier": name, "seed": split.seed,
                    "threshold": .5, "threshold_comparison": ">=", "positive_class": 1,
                    "training_hash": frame_record(frame), "test_hash": frame_record(split.frames()[1]),
                    "ap_definition": "sklearn.average_precision_score; not trapezoidal integration",
                    "parameter_search": False, "scaler_fit": "condition training only (LR pipeline)",
                    "start": now(), "status": "running"}
        atomic_json(d / "execution.json", metadata)
        start = time.monotonic()
        caught = []
        try:
            require(name in estimators, "Required classifier backend unavailable: " + name)
            estimator = estimators[name]
            metadata["effective_parameters"] = json_config(estimator)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(X.copy(), y.copy())
                positive = positive_probability(estimator, split.X_test.copy())
            pred = (positive >= .5).astype(int)
            metadata["class_order"] = estimator.classes_.tolist()
            metadata["positive_probability_column"] = int(np.flatnonzero(estimator.classes_ == 1)[0])
            if hasattr(estimator, "get_booster"):
                metadata["backend_resolved_configuration"] = json.loads(estimator.get_booster().save_config())
            predictions = pd.DataFrame([{**record, "y_true": int(split.y_test[i]),
                                         "probability_positive": float(positive[i]), "y_pred": int(pred[i])}
                                        for i, record in enumerate(split.partition["test"])])
            save_csv(predictions, d / "predictions.csv")
            metadata["predictions_sha256"] = digest(d / "predictions.csv")
            if save_estimator:
                with (d / "estimator.pkl").open("xb") as stream:
                    pickle.dump(estimator, stream)
                metadata["estimator_sha256"] = digest(d / "estimator.pkl")
            for row in binary_results(split.y_test, positive):
                outputs.append({**row, "condition": condition, "classifier": name,
                                "population": "real_test", "n_train": len(y), "n_test": len(split.y_test),
                                "prediction_evidence": str(d / "predictions.csv")})
            metadata["status"] = "completed"
        except Exception as exc:
            metadata.update(status="failed", cause=str(exc), traceback=traceback.format_exc())
            outputs.append(result("utility_execution", None, condition=condition, classifier=name,
                                  status="failed", reason=f"{type(exc).__name__}: {exc}"))
        finally:
            metadata.update(end=now(), elapsed_seconds=time.monotonic()-start,
                            warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in caught])
            atomic_json(d / "execution.json", metadata)
    return outputs


def dcr_results(split, S, geometry, design, directory):
    R, H = split.frames()
    records, distances = [], []
    for population, Q, Qlabel in (("synthetic", S, S["class"].to_numpy()),
                                  ("holdout", H, split.y_test)):
        for label in (None, 0, 1):
            mask_q = np.ones(len(Q), bool) if label is None else Qlabel == label
            mask_r = np.ones(len(R), bool) if label is None else split.y_train == label
            q_indices = np.flatnonzero(mask_q)
            d = geometry.nearest(Q.loc[mask_q, split.names], R.loc[mask_r, split.names])
            summary = distance_summary(d)
            for metric, value in summary.items():
                records.append(result("dcr_" + metric, value,
                    unit="rows" if metric in {"n_rows", "zero_count", "trimmed_each_tail"} else "mixed_distance",
                    population=population, **{"class": "all" if label is None else label},
                    n_rows=len(d), reference_rows=int(mask_r.sum())))
            for i, distance in zip(q_indices, d):
                distances.append({"population": population, "analysis_class": "all" if label is None else label,
                    "query_row": int(i), "clinical_class": int(Qlabel[i]), "distance": float(distance),
                    "row_id": split.partition["test"][i]["row_id"] if population == "holdout" else None,
                    "selection_row": int(i) if population == "synthetic" else None})
    save_csv(pd.DataFrame(distances), directory / "distances.csv")
    write_json(directory / "definition.json", {"reference": "same real R; same-class R for conditional analysis",
        "quantiles": "numpy linear", "trim": "floor(0.1*n) per extreme", "zeros": "retained",
        "ratios": "none", "geometry": geometry.record()})
    return records


def match_results(split, S, directory):
    R = split.frames()[0][split.names]
    synth = S[split.names]
    records, row_records = [], []
    for kind, r, s in (("exact_values", R, synth), ("rounded_6_decimals", R.round(6), synth.round(6))):
        real = Counter(map(tuple, real_numeric_matrix(r).tolist()))
        synthetic = Counter(map(tuple, real_numeric_matrix(s).tolist()))
        counts = np.asarray([real[tuple(row)] for row in real_numeric_matrix(s)])
        repetitions = np.asarray([synthetic[tuple(row)] for row in real_numeric_matrix(s)])
        n = int((counts > 0).sum())
        records += [result("matches_count", n, unit="rows", condition=kind, n_rows=len(s)),
                    result("matches_ratio", n / len(s), condition=kind, n_rows=len(s)),
                    result("synthetic_duplicate_extra_rows", len(s)-len(synthetic), unit="rows", condition=kind),
                    result("real_duplicate_extra_rows", len(r)-len(real), unit="rows", condition=kind)]
        row_records.extend({"comparison": kind, "selection_row": i, "clinical_class": int(S["class"].iloc[i]),
                            "train_matching_rows": int(nr), "synthetic_content_multiplicity": int(ns)}
                           for i, (nr, ns) in enumerate(zip(counts, repetitions)))
    save_csv(pd.DataFrame(row_records), directory / "multiplicities.csv")
    return records


def validity_results(split, S, geometry, spec, directory):
    X, synth = split.X_train, real_numeric_matrix(S[split.names])
    records, details = [], []
    row_outside, row_invalid = np.zeros(len(synth), bool), np.zeros(len(synth), bool)
    for j, name in enumerate(split.names):
        low, high = float(X[:, j].min()), float(X[:, j].max())
        values = synth[:, j]
        magnitude = np.maximum(np.maximum(low-values, values-high), 0.)
        outside = magnitude > 0
        domain = [0, 1] if spec["types"][name] == "binary" else spec["categorical_domains"].get(name)
        invalid = ~np.isin(values, domain) if domain is not None else np.zeros(len(synth), bool)
        if spec["types"][name] == "count":
            invalid |= values < 0
        row_outside |= outside
        row_invalid |= invalid
        for metric, value, unit in (("outside_train_range_count", int(outside.sum()), "rows"),
                                    ("outside_train_range_ratio", float(outside.mean()), "ratio"),
                                    ("invalid_documented_domain_count", int(invalid.sum()), "rows"),
                                    ("maximum_deviation_original_units", float(magnitude.max()), "original_variable_units")):
            records.append(result(metric, value, unit=unit, attribute=name, n_rows=len(synth)))
        for i in np.flatnonzero(outside | invalid):
            details.append({"selection_row": int(i), "variable": name, "value": float(values[i]),
                "outside_train_range": bool(outside[i]), "invalid_documented_domain": bool(invalid[i]),
                "train_min": low, "train_max": high, "deviation_original_units": float(magnitude[i]),
                "deviation_normalized_train_range": float(magnitude[i]/(high-low)) if high > low else None,
                "normalization_status": "ok" if high > low else "constant_train_range",
                "outside_G_range": bool(values[i] < geometry.low[j] or values[i] > geometry.high[j])})
    save_csv(pd.DataFrame(details, columns=["selection_row", "variable", "value", "outside_train_range",
        "invalid_documented_domain", "train_min", "train_max", "deviation_original_units",
        "deviation_normalized_train_range", "normalization_status", "outside_G_range"]), directory / "deviations.csv")
    write_json(directory / "policy.json", {"action": "diagnose only; no filtering, clipping or correction",
        "range": "full prepared real train; empirical, not clinical limits", "distance_clipping": "contribution only",
        "count_fraction_policy": "fractional preprocessed/generated values retained; no automatic integer restriction",
        "documented_domains": {name: ([0, 1] if spec["types"][name] == "binary" else spec["categorical_domains"].get(name))
                               for name in split.names}})
    return records + [result("rows_outside_train_range_ratio", float(row_outside.mean()), n_rows=len(S)),
                      result("rows_invalid_documented_domain_ratio", float(row_invalid.mean()), n_rows=len(S))]


def detectability_results(split, S, design, directory):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    require(len(S) == len(split.X_train), "Detectability expects matched real/synthetic sizes")
    X = np.vstack([split.X_train, real_numeric_matrix(S[split.names])])
    y = np.r_[np.ones(len(S), int), np.zeros(len(S), int)]
    lookup = {tuple(x.tolist()): design["row_attack_groups"][r["row_id"]]
              for x, r in zip(split.X_train, split.partition["train"])}
    group_ids = [lookup[tuple(x.tolist())] for x in split.X_train]
    group_ids += [lookup.get(tuple(x.tolist()), value_hash(x.tolist())) for x in real_numeric_matrix(S[split.names])]
    group_ids = np.asarray(group_ids)
    require(len(set(group_ids)) >= 5, "Insufficient detector groups for five folds")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split.seed)
    predictions, records, params = [], [], []
    for fold, (fit, test) in enumerate(cv.split(X, y, group_ids), 1):
        require(not set(group_ids[fit]) & set(group_ids[test]), "Detectability group leakage")
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[test])) != 2:
            records.append(result("detectability_auroc", None, fold=fold, status="not_computable", reason="one_origin_class_in_group_fold"))
            continue
        model = Pipeline([("scaler", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=1000, random_state=split.seed))])
        model.fit(X[fit], y[fit])
        scores = positive_probability(model, X[test])
        auc = float(roc_auc_score(y[test], scores))
        pmse = float(np.mean((scores-.5)**2))
        records += [result("detectability_auroc", auc, fold=fold, n_train=len(fit), n_test=len(test)),
                    result("detectability_pmse_raw", pmse, fold=fold),
                    result("detectability_pmse_normalized", pmse/.25, fold=fold)]
        params.append({"fold": fold, "parameters": json_config(model), "classes": model.classes_.tolist(),
                       "train_groups": sorted(set(group_ids[fit])), "evaluation_groups": sorted(set(group_ids[test]))})
        predictions.extend({"combined_row": int(i), "fold": fold, "origin_real": int(y[i]),
                            "probability_real": float(score), "group": group_ids[i],
                            "row_id": split.partition["train"][i]["row_id"] if i < len(S) else None,
                            "selection_row": int(i-len(S)) if i >= len(S) else None}
                           for i, score in zip(test, scores))
    save_csv(pd.DataFrame(predictions), directory / "predictions.csv")
    write_json(directory / "parameters.json", {"models": params, "interpretation": "real/synthetic distinguishability; not a privacy guarantee",
        "pmse_raw": "mean((p(real)-0.5)^2)", "pmse_normalized": "pmse_raw/0.25",
        "grouping": "transitive original real groups and exact prepared predictor equality across real/synthetic",
        "target_clinical_used": False})
    for metric in ("detectability_auroc", "detectability_pmse_raw", "detectability_pmse_normalized"):
        usable = [r["value"] for r in records if r["metric"] == metric and r["status"] == "ok"]
        records.append(result(metric, float(np.mean(usable)) if len(usable) == 5 else None,
                              fold="mean", status="ok" if len(usable) == 5 else "not_computable",
                              reason=None if len(usable) == 5 else "not_all_folds_computable"))
    return records


def candidate_frame(split, rows):
    frames = split.frames()
    return pd.DataFrame([frames[0 if r["side"] == "train" else 1].iloc[r["index"]].to_dict() for r in rows],
                        columns=split.names+["class"])


def mia_results(split, S, geometry, design, directory, *, bootstrap_repetitions):
    blocks = design["blocks"]
    features, labels = [], []
    for rows in blocks:
        frame = candidate_frame(split, rows)
        features.append(geometry.nearest(frame[split.names], S[split.names])[:, None])
        labels.append(np.asarray([int(r["side"] == "train") for r in rows]))
    records, all_predictions, configuration, aucs = [], [], [], []
    for fold, (fit, cal, test) in enumerate(design["rotation"], 1):
        sets = [{r["attack_group"] for r in blocks[i]} for i in (fit, cal, test)]
        require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "MIA group leakage")
        model = explicit_attack_forest(split.seed)
        model.fit(features[fit], labels[fit])
        calibration = positive_probability(model, features[cal])
        calibration_record = calibrate_membership_threshold(calibration[labels[cal] == 0])
        scores = positive_probability(model, features[test])
        pred = (scores > calibration_record["threshold"]).astype(int)
        require(len(np.unique(labels[test])) == 2, "MIA evaluation requires both membership classes")
        auc = float(roc_auc_score(labels[test], scores))
        aucs.append(auc)
        records.append(result("mia_auroc", auc, fold=fold, n_train=len(labels[fit]), n_test=len(labels[test])))
        tp = int(((pred == 1) & (labels[test] == 1)).sum())
        fp = int(((pred == 1) & (labels[test] == 0)).sum())
        fn = int(((pred == 0) & (labels[test] == 1)).sum())
        tn = int(((pred == 0) & (labels[test] == 0)).sum())
        counts = dict(tp=tp, fp=fp, fn=fn, tn=tn, n_members=tp+fn, n_nonmembers=fp+tn, fold=fold)
        for name, value, unit in (("mia_tpr_target_fpr_1pct_calibrated", tp/(tp+fn), "ratio"),
                                  ("mia_observed_fpr", fp/(fp+tn), "ratio"),
                                  ("mia_calibration_threshold", calibration_record["threshold"], "probability"),
                                  ("mia_calibration_fpr_resolution", calibration_record["fpr_resolution"], "ratio")):
            records.append(result(name, value, unit=unit, **counts,
                status="insufficient_resolution" if calibration_record["resolution_status"] == "insufficient_low_fpr_resolution" else "ok",
                reason=calibration_record["resolution_status"]))
        all_predictions.extend({**row, "fold": fold, "membership": int(y), "score": float(score),
            "minimum_synthetic_distance": float(distance), "threshold": calibration_record["threshold"],
            "predicted_member": int(p), "calibration_resolution_status": calibration_record["resolution_status"]}
            for row, y, score, distance, p in zip(blocks[test], labels[test], scores, features[test][:, 0], pred))
        configuration.append({"fold": fold, "fit_block": fit+1, "calibration_block": cal+1, "evaluation_block": test+1,
            "fit_groups": sorted(sets[0]), "calibration_groups": sorted(sets[1]), "evaluation_groups": sorted(sets[2]),
            "parameters": json_config(model), "classes": model.classes_.tolist(), "calibration": calibration_record})
    predictions = pd.DataFrame(all_predictions)
    save_csv(predictions, directory / "predictions.csv")
    for i, (rows, X, y) in enumerate(zip(blocks, features, labels), 1):
        save_csv(pd.DataFrame([{**r, "membership": int(v), "minimum_synthetic_distance": float(distance)}
                              for r, v, distance in zip(rows, y, X[:, 0])]), directory / f"block_{i}_features.csv")
    write_json(directory / "parameters.json", {"folds": configuration,
        "threat": "released S + candidate predictors + known auxiliary membership examples; no generator weights",
        "geometry_information": "schema and range summaries of G; G disjoint from all attack candidate groups",
        "score_feature": "single minimum mixed distance to S; clinical class excluded",
        "primary_aggregation": "unweighted arithmetic mean of 3 held-out block AUROCs",
        "secondary_name": "TPR with threshold calibrated to target FPR 1%; report observed FPR separately",
        "no_interpolation": True, "target_fpr": .01})
    records.append(result("mia_auroc", float(np.mean(aucs)), fold="mean"))
    y, p = predictions.membership.to_numpy(), predictions.predicted_member.to_numpy()
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    counts = dict(tp=tp, fp=fp, tn=tn, fn=fn, n_members=tp+fn, n_nonmembers=fp+tn, fold="aggregate")
    limited = any(c["calibration"]["resolution_status"] == "insufficient_low_fpr_resolution" for c in configuration)
    for metric, value in (("mia_tpr_target_fpr_1pct_calibrated", tp/(tp+fn)), ("mia_observed_fpr", fp/(fp+tn))):
        records.append(result(metric, value, **counts, status="insufficient_resolution" if limited else "ok",
                              reason="calibration blocks have fewer than 100 nonmembers" if limited else "finite-sample calibration; not a population FPR guarantee"))
    intervals = grouped_rate_intervals(predictions, split.seed, bootstrap_repetitions)
    intervals["calibration_resolution_insufficient"] = limited
    intervals["population_1pct_certification"] = False
    write_json(directory / "uncertainty.json", intervals)
    for rate, interval in intervals["rates"].items():
        for bound in ("low", "high"):
            records.append(result(f"mia_{rate}_conditional_ci95_{bound}", interval[bound], **counts,
                                  reason=intervals["caveat"]))
    return records


def attribute_results(split, S, design, directory):
    secret, known, regression = SECRETS[split.dataset]
    train, test = split.frames()
    Gidx = np.asarray([r["index"] for r in design["G"]], dtype=int)
    member_mask = np.ones(len(train), bool); member_mask[Gidx] = False
    observed_train = observed_mask(split, "train", [secret]+known)
    observed_test = observed_mask(split, "test", [secret]+known)
    Gmask = np.zeros(len(train), bool); Gmask[Gidx] = True
    auxiliary = train.loc[Gmask & observed_train]
    write_json(directory / "threat.json", {"secret": secret, "known_predictors": known,
        "clinical_target_excluded": True, "other_predictors_excluded": True,
        "reference_knowledge": "observed auxiliary fields and secrets of reserved G only; no evaluated candidate secrets",
        "publication_training": "G observed subset + unchanged shared S",
        "retinopathy": "partial anatomical descriptors without full image; ma1 detector count at confidence 0.5, not confirmed clinical count"
                        if split.dataset == "diabetic_retinopathy" else None,
        "synthetic_labels": "generator values, not observed clinical secrets; includes generation/imputation effects",
        "G_total": len(Gidx), "G_observed": len(auxiliary), "G_excluded_missing_original": int(len(Gidx)-len(auxiliary)),
        "G_originally_observed_indices": np.flatnonzero(Gmask & observed_train).tolist()})
    require(len(auxiliary) > 0, "No originally observed auxiliary training secrets/features")
    if not regression and not set(S[secret].unique()).issubset({0, 1}):
        write_json(directory / "invalid_sensitive_domain.json", {"attribute": secret,
            "selection_rows": np.flatnonzero(~S[secret].isin([0, 1])).tolist(),
            "action": "not computable; no recoding/filtering of S"})
        return [result(metric, None, attribute=secret, status="not_computable",
                       reason="synthetic sensitive labels outside documented binary domain")
                for metric in ("attribute_inference_auroc", "attribute_inference_average_precision")]
    sources = {"G": auxiliary, "G_plus_S": pd.concat([auxiliary, S], ignore_index=True)}
    fitted, parameters = {}, {}
    for condition, training in sources.items():
        model = explicit_attack_forest(split.seed, regression=regression)
        model.fit(real_numeric_matrix(training[known]), real_numeric_matrix(training[[secret]])[:, 0])
        fitted[condition] = model
        parameters[condition] = {"parameters": json_config(model), "training_rows": len(training),
            "training_predictors": known, "training_secret": secret,
            "training_hash": frame_record(training[known+[secret]])}
    write_json(directory / "parameters.json", parameters)
    predictions, records = [], []
    for population, frame, eligible, observed, partition in (
            ("member", train, member_mask, observed_train, split.partition["train"]),
            ("nonmember", test, np.ones(len(test), bool), observed_test, split.partition["test"])):
        mask = eligible & observed
        indices = np.flatnonzero(mask)
        n_eligible = int(eligible.sum())
        coverage = len(indices)/n_eligible if n_eligible else None
        records += [result("attribute_coverage", coverage, attribute=secret, population=population,
                           n_rows=len(indices), eligible_rows=n_eligible, excluded_missing_original=n_eligible-len(indices)),
                    result("attribute_excluded_missing_original", n_eligible-len(indices), unit="rows", attribute=secret, population=population)]
        if not len(indices):
            records.append(result("attribute_inference_mae" if regression else "attribute_inference_auroc", None,
                attribute=secret, population=population, status="not_computable", reason="no_observed_candidates"))
            continue
        X = real_numeric_matrix(frame.loc[mask, known])
        y = real_numeric_matrix(frame.loc[mask, [secret]])[:, 0]
        metrics = {}
        for condition, model in fitted.items():
            if regression:
                predicted = model.predict(X)
                scores = predicted
                metric_rows = [result("attribute_inference_mae", float(mean_absolute_error(y, predicted)), unit="secret_units")]
            else:
                require(set(np.unique(y)).issubset({0, 1}), "Invalid observed sensitive binary values")
                # Invalid synthetic sensitive labels are diagnosed, not recoded or dropped.
                require(set(np.asarray(model.classes_).tolist()).issubset({0, 1}),
                        "Synthetic sensitive labels outside binary domain; classification attack not computable without changing S")
                scores = (positive_probability(model, X) if len(model.classes_) == 2 else
                          np.full(len(X), float(model.classes_[0] == 1)))
                predicted = (scores >= .5).astype(int)
                metric_rows = [r for r in binary_results(y, scores, prefix="attribute_inference")
                               if r["metric"] in {"attribute_inference_auroc", "attribute_inference_average_precision"}]
            metrics[condition] = {r["metric"]: r for r in metric_rows}
            records.extend({**r, "condition": condition, "attribute": secret, "population": population,
                            "attacker": "random_forest_regressor" if regression else "random_forest_classifier",
                            "coverage": coverage, "n_rows": len(y), "n_train": len(sources[condition])} for r in metric_rows)
            predictions.extend({"condition": condition, "population": population, "attribute": secret,
                **partition[i], "attack_group": design["row_attack_groups"][partition[i]["row_id"]],
                "truth_originally_observed": True, "y_true": float(truth), "prediction": float(prediction),
                "probability_positive": None if regression else float(score)}
                for i, truth, prediction, score in zip(indices, y, predicted, scores))
        for metric, baseline in metrics["G"].items():
            published = metrics["G_plus_S"][metric]
            defined = baseline["value"] is not None and published["value"] is not None
            improvement = (baseline["value"]-published["value"] if regression else published["value"]-baseline["value"]) if defined else None
            records.append(result(metric + "_improvement", improvement, attribute=secret, population=population,
                condition="reference_minus_publication" if regression else "publication_minus_reference",
                unit="secret_units" if regression else "ratio", status="ok" if defined else "not_computable",
                reason=None if defined else "undefined_operand", n_rows=len(y), coverage=coverage))
    save_csv(pd.DataFrame(predictions), directory / "predictions.csv")
    return records


def neighborhood_results(split, S, geometry, design, directory):
    R = split.frames()[0]
    Gidx = np.asarray([r["index"] for r in design["G"]], int)
    Aidx = np.setdiff1d(np.arange(len(R)), Gidx)
    if not len(Gidx) or not len(Aidx):
        return [result("neighborhood_radius", None, status="not_computable", reason="empty_calibration_or_reference")]
    calibration = geometry.nearest(R.iloc[Gidx][split.names], R.iloc[Aidx][split.names])
    rank = int(math.ceil(.25 * len(calibration)))
    radius = float(np.sort(calibration)[rank-1])
    groups = [design["row_attack_groups"][split.partition["train"][i]["row_id"]] for i in Aidx]
    rows, distinct = geometry.neighborhood(S[split.names], R.iloc[Aidx][split.names], radius, groups)
    save_csv(pd.DataFrame({"selection_row": np.arange(len(S)), "clinical_class": S["class"],
                          "row_neighbors": rows, "distinct_group_neighbors": distinct}), directory / "neighbors.csv")
    save_csv(pd.DataFrame({"train_index": Gidx, "minimum_distance_to_A": calibration}), directory / "calibration_distances.csv")
    write_json(directory / "definition.json", {"name": "Neighborhood-Based k-Anonymity", "formal_guarantee": False,
        "reference": "A=R\\G, fixed for calibration and counting", "reference_indices": Aidx.tolist(),
        "calibration": "G->A; nearest-rank 25th percentile, ceil(0.25*n), no interpolation",
        "calibration_rank_one_based": rank, "radius": radius, "radius_zero_retained": radius == 0,
        "inclusion": "distance <= radius", "primary_count": "rows, including duplicate rows",
        "secondary_count": "distinct transitive original/prepared groups"})
    records = [result("neighborhood_radius", radius, unit="mixed_distance", n_train=len(Aidx))]
    for k in (3, 5):
        for name, counts in (("rows", rows), ("distinct_groups", distinct)):
            records += [result(f"neighborhood_k{k}_{name}_ratio", float((counts >= k).mean()), n_rows=len(S)),
                        result(f"neighborhood_k{k}_{name}_count", int((counts >= k).sum()), unit="rows", n_rows=len(S))]
    return records


def nnaa_results(split, S, geometry, directory):
    if len(S) != len(split.X_train) or len(S) < 2:
        return [result("nnaa_overall", None, status="not_computable", reason="unequal_sets_or_fewer_than_two_rows")]
    summary, individual = mixed_nnaa(geometry, split.frames()[0][split.names], S[split.names])
    save_csv(individual, directory / "individual.csv")
    write_json(directory / "definition.json", {"reference": "equal-sized R and S; no holdout variant",
        "self_neighbor": "excluded by row index, duplicates remain other neighbors", "tie_contribution": .5})
    return [result("nnaa_"+key, value, unit="rows" if key.endswith("ties") else "ratio") for key, value in summary.items()]


def check_protocol(config):
    p = config["evaluation_protocol"]
    require(p["name"] == "matched_train_privacy_v1" and p["run_id"] == PINNED_RUN, "Unknown evaluation protocol/run")
    require(tuple(p["models"]) == MODELS and tuple(p["seeds"]) == SEEDS and set(p["datasets"]) == set(QUOTAS),
            "Evaluation dataset/seed/generator contract differs")
    expected = {"synthetic_counts": "real_train_per_class", "utility_conditions": ["trtr_original", "tstr"],
                "selection": "without_replacement_source_rows; content_duplicates_retained",
                "G_fraction": .2, "mia_blocks": 3, "mia_target_fpr": .01, "utility_threshold": .5,
                "neighborhood_k": [3, 5], "neighborhood_quantile": .25,
                "distance": "Gower_equal_predictor_weights_ranges_G", "detectability_folds": 5}
    require(all(p.get(k) == v for k, v in expected.items()), "Evaluation scientific policy differs from approved contract")
    require(p["bootstrap_repetitions"] == 200, "Conditional uncertainty budget differs from frozen configuration")
    require(p["attribute_scenarios"] == {ds: {"secret": s, "known": k, "regression": r}
                                         for ds, (s, k, r) in SECRETS.items()}, "Sensitive-attribute contract differs")
    require(Path(p["input_root"]).resolve() == (ROOT_DIR / "results/revision_a1_simple").resolve(), "Unexpected synthetic input root")
    return p


def artifacts_unchanged(integrity, provenance=None):
    for path, expected in integrity["prepared_artifacts"].items():
        require(digest(Path(path)) == expected, "Prepared artifact changed during evaluation: " + path)
    require(digest(Path(integrity["source"]["path"])) == integrity["source"]["sha256"], "Source changed during evaluation")
    if provenance:
        for file in list(provenance["pools"].values()) + list(provenance["historical_files"].values()):
            require(digest(Path(file["path"])) == file["sha256"], "Generation evidence changed during evaluation")
        for extension in provenance["extensions"]:
            require(digest(Path(extension["original_file"])) == extension["original_sha256"], "Original Framingham backup changed")


def run_evaluator(args, config, logger):
    """Single sequential entry point. A fresh output directory is mandatory."""
    policy = check_protocol(config)
    run = RevisedRun(args.revision_a1_run, config_path=args.config)
    datasets = list(getattr(args, "datasets", None) or policy["datasets"])
    seeds = list(getattr(args, "seeds", None) or policy["seeds"])
    models = list(getattr(args, "models", None) or policy["models"])
    for requested, allowed in ((datasets, policy["datasets"]), (seeds, policy["seeds"]), (models, MODELS)):
        require(len(requested) == len(set(requested)) and set(requested).issubset(allowed), "Invalid/duplicate evaluation selection")
    pool_root = ROOT_DIR / policy["input_root"]
    if getattr(args, "dry_run", False):
        for dataset in datasets:
            for seed in seeds:
                split = run.load(dataset, seed)
                design = make_design(split)
                for model in models:
                    pools, provenance = read_pools(split, model, pool_root, run)
                    S, selections = select_synthetic(pools, split, provenance)
                    logger.info("VERIFIED DRY-RUN %s seed=%s %s: S=%s; G=%s; no fitting/writes", dataset, seed, model,
                                S["class"].value_counts().to_dict(), len(design["G"]))
        return 0
    root = Path(getattr(args, "evaluation_output", None) or ROOT_DIR / policy["output_root"]).resolve()
    require(not root.exists(), "Evaluation output already exists; no overwrite/retraining reuse: " + str(root))
    # Never write into generation, preparation, historical or pilot trees.
    for protected in (pool_root, ROOT_DIR / "results/revision_a1", ROOT_DIR / "data", ROOT_DIR / "models"):
        require(not root.is_relative_to(protected.resolve()), "Evaluation output overlaps protected input tree")
    root.mkdir(parents=True)
    configuration = Path(args.config).read_bytes()
    (root / "configuration.json").write_bytes(configuration)  # Preserve column_mapping insertion order.
    campaign = {"protocol": policy["name"], "run_id": PINNED_RUN, "start": now(), "status": "running",
        "configuration_path": str(Path(args.config).resolve()), "configuration_sha256": digest(Path(args.config)),
        "code_sha256": code_record(), "runtime": runtime_record(),
        "environment": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                         "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "selection": {"datasets": datasets, "seeds": seeds, "models": models},
        "historical_identities": "retained independently; evaluation never rewrites generation identities",
        "input_filenames": ["synthetic_class_0.csv", "synthetic_class_1.csv"]}
    campaign["identity"] = value_hash({k: v for k, v in campaign.items() if k not in {"start", "status"}})
    write_json(root / "campaign_identity.json", campaign)
    inventory = [{"dataset": ds, "seed": seed, "generador": model,
                  "status": "pending" if ds in datasets and seed in seeds and model in models else "not_requested"}
                 for ds in policy["datasets"] for seed in policy["seeds"] for model in MODELS]
    require(len(inventory) == 160, "Expected 160-combination inventory")
    def persist_inventory():
        atomic_json(root / "inventory.json", inventory)
        # CSV is a current-state view; experimental inputs and per-component evidence remain immutable.
        pd.DataFrame(inventory).to_csv(root / "inventory.csv", index=False)
    persist_inventory()
    current = None
    try:
        for dataset in datasets:
            for seed in seeds:
                selected_entries = [e for e in inventory if e["dataset"] == dataset and e["seed"] == seed and e["generador"] in models]
                split_dir = root / dataset / f"seed_{seed}"
                shared = split_dir / "shared"
                shared.mkdir(parents=True)
                try:
                    split = run.load(dataset, seed)
                    require(tuple(np.bincount(split.y_train, minlength=2)) == QUOTAS[dataset], "Prepared quota mismatch")
                    integrity = input_integrity_record(split, run)
                    design = make_design(split)
                    Gidx = [r["index"] for r in design["G"]]
                    geometry = MixedGeometry(split.frames()[0].iloc[Gidx][split.names], split.names, run.policy["datasets"][dataset]["types"])
                    write_json(shared / "attack_design.json", design)
                    write_json(shared / "geometry.json", geometry.record())
                    write_json(shared / "inputs.json", integrity)
                    shared_identity = {"campaign": campaign["identity"], "dataset": dataset, "seed": seed,
                        "input_hashes": split.artifact_hashes, "attack_design_sha256": digest(shared / "attack_design.json"),
                        "geometry_sha256": digest(shared / "geometry.json")}
                    shared_identity["identity"] = value_hash(shared_identity)
                    write_json(shared / "identity.json", shared_identity)
                    baseline = Results(root, {"dataset": dataset, "seed": seed, "generador": "real",
                                              "evaluation_identity": shared_identity["identity"]}, shared)
                    baseline.phase("trtr_original", lambda d: utility_condition(split, split.frames()[0], "trtr_original", d,
                                                                               save_estimator=getattr(args, "save_classifiers", False)))
                    write_json(shared / "baseline_reference.json", {"identity": shared_identity["identity"],
                        "results_path": str(shared / "results_long.csv"), "results_sha256": digest(shared / "results_long.csv"),
                        "shared_by_generators": models, "count_as": "one TRTR per dataset/seed/classifier, not per generator"})
                except Exception as exc:
                    atomic_json(shared / "preparation_failure.json", {"cause": str(exc), "traceback": traceback.format_exc()})
                    for entry in selected_entries:
                        entry.update(status="failed", cause="shared preparation: " + str(exc), end=now())
                        Results(root, {"dataset": dataset, "seed": seed, "generador": entry["generador"]}, shared).add(
                            "shared_preparation", [result("execution", None, status="failed", reason=str(exc))], shared / "preparation_failure.json")
                    persist_inventory()
                    continue
                for entry in selected_entries:
                    current = entry
                    model = entry["generador"]
                    directory = split_dir / model
                    directory.mkdir()
                    entry.update(status="running", start=now(), artifact_dir=str(directory))
                    persist_inventory()
                    start = time.monotonic()
                    logger.info("EVALUATION %s seed=%s generator=%s", dataset, seed, model)
                    sink = Results(root, {"dataset": dataset, "seed": seed, "generador": model}, directory)
                    try:
                        pools, provenance = read_pools(split, model, pool_root, run)
                        S, selected = select_synthetic(pools, split, provenance)
                        write_json(directory / "generation_provenance.json", provenance)
                        save_csv(selected, directory / "selection.csv")
                        save_csv(S, directory / "selected_synthetic.csv")
                        identity = {"shared_evaluation_identity": shared_identity["identity"], "historical_generation_identity": provenance["historical_identity_digest"],
                            "generation_provenance_sha256": digest(directory / "generation_provenance.json"),
                            "pool_hashes": provenance["pools"], "selection_sha256": digest(directory / "selection.csv"),
                            "selected_csv_sha256": digest(directory / "selected_synthetic.csv"), "selected_values": frame_record(S),
                            "baseline_reference_sha256": digest(shared / "baseline_reference.json")}
                        identity["identity"] = value_hash(identity)
                        write_json(directory / "evaluation_identity.json", identity)
                        entry["evaluation_identity"] = sink.context["evaluation_identity"] = identity["identity"]
                        write_json(directory / "baseline_reference.json", {"path": str(shared / "baseline_reference.json"),
                                   "sha256": digest(shared / "baseline_reference.json")})
                        spec = run.policy["datasets"][dataset]
                        phases = [
                            ("utility_tstr", lambda d: utility_condition(split, S, "tstr", d, save_estimator=getattr(args, "save_classifiers", False))),
                            ("validity", lambda d: validity_results(split, S, geometry, spec, d)),
                            ("matches", lambda d: match_results(split, S, d)),
                            ("dcr", lambda d: dcr_results(split, S, geometry, design, d)),
                            ("nnaa", lambda d: nnaa_results(split, S, geometry, d)),
                            ("detectability", lambda d: detectability_results(split, S, design, d)),
                            ("mia", lambda d: mia_results(split, S, geometry, design, d, bootstrap_repetitions=policy["bootstrap_repetitions"])),
                            ("attribute_inference", lambda d: attribute_results(split, S, design, d)),
                            ("neighborhood", lambda d: neighborhood_results(split, S, geometry, design, d))]
                        for name, fn in phases:
                            state = sink.phase(name, fn)
                            logger.info("  %s: %s (%.2fs)", name, state["status"], state["elapsed_seconds"])
                        artifacts_unchanged(integrity, provenance)
                        require(code_record() == campaign["code_sha256"] and digest(Path(args.config)) == campaign["configuration_sha256"],
                                "Evaluation code/config changed during execution")
                        statuses = {r["status"] for r in sink.rows + baseline.rows}
                        entry["status"] = "failed" if "failed" in statuses else "completed_with_not_computable" if "not_computable" in statuses else "completed"
                        entry["component_statuses"] = {name: read_json(directory / name / "execution.json")["status"] for name, _ in phases}
                        entry["rows_selected_per_class"] = {str(k): int((S["class"] == k).sum()) for k in (0, 1)}
                        entry["results_sha256"] = digest(directory / "results_long.csv")
                    except Exception as exc:
                        entry.update(status="failed", cause=f"{type(exc).__name__}: {exc}")
                        known_inputs = []
                        for label in (0, 1):
                            path = pool_root / dataset / f"seed_{seed}" / model / f"synthetic_class_{label}.csv"
                            if path.is_file():
                                known_inputs.append({"path": str(path), "sha256": digest(path)})
                        atomic_json(directory / "failure.json", {"cause": entry["cause"], "traceback": traceback.format_exc(),
                                                                "available_input_files": known_inputs})
                        sink.add("combination", [result("execution", None, status="failed", reason=entry["cause"])], directory / "failure.json")
                        logger.error("Failed %s seed=%s %s: %s", dataset, seed, model, exc)
                    finally:
                        entry.update(end=now(), elapsed_seconds=time.monotonic()-start)
                        persist_inventory()
    except BaseException as exc:
        if current is not None and current["status"] == "running":
            current.update(status="interrupted", cause=type(exc).__name__, end=now())
        persist_inventory()
        atomic_json(root / "summary.json", {"status": "interrupted", "cause": str(exc), "end": now()})
        raise
    selected_states = [e["status"] for e in inventory if e["status"] != "not_requested"]
    failed = any(s == "failed" for s in selected_states)
    summary = {"status": "finished_with_failures" if failed else "finished_with_not_computable" if "completed_with_not_computable" in selected_states else "completed",
               "end": now(), "counts": dict(Counter(e["status"] for e in inventory)), "expected_full_campaign_combinations": 160,
               "requested_combinations": len(selected_states), "completed_full_campaign": len(selected_states) == 160 and not failed,
               "consolidated_csv": str(root / "results_long.csv"), "consolidated_sha256": digest(root / "results_long.csv")}
    atomic_json(root / "summary.json", summary)
    logger.info("Evaluation finished: %s", summary)
    return 1 if failed else 0
