"""Train-only fixed-size real baselines and auditable utility outputs for A1."""
from __future__ import annotations

import traceback
import warnings
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.revision_a1 import write_json
from src.revision_a1_runtime import array_record, require, value_hash, real_numeric_matrix

CONDITIONS = ("trtr_original", "trtr_balanced_fixed", "tstr")


def json_config(value):
    """Stable estimator configuration without pickling or fitting anything."""
    if isinstance(value, np.generic):
        return json_config(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"numeric_special": "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
    if hasattr(value, "get_params"):
        return {"class": f"{type(value).__module__}.{type(value).__name__}",
                "params": json_config(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(k): json_config(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_config(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value)}")


def classifier_specs(seed):
    from src.evaluation.utility import _get_classifiers
    return {name: json_config(estimator) for name, estimator in _get_classifiers(seed).items()}


@dataclass
class TrainingCondition:
    condition: str
    X: np.ndarray
    y: np.ndarray
    selections: dict


def real_baselines(X_train, y_train, train_rows, seed, per_class=700):
    """No test argument: all original rows, or balanced fixed-size resampling.

    For n<quota retain every row once then add random draws with replacement;
    for n>quota choose without replacement. This is not pure ROS.
    """
    require(isinstance(per_class, int) and not isinstance(per_class, bool) and per_class > 0, "Invalid quota")
    X, y = np.asarray(X_train), np.asarray(y_train)
    require(X.ndim == 2 and y.shape == (len(X),) and len(train_rows) == len(X), "Train alignment mismatch")
    require(np.isfinite(X).all() and set(np.unique(y)) == {0, 1}, "Invalid prepared train")
    require(len({r["row_id"] for r in train_rows}) == len(X), "Duplicate source row IDs")
    rng = np.random.default_rng(seed)
    balanced = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) < per_class:
            chosen = np.concatenate((indices, rng.choice(indices, per_class - len(indices), replace=True)))
        elif len(indices) > per_class:
            chosen = rng.choice(indices, per_class, replace=False)
        else:
            chosen = indices.copy()
        balanced.extend(chosen.tolist())
    balanced = rng.permutation(np.asarray(balanced, dtype=np.int64))
    outputs = []
    for condition, chosen in ((CONDITIONS[0], np.arange(len(X))), (CONDITIONS[1], balanced)):
        multiplicities = np.bincount(chosen, minlength=len(X))
        by_class = {}
        for label in (0, 1):
            m = multiplicities[y == label]
            by_class[str(label)] = {"available": len(m), "delivered": int(m.sum()),
                "distinct_source_rows": int((m > 0).sum()), "extra_copies": int(np.maximum(m-1, 0).sum()),
                "unselected_source_rows": int((m == 0).sum()), "max_multiplicity": int(m.max()),
                "multiplicity_histogram": {str(k): v for k, v in sorted(Counter(m.tolist()).items())}}
        evidence = {"condition": condition, "policy": "all_prepared_train_rows" if condition == CONDITIONS[0]
                    else "balanced_fixed_size_real_resampling_v1; not pure ROS",
                    "seed": seed, "rng": "numpy.default_rng/PCG64", "per_class": per_class,
                    "input_hashes": {"X_train": array_record(X), "y_train": array_record(y)},
                    "by_class": by_class,
                    "selection": [{"output_row": j, "train_matrix_row": int(i), **train_rows[i]}
                                  for j, i in enumerate(chosen)],
                    "multiplicities": [{"train_matrix_row": i, "row_id": row["row_id"],
                                        "group_id": row["group_id"], "count": int(multiplicities[i])}
                                       for i, row in enumerate(train_rows)]}
        # matrix_row belongs to the source partition; output_row identifies the resample.
        outputs.append(TrainingCondition(condition, X[chosen].copy(), y[chosen].copy(), evidence))
    return outputs


def tstr_condition(frame, names, per_class):
    require(list(frame.columns) == list(names) + ["class"], "TSTR column order mismatch")
    matrix = real_numeric_matrix(frame[names])
    target = real_numeric_matrix(frame[["class"]]).reshape(-1)
    require(frame["class"].value_counts().to_dict() == {0: per_class, 1: per_class},
            "TSTR effective class counts must equal the balanced real baseline quota; use IR=1")
    X = matrix
    y = frame["class"].to_numpy(copy=True)
    if y.dtype.kind == "O":
        y = target.astype(np.int64)
    return TrainingCondition("tstr", X, y, {"policy": "synthetic_rows_as_delivered", "per_class": per_class,
                                            "numeric_representation": {"source_dtypes": [str(t) for t in frame[names].dtypes],
                                                "matrix_dtype": str(X.dtype), "values_preserved": True,
                                                "nullable_numeric_conversion": frame[names].to_numpy().dtype.kind == "O"},
                                            "counts": {"0": per_class, "1": per_class}})


def prediction_schema():
    return {"schema_version": 1, "threshold": 0.5, "threshold_operator": ">=", "positive_class": 1,
            "prediction_columns": ["test_matrix_row", "source_position", "row_id", "group_id",
                                   "y_true", "y_pred", "probability_positive"],
            "metrics": {"roc_auc": "sklearn.metrics.roc_auc_score",
                        "average_precision": "sklearn.metrics.average_precision_score; not trapezoidal PR AUC",
                        "f1": "sklearn.metrics.f1_score at probability_positive >= 0.5"},
            "states": ["planned", "running", "completed", "failed"],
            "execution_kinds": ["benchmark", "simulation", "pilot"]}


def evaluate_condition(training, X_test, y_test, test_rows, estimator, output, context,
                       *, execution_kind, allow_training=False, save_estimator=False):
    """The only fit call is guarded. Simulations must supply an explicitly marked double."""
    require(execution_kind in {"benchmark", "simulation", "pilot"}, "Unknown execution kind")
    require(training.condition in CONDITIONS, "Unknown condition")
    if execution_kind == "simulation":
        require(getattr(estimator, "is_simulated", False) is True, "Simulation requires a marked test double")
    else:
        require(allow_training, "Classifier training is disabled unless explicitly enabled by the caller")
        require(not getattr(estimator, "is_simulated", False), "A simulated estimator cannot produce benchmark results")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": prediction_schema(), "context": context, "condition": training.condition,
                "execution_kind": execution_kind, "status": "running", "errors": [], "warnings": [],
                "configuration": json_config(estimator), "selection_sha256": value_hash(training.selections),
                "inputs": {"X_train": array_record(training.X), "y_train": array_record(training.y),
                           "X_test": array_record(X_test), "y_test": array_record(y_test),
                           "test_identifiers_sha256": value_hash(test_rows)},
                "classifier_classes": None, "positive_probability_column": None,
                "metrics": None, "prediction_file_sha256": None, "estimator_file_sha256": None}
    metadata["configuration_sha256"] = value_hash(metadata["configuration"])
    caught = []
    try:
        require(len(test_rows) == len(X_test) == len(y_test) and
                all(r["matrix_row"] == j for j, r in enumerate(test_rows)), "Test alignment mismatch")
        require(training.X.shape[1] == np.asarray(X_test).shape[1] and np.isfinite(X_test).all(), "Test schema mismatch")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(training.X.copy(), training.y.copy())
            classes = np.asarray(estimator.classes_)
            require(classes.shape == (2,) and set(classes.tolist()) == {0, 1}, "Classifier classes must be 0 and 1")
            positive_column = int(np.flatnonzero(classes == 1)[0])
            metadata["classifier_classes"] = classes.tolist()
            metadata["positive_probability_column"] = positive_column
            probabilities = np.asarray(estimator.predict_proba(np.asarray(X_test).copy()))
            require(probabilities.shape == (len(X_test), 2) and np.isfinite(probabilities).all() and
                    ((probabilities >= 0) & (probabilities <= 1)).all() and
                    np.allclose(probabilities.sum(axis=1), 1), "Invalid probabilities")
            positive = probabilities[:, positive_column]
            predicted = (positive >= .5).astype(np.int32)
            metadata["metrics"] = {"roc_auc": float(roc_auc_score(y_test, positive)),
                                   "average_precision": float(average_precision_score(y_test, positive)),
                                   "f1": float(f1_score(y_test, predicted, zero_division=0))}
        predictions = pd.DataFrame([{ "test_matrix_row": j, "source_position": row["source_position"],
                       "row_id": row["row_id"], "group_id": row["group_id"], "y_true": int(y_test[j]),
                       "y_pred": int(predicted[j]), "probability_positive": float(positive[j])}
                      for j, row in enumerate(test_rows)], columns=prediction_schema()["prediction_columns"])
        predictions.to_csv(output / "predictions.csv", index=False, mode="x")
        from src.revision_a1 import digest
        metadata["prediction_file_sha256"] = digest(output / "predictions.csv")
        if save_estimator:
            with (output / "estimator.pkl").open("xb") as stream:
                pickle.dump(estimator, stream)
            metadata["estimator_file_sha256"] = digest(output / "estimator.pkl")
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        metadata["warnings"] = [{"category": w.category.__name__, "message": str(w.message),
                                  "filename": w.filename, "lineno": w.lineno} for w in caught]
        write_json(output / "execution.json", metadata)
    return metadata


def evaluate_utility_conditions(split, synth, per_class, output, context, *,
                                execution_kind="benchmark", allow_training=False, estimator_factory=None,
                                save_estimators=False):
    from src.evaluation.utility import _get_classifiers
    synthetic = tstr_condition(synth, split.names, per_class)  # Check quotas before any fit.
    baselines = real_baselines(split.X_train, split.y_train, split.partition["train"], split.seed, per_class)
    specs = classifier_specs(split.seed)
    records = []
    for training in [*baselines, synthetic]:
        generator = context["model"] if training.condition == "tstr" else "real"
        training_dir = Path(output) / training.condition / generator
        training_dir.mkdir(parents=True, exist_ok=False)
        write_json(training_dir / "training_selection.json", training.selections)
        estimators = (estimator_factory or _get_classifiers)(split.seed)
        require(set(estimators) == set(specs), "Classifier set differs between conditions")
        for name, estimator in estimators.items():
            if execution_kind in {"benchmark", "pilot"}:
                require(json_config(estimator) == specs[name], "Classifier configuration differs from the common baseline specification")
            record = evaluate_condition(training, split.X_test, split.y_test, split.partition["test"], estimator,
                                        training_dir / name,
                                        {**context, "generator": generator, "classifier": name,
                                         "training_selection_reference": "../training_selection.json",
                                         "benchmark_classifier_configuration": specs[name]},
                                        execution_kind=execution_kind, allow_training=allow_training, save_estimator=save_estimators)
            records.append(record)
    return records
