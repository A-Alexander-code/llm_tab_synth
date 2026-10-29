"""No-training evidence: real NumPy/geometry, simulated generators/classifiers."""
from __future__ import annotations

import logging
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from config.const import ROOT_DIR, SUPPORTED_MODELS, REVISION_A1_SOURCE_FILES
from src.revision_a1 import canonical, digest, encode, prepare, verify_saved
from src.revision_a1_runtime import (RevisedRun, array_record, evaluate_split, frame_record,
                                     generator_inputs, require, sample_split, value_hash)


def overlap_record(train, test, train_y, test_y):
    common = set(train) & set(test)
    train_rows = [i for i, row in enumerate(train) if row in common]
    test_rows = [i for i, row in enumerate(test) if row in common]
    return {"shared_patterns": len(common), "train_rows": len(train_rows), "test_rows": len(test_rows),
            "train_classes": dict(Counter(str(train_y[i]) for i in train_rows)),
            "test_classes": dict(Counter(str(test_y[i]) for i in test_rows)),
            "patterns": [{"pattern_sha256": value_hash(row),
                          "train_classes": dict(Counter(str(train_y[i]) for i, r in enumerate(train) if r == row)),
                          "test_classes": dict(Counter(str(test_y[i]) for i, r in enumerate(test) if r == row))}
                         for row in sorted(common, key=repr)]}


def overlap_stages(split):
    source, prep = split.source, split.preprocessing
    indices = [source.names.index(c) for c in prep.kept]
    stages = {}
    for stage in ("original_normalized", "after_column_selection", "after_imputation"):
        sides = []
        for side in ("train", "test"):
            rows = [source.X[r["source_position"] - 1] for r in split.partition[side]]
            if stage == "after_imputation":
                patterns = [tuple(row) for row in getattr(split, f"X_{side}").tolist()]
            else:
                keep = range(len(source.names)) if stage == "original_normalized" else indices
                patterns = [tuple(canonical(row[i]) for i in keep) for row in rows]
            sides.append(patterns)
        stages[stage] = overlap_record(*sides, split.y_train, split.y_test)
    return stages


def geometry(split):
    from src.evaluation.privacy import compute_dcr_holdout, compute_kanonymity
    train = pd.DataFrame(split.X_train, columns=split.names)
    test = pd.DataFrame(split.X_test, columns=split.names)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train.fillna(0))
    tree = BallTree(scaled_train, metric="euclidean")
    distances = tree.query(scaler.transform(test.fillna(0)), k=1)[0][:, 0]
    percentile = float(np.percentile(distances, 25))
    # Test rows serve solely as geometric query probes; they are not synthetic data.
    rows = compute_dcr_holdout(test, train)
    kanon = compute_kanonymity(test, train, test)
    actual = {r["metric"]: r["value"] for r in rows + kanon}
    require(actual["dcr_holdout_mean"] == float(distances.mean()), "DCR implementation mismatch")
    require(actual["kanon_epsilon"] == (percentile if percentile > 0 else 1e-6), "Radius mismatch")
    return {"distance_sha256": array_record(distances), "zero_distance_test_rows": int((distances == 0).sum()),
            "zero_distance_test_classes": dict(Counter(str(y) for y in split.y_test[distances == 0])),
            "epsilon_p25_before_fallback": percentile, "fallback_used": percentile <= 0,
            "epsilon_p25_without_zero_rows": float(np.percentile(distances[distances > 0], 25)),
            "holdout_mean_without_zero_rows": float(distances[distances > 0].mean()),
            "actual_metrics_on_real_query_probes": actual}


def zero_geometry_probe():
    from src.evaluation.privacy import compute_dcr_holdout, compute_kanonymity
    train = pd.DataFrame({"feature": [0., 0., 0., 0., 0.]})
    test = pd.DataFrame({"feature": [0., 0.]})
    rows = compute_dcr_holdout(test, train) + compute_kanonymity(test, train, test)
    metrics = {r["metric"]: r["value"] for r in rows}
    require(metrics["kanon_epsilon"] == 1e-6 and metrics["dcr_holdout_mean"] == 0,
            "Zero fallback behavior changed")
    from src.plots.privacy import build_privacy_table
    probes = []
    for numerator in (0., 1.):
        records = [{"dataset": "fixture", "model": "fixture", "output_ir": 1.,
                    "block": "privacy_dcr", "metric": metric, "mean": value}
                   for metric, value in (("dcr_mean", numerator), ("dcr_holdout_mean", 0.))]
        ratio = build_privacy_table(pd.DataFrame(records), ("fixture",), ("fixture",))["ratio_dcr_mean"].iloc[0]
        require(pd.isna(ratio), "Zero DCR denominator handling changed")
        probes.append({"numerator": numerator, "denominator": 0., "ratio": "NaN"})
    return {"metrics": metrics, "actual_plot_ratio_probes": probes}


class FixtureGenerator:
    """A fixture replayer, never a learned model or a synthetic-data result."""
    def __init__(self, expected, records):
        self.expected = expected
        self.records = records

    def fit(self, frame, model_dir):
        pd.testing.assert_frame_equal(frame, self.expected, check_exact=True)
        self.frame = frame.copy()
        self.records.append({"fit": frame_record(frame)})
        model_dir.mkdir(parents=True, exist_ok=True)

    def sample(self, n, model_dir):
        if "__target__" in self.frame:
            parts = []
            for label in (0, 1):
                rows = self.frame.loc[self.frame["__target__"] == label].drop(columns="__target__")
                frame = rows.iloc[np.arange(n) % len(rows)].reset_index(drop=True)
                frame["__label__"] = np.full(n, label, dtype="<i4")
                parts.append(frame)
            return pd.concat(parts, ignore_index=True)
        return self.frame.iloc[np.arange(n) % len(self.frame)].reset_index(drop=True)


def simulated_flow(run, split, model, base):
    import main_evaluator as evaluator
    from src.evaluation import utility
    expected = generator_inputs(split, model)
    fit_records = []
    classifier_records = []
    factories = iter(frame for _, frame in expected)

    def factory(name, params):
        from src.revision_generator_parameters import parameter_plan
        effective = parameter_plan(model, run.config["model_params"][model], split.seed)["effective_wrapper_parameters"]
        if model in ('ctgan', 'copulagan', 'tvae', 'gaussian_copula'):
            # The approved contract adds shared semantic types after generic parameter
            # migration. Reconstruct that addition independently; keep full equality.
            from src.revision_sdv import schema_record
            effective['_revision_parameters']['sdv_semantic_schema'] = schema_record(
                split.names, run.policy['datasets'][split.dataset]['types'])
        require(name == model and params == effective, "Generator parameters mismatch")
        return FixtureGenerator(next(factories), fit_records)

    def classifier(clf, X_train, y_train, X_test, y_test, clf_name, mode, save_path=None):
        require(np.array_equal(X_test, split.X_test) and np.array_equal(y_test, split.y_test), "Classifier test changed")
        if mode == "trtr":
            require(np.array_equal(X_train, split.X_train) and np.array_equal(y_train, split.y_train), "TRTR train changed")
        else:
            require(set(np.unique(y_train)) == {0, 1} and X_train.shape[1] == len(split.names), "TSTR schema changed")
            require(np.array_equal(X_train, expected_final["frame"][split.names].to_numpy()) and
                    np.array_equal(y_train, expected_final["frame"]["class"].to_numpy()), "TSTR data changed")
        classifier_records.append({"mode": mode, "X_train": array_record(X_train), "y_train": array_record(y_train),
                                   "X_test": array_record(X_test), "y_test": array_record(y_test)})
        return []

    stubbed = ("compute_detectability", "compute_univariate", "compute_multivariate", "compute_exact_matches",
               "compute_dcr", "compute_dcr_holdout", "compute_mia", "compute_attribute_inference",
               "compute_nnaa", "compute_kanonymity")
    from contextlib import ExitStack
    expected_final = {}
    with ExitStack() as stack:
        stack.enter_context(patch("src.revision_a1_runtime.MODELS_DIR", base / "models"))
        stack.enter_context(patch("src.revision_a1_runtime.RESULTS_DIR", base / "results"))
        for name in stubbed:
            def recorder(*args, _name=name, **kwargs):
                for frame in args:
                    if isinstance(frame, pd.DataFrame):
                        columns = list(frame.columns)
                        target_expected = _name == "compute_attribute_inference"
                        require(columns == split.names + (["class"] if target_expected else []), "Privacy/metric input schema changed")
                return []
            recorder.__name__ = name
            stack.enter_context(patch.object(evaluator, name, recorder))
        stack.enter_context(patch.object(utility, "_get_classifiers", lambda seed: {"simulated": object()}))
        stack.enter_context(patch.object(utility, "_evaluate_classifier", classifier))
        sample_split(run, split, model, factory)
        output = run.paths(split, model)[2]
        from src.revision_a1_runtime import load_samples
        from src.evaluation.builder import build_synthetic_dataset
        samples = load_samples(run, split, model)
        majority = int(np.argmax(np.bincount(split.y_train)))
        expected_final["frame"] = build_synthetic_dataset(samples[majority], samples[1-majority], 1.,
                                                          split.seed, majority, 1-majority)
        evaluate_split(run, split, model, [1.0], evaluator._evaluate_combination, logging.getLogger("a1-audit"))
        evidence = json.loads((output / "metrics/input_hashes.json").read_text())
        sample_manifest = json.loads((output / "samples_manifest.json").read_text())
    require(len(classifier_records) == 2, "TRTR/TSTR dispatch missing")
    return {"model": model, "kind": "simulated generator + classifier + privacy/statistical metrics",
            "generator_fit": fit_records, "classifier_arrays": classifier_records,
            "sample_csv_hashes": sample_manifest["files"], "metric_dispatch": evidence["calls"]}


def audit(run, *, simulate=True):
    code_paths = ("src/revision_a1_runtime.py", "src/revision_a1_audit.py", "main_check_revision_a1.py",
                  "main_sampler.py", "main_evaluator.py", "src/plots/privacy.py")
    before_code = {p: digest(ROOT_DIR / p) for p in code_paths}
    reports, diversity = [], {}
    with tempfile.TemporaryDirectory(prefix="a1-simulated-") as temporary:
        for dataset in run.policy["datasets"]:
            tests = []
            for seed in run.policy["seeds"]:
                split = run.load(dataset, seed)
                # Audit-only reproduction; production consumers never split or fit preprocessing.
                expected, _ = prepare(split.source, run.policy["datasets"][dataset], seed, run.policy["test_fraction"])
                expected["provenance.json"] = encode(run.manifest["identity"])
                verify_saved(run.directory / dataset / f"seed_{seed}", expected, split.source)
                test_ids = sorted(r["row_id"] for r in split.partition["test"])
                tests.append((seed, value_hash(test_ids), set(test_ids)))
                item = {"dataset": dataset, "seed": seed, "verified": True, "audit_only_preparation_reproduced": True,
                        "arrays": {name: array_record(getattr(split, name)) for name in ("X_train", "X_test", "y_train", "y_test")},
                        "columns": split.names, "artifact_hashes": split.artifact_hashes,
                        "partition_sha256": value_hash(split.partition), "test_set_sha256": value_hash(test_ids),
                        "overlap_stages": overlap_stages(split), "geometry": geometry(split)}
                if simulate:
                    item["simulated_flow"] = [simulated_flow(run, split, model, Path(temporary))
                                              for model in SUPPORTED_MODELS if model != "forestdiffusion"]
                    item["blocked_forestdiffusion_input"] = [frame_record(f) for _, f in
                                                               generator_inputs(split, "forestdiffusion", diagnostic=True)]
                reports.append(item)
            diversity[dataset] = {"distinct_test_sets": len({h for _, h, _ in tests}),
                                  "seed_hashes": {str(s): h for s, h, _ in tests},
                                  "pairwise_shared_test_rows": [{"seeds": [a[0], b[0]], "rows": len(a[2] & b[2])}
                                                                for i, a in enumerate(tests) for b in tests[i+1:]]}
    sources_after = {name: digest(path) for name, path in REVISION_A1_SOURCE_FILES.items()}
    require(sources_after == run.manifest["sources_before"], "Sources changed during audit")
    require(before_code == {p: digest(ROOT_DIR / p) for p in code_paths}, "Code changed during audit")
    return {"run_id": run.run_id, "runtime": run.runtime, "sources_after": sources_after,
            "preparation_runtime": run.manifest["dependencies"],
            "training_performed": False, "simulated": simulate,
            "verified_combinations": len(reports), "split_diversity": diversity, "combinations": reports,
            "zero_distance_fixture_real_metrics": zero_geometry_probe(),
            "code_sha256": before_code,
            "limitations": ["Generator libraries were not fitted or sampled; wrapper internal/library behavior remains unverified at runtime.",
                            "Only geometric privacy metrics ran with real implementations; query probes are real rows.",
                            "ForestDiffusion actual execution is blocked; its full-data input contract is diagnostic only."]}
