"""Integration tests require the pinned preparation and the benchmark venv."""
import argparse
import copy
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.const import DATA_REVISION_A1_DIR, PARAMS_FILE, SUPPORTED_MODELS
from src.revision_a1 import digest, encode
from src.revision_a1_audit import geometry, overlap_stages, simulated_flow, zero_geometry_probe
from src.revision_a1_runtime import (PINNED_RUN, RevisedRun, evaluate_split, generator_inputs,
                                     load_samples, run_sampler, sample_split, validate_synthetic,
                                     value_hash)


class A1IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_run = RevisedRun(PINNED_RUN)
        cls.split = cls.artifact_run.load("cervical_cancer", 13)

    def copy_one_split(self, temporary):
        root = Path(temporary)
        source = DATA_REVISION_A1_DIR / PINNED_RUN
        target = root / PINNED_RUN
        target.mkdir()
        for name in ("run_manifest.json", "summary.json", "summary.csv", "failures.json"):
            shutil.copy2(source / name, target / name)
        directory = target / self.split.dataset / "seed_13"
        shutil.copytree(source / self.split.dataset / "seed_13", directory)
        return root, directory

    def test_numpy_load_all_twenty_and_distinct_test_sets(self):
        count = 0
        for ds in self.artifact_run.policy["datasets"]:
            hashes = []
            for seed in self.artifact_run.policy["seeds"]:
                split = self.artifact_run.load(ds, seed)
                self.assertEqual(split.X_train.dtype, np.dtype("<f8"))
                self.assertEqual(split.y_test.dtype, np.dtype("<i4"))
                hashes.append(value_hash(sorted(r["row_id"] for r in split.partition["test"])))
                count += 1
            self.assertEqual(len(set(hashes)), 5)
        self.assertEqual(count, 20)

    def test_missing_artifact_stops_without_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, directory = self.copy_one_split(temporary)
            (directory / "X_test.npy").unlink()
            with self.assertRaises(FileNotFoundError):
                RevisedRun(PINNED_RUN, data_root=root).load(self.split.dataset, 13)

    def test_corrupted_npy_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, directory = self.copy_one_split(temporary)
            path = directory / "X_train.npy"
            path.write_bytes(path.read_bytes()[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                RevisedRun(PINNED_RUN, data_root=root).load(self.split.dataset, 13)

    def test_rehashed_row_id_or_column_order_still_rejected(self):
        for filename in ("partition.json", "feature_meta.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root, directory = self.copy_one_split(temporary)
                path = directory / filename
                value = json.loads(path.read_text())
                if filename == "partition.json":
                    value["train"][0]["row_id"] = "misaligned"
                else:
                    value["feature_names"].reverse()
                path.write_bytes(encode(value))
                hashes_path = directory / "artifact_hashes.json"
                hashes = json.loads(hashes_path.read_text())
                hashes[filename] = digest(path)
                hashes_path.write_bytes(encode(hashes))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    RevisedRun(PINNED_RUN, data_root=root).load(self.split.dataset, 13)

    def test_numpy_dtype_and_finiteness_are_checked_independently(self):
        original_load = np.load
        for change in (lambda x: x.astype("float32"), lambda x: np.full_like(x, np.nan)):
            def altered(path, **kwargs):
                value = original_load(path, **kwargs)
                return change(value) if Path(path).name == "X_train.npy" else value
            with self.subTest(change=change), patch("src.revision_a1_runtime.np.load", side_effect=altered):
                with self.assertRaises(ValueError):
                    RevisedRun(PINNED_RUN).load(self.split.dataset, 13)

    def test_unknown_run_dataset_seed_and_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            RevisedRun("../processed")
        for ds, seed in (("historical", 13), (self.split.dataset, 999)):
            with self.assertRaises(ValueError):
                self.artifact_run.load(ds, seed)
        config = copy.deepcopy(self.artifact_run.config)
        config["revision_a1"]["test_fraction"] = .3
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "params.json"
            path.write_bytes(encode(config))
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                RevisedRun(PINNED_RUN, config_path=path)

    def test_forest_routes_to_classwise_adapter_without_legacy_factory(self):
        def forbidden(*args):
            self.fail("Real factory must not run")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch("src.revision_a1_runtime.RESULTS_DIR", base / "results"), \
                 patch('src.revision_forest._sample_revised_split', side_effect=RuntimeError('explicit adapter boundary')) as adapter:
                with self.assertRaisesRegex(RuntimeError, 'explicit adapter boundary'):
                    sample_split(self.artifact_run, self.split, "forestdiffusion", forbidden)
                adapter.assert_called_once_with(self.artifact_run, self.split)
            self.assertEqual(list(base.iterdir()), [])

    def test_all_nine_simulated_sampler_evaluator_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for model in SUPPORTED_MODELS:
                if model == "forestdiffusion":
                    continue
                evidence = simulated_flow(self.artifact_run, self.split, model, Path(temporary))
                self.assertEqual([r["mode"] for r in evidence["classifier_arrays"]], ["trtr", "tstr"])
                self.assertEqual(len(evidence["generator_fit"]), 1 if model == "tabpfgen" else 2)
                self.assertEqual(len(evidence["metric_dispatch"][0]["calls"]), 12)

    def test_revised_dry_run_does_not_call_historical_split_or_factory(self):
        import main_sampler
        uncached_run = RevisedRun(PINNED_RUN)
        args = argparse.Namespace(revision_a1_run=PINNED_RUN, config=PARAMS_FILE,
                                  models=["ctgan"], datasets=[self.split.dataset], seeds=[13], dry_run=True)
        with patch.object(main_sampler, "split_and_save", side_effect=AssertionError("historical split")), \
             patch.object(main_sampler, "get_generator", side_effect=AssertionError("real generator")), \
             patch("src.revision_a1.split_groups", side_effect=AssertionError("revised split must not rerun")), \
             patch("src.revision_a1.Preprocessor.fit", side_effect=AssertionError("preprocessing must not refit")), \
             patch("src.revision_a1_runtime.RevisedRun", return_value=uncached_run):
            main_sampler.run(args)

    def test_identity_includes_all_parameters_and_no_model_reuse(self):
        model = "ctgan"
        identity, _, _ = self.artifact_run.paths(self.split, model)
        changed = copy.deepcopy(identity)
        changed["config"]["model_params"][model]["epochs"] += 1
        self.assertNotEqual(value_hash(identity), value_hash(changed))
        changed = copy.deepcopy(identity)
        changed["artifact_hashes"]["y_train.npy"] = "different labels"
        self.assertNotEqual(value_hash(identity), value_hash(changed))
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch("src.revision_a1_runtime.MODELS_DIR", base / "models"), \
                 patch("src.revision_a1_runtime.RESULTS_DIR", base / "results"):
                _, model_dir, _ = self.artifact_run.paths(self.split, model)
                model_dir.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "never reuses"):
                    sample_split(self.artifact_run, self.split, model, lambda *a: self.fail("factory called"))

    def test_invalid_synthetic_schema_values_and_size_stop(self):
        valid = pd.DataFrame({"a": [1.5], "b": [2.25]})
        pd.testing.assert_frame_equal(validate_synthetic(valid, ["a", "b"], 1), valid)
        for frame in (valid[["b", "a"]], valid.assign(a=np.nan), valid.assign(a="text"), valid.iloc[:0]):
            with self.assertRaises(ValueError):
                validate_synthetic(frame, ["a", "b"], 1)

    def test_strict_evaluator_does_not_swallow_metric_failures(self):
        import main_evaluator
        def broken_metric(*args):
            raise RuntimeError("metric broken")
        split, model = self.split, "ctgan"
        train, _ = split.frames()
        samples = [train.loc[train["class"] == k].reset_index(drop=True) for k in (0, 1)]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch("src.revision_a1_runtime.RESULTS_DIR", base), \
                 patch("src.revision_a1_runtime.load_samples", return_value=samples), \
                 patch.object(main_evaluator, "compute_univariate", broken_metric):
                output = self.artifact_run.paths(split, model)[2]
                output.mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, "metric broken"):
                    evaluate_split(self.artifact_run, split, model, [1.], main_evaluator._evaluate_combination,
                                   logging.getLogger("test-a1"))

    def test_overlap_origin_classes_and_real_zero_distance_fallback(self):
        patterns = [2, 2, 3, 2, 0]
        for seed, expected in zip(self.artifact_run.policy["seeds"], patterns):
            split = self.artifact_run.load("cervical_cancer", seed)
            stages = overlap_stages(split)
            self.assertEqual(stages["original_normalized"]["shared_patterns"], 0)
            self.assertEqual(stages["after_column_selection"]["shared_patterns"], 0)
            final = stages["after_imputation"]
            self.assertEqual(final["shared_patterns"], expected)
            self.assertEqual(final["train_rows"], expected)
            self.assertEqual(final["test_rows"], expected)
            if expected:
                self.assertEqual(final["train_classes"], {"0": expected})
                self.assertEqual(final["test_classes"], {"0": expected})
            actual = geometry(split)
            self.assertEqual(actual["zero_distance_test_rows"], expected)
            self.assertFalse(actual["fallback_used"])
        self.assertEqual(zero_geometry_probe()["metrics"]["kanon_epsilon"], 1e-6)


if __name__ == "__main__":
    unittest.main()
