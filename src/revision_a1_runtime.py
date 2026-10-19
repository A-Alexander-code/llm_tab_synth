"""Strict A1 artifact consumer and isolated sampler/evaluator entry points.

This module never loads historical data, splits, imputes or postprocesses samples.
Source replay is validation only: components always receive the saved NPY arrays.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.const import (DATA_REVISION_A1_DIR, MODELS_DIR, PARAMS_FILE,
                          RESULTS_DIR, REVISION_A1_SOURCE_FILES, ROOT_DIR,
                          SUPPORTED_MODELS)
from src.revision_a1 import Preprocessor, SourceData, digest, encode, load_source, write_json

PINNED_RUN = "33a9b4fce40753d94f4a3e2bf0c193f129e8edb5b6a19bb393544f747e2f3543"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def value_hash(value):
    return hashlib.sha256(encode(value)).hexdigest()


def array_record(value):
    """Hash shape, dtype, and C-order values, independently of NPY headers."""
    array = np.asarray(value)
    header = {"shape": list(array.shape), "dtype": array.dtype.str}
    return {**header, "sha256": hashlib.sha256(encode(header) + array.tobytes(order="C")).hexdigest()}


def frame_record(frame):
    return {"columns": list(frame.columns),
            "columns_data": {c: array_record(frame[c].to_numpy()) for c in frame.columns}}


def runtime_record():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "torch", "transformers",
                 "ForestDiffusion", "tabpfgen", "be-great", "realtabformer", "ctgan", "sdv", "mostlyai-engine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages}


@dataclass
class RevisedSplit:
    dataset: str
    seed: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    names: list[str]
    partition: dict
    meta: dict
    preprocessing: Preprocessor
    source: SourceData
    artifact_hashes: dict

    def frames(self):
        frames = []
        for side in ("train", "test"):
            frame = pd.DataFrame(getattr(self, f"X_{side}").copy(), columns=self.names)
            frame["class"] = getattr(self, f"y_{side}").copy()
            frames.append(frame)
        return tuple(frames)


class RevisedRun:
    def __init__(self, run_id, *, data_root=DATA_REVISION_A1_DIR, config_path=PARAMS_FILE):
        require(run_id == PINNED_RUN, "A1 requires the explicitly audited, pinned run ID")
        self.run_id = run_id
        self.directory = Path(data_root) / run_id
        self.manifest = read_json(self.directory / "run_manifest.json")
        identity = self.manifest["identity"]
        require(identity["fingerprint"] == run_id, "Run ID mismatch")
        require(value_hash({k: v for k, v in identity.items() if k != "fingerprint"}) == run_id,
                "Run fingerprint mismatch")
        require(self.manifest["passed"] and not self.manifest["failures"] and
                self.manifest["source_hashes_unchanged"], "Incomplete or failed preparation")
        self.config = read_json(Path(config_path))
        require(self.config["experiment"]["target_column"] == "class", "A1 target must be class")
        require(isinstance(self.config["experiment"]["n_samples_per_class"], int) and
                self.config["experiment"]["n_samples_per_class"] > 0, "Invalid sample quota")
        require(self.config["revision_a1"] == identity["config"], "Preparation config mismatch")
        self.policy = self.config["revision_a1"]
        require(set(self.policy["datasets"]) == set(REVISION_A1_SOURCE_FILES), "Source inventory mismatch")
        require(self.manifest["successful_combinations"] ==
                len(self.policy["datasets"]) * len(self.policy["seeds"]), "Combination count mismatch")
        for name, expected in self.manifest["run_artifact_sha256"].items():
            require(Path(name).name == name, "Unsafe manifest path")
            require(digest(self.directory / name) == expected, f"Run artifact hash mismatch: {name}")
        self.summaries = {(s["dataset"], s["seed"]): s for s in read_json(self.directory / "summary.json")}
        require(len(self.summaries) == self.manifest["successful_combinations"], "Summary count mismatch")
        for name, expected in identity["code_sha256"].items():
            require(digest(ROOT_DIR / name) == expected, f"Preparation code changed: {name}")
        self.sources = {}
        for dataset, spec in self.policy["datasets"].items():
            source = load_source(REVISION_A1_SOURCE_FILES[dataset], spec)
            require(source.source_hash == identity["source_sha256"][dataset] ==
                    self.manifest["sources_before"][dataset] == self.manifest["sources_after"][dataset],
                    f"Source hash mismatch: {dataset}")
            self.sources[dataset] = source
        self.loaded = {}
        self.runtime = runtime_record()

    def load(self, dataset, seed):
        require(dataset in self.policy["datasets"] and seed in self.policy["seeds"], "Unknown dataset/seed")
        key = (dataset, seed)
        if key in self.loaded:
            return self.loaded[key]
        directory = self.directory / dataset / f"seed_{seed}"
        source = self.sources[dataset]
        hashes = read_json(directory / "artifact_hashes.json")
        required = {"preprocessing.json", "partition.json", "verification.json", "schema.json", "source_audit.json",
                    "feature_meta.json", "provenance.json", "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"}
        require(set(hashes) == required, "Artifact inventory mismatch")
        for name, expected in hashes.items():
            require(digest(directory / name) == expected, f"Artifact hash mismatch: {name}")
        require(read_json(directory / "provenance.json") == self.manifest["identity"], "Provenance mismatch")
        require(read_json(directory / "schema.json") == self.policy["datasets"][dataset], "Schema mismatch")
        require(read_json(directory / "source_audit.json") == json.loads(encode(source.audit)), "Source audit mismatch")
        require(read_json(directory / "verification.json")["passed"], "Preparation checks failed")
        prep = Preprocessor.from_dict(read_json(directory / "preprocessing.json"))
        meta = read_json(directory / "feature_meta.json")
        require(meta == {k: v for k, v in self.summaries[key].items() if k not in {"dataset", "source_hash_unchanged"}},
                "Summary/metadata mismatch")
        require(prep.names == source.names and meta["source_sha256"] == source.source_hash, "Preprocessing source mismatch")
        partition = read_json(directory / "partition.json")
        require(meta["seed"] == seed and list(prep.kept) == meta["feature_names"], "Column order/seed mismatch")
        arrays = {}
        positions = {}
        for side in ("train", "test"):
            X = np.load(directory / f"X_{side}.npy", allow_pickle=False)
            y = np.load(directory / f"y_{side}.npy", allow_pickle=False)
            require(X.dtype == np.dtype("<f8") and y.dtype == np.dtype("<i4"), "Unexpected NPY dtype")
            require(X.shape == (meta[side]["rows"], len(prep.kept)) and y.shape == (len(X),), "Shape mismatch")
            require(np.isfinite(X).all() and np.isfinite(y).all(), "Nonfinite matrix")
            ids = partition[side]
            require(len(ids) == len(X), "Partition alignment mismatch")
            positions[side] = [r["source_position"] - 1 for r in ids]
            for j, (row, i) in enumerate(zip(ids, positions[side])):
                require(row["matrix_row"] == j and row["row_id"] == source.row_ids[i] and
                        row["group_id"] == source.group_ids[i], "Row/group alignment mismatch")
            require(np.array_equal(X, np.asarray(prep.transform([source.X[i] for i in positions[side]]))),
                    "JSON preprocessing replay mismatch")
            require(np.array_equal(y, np.asarray([source.y[i] for i in positions[side]], dtype="<i4")),
                    "Target alignment mismatch")
            require({str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} ==
                    meta[side]["class_counts"], "Class count mismatch")
            X.setflags(write=False)
            y.setflags(write=False)
            arrays[f"X_{side}"] = X
            arrays[f"y_{side}"] = y
        require(not set(positions["train"]) & set(positions["test"]) and
                sorted(positions["train"] + positions["test"]) == list(range(len(source.X))), "Row coverage mismatch")
        require(not {r["group_id"] for r in partition["train"]} &
                {r["group_id"] for r in partition["test"]}, "Group leakage")
        split = RevisedSplit(dataset, seed, **arrays, names=list(prep.kept), partition=partition,
                             meta=meta, preprocessing=prep, source=source,
                             artifact_hashes=hashes)
        self.loaded[key] = split
        return split

    def identity(self, split, model):
        require(model in SUPPORTED_MODELS, f"Unknown model: {model}")
        # Full parameters, labels, schema, split and implementation are all bound.
        paths = [ROOT_DIR / p for p in ("src/revision_a1_runtime.py", "main_sampler.py", "main_evaluator.py",
                 "src/training_workspace.py", "src/utils/seed_utils.py")]
        for folder in ("src/generators", "src/evaluation"):
            paths.extend(sorted((ROOT_DIR / folder).glob("*.py")))
        return {"protocol": "revision_a1_v1", "run_id": self.run_id,
                "run_manifest_sha256": digest(self.directory / "run_manifest.json"),
                "dataset": split.dataset, "seed": split.seed, "model": model,
                "artifact_hashes": split.artifact_hashes, "config": self.config,
                "runtime": self.runtime,
                "code_sha256": {str(p.relative_to(ROOT_DIR)): digest(p) for p in paths}}

    def paths(self, split, model):
        identity = self.identity(split, model)
        experiment = value_hash(identity)
        relative = Path("revision_a1") / self.run_id / split.dataset / f"seed_{split.seed}" / model / experiment
        return identity, MODELS_DIR / relative, RESULTS_DIR / relative


def generator_inputs(split, model, *, diagnostic=False):
    """The existing classwise protocol, or supervised full-data TabPFGen."""
    require(model in SUPPORTED_MODELS, f"Unknown model: {model}")
    if model == "forestdiffusion" and not diagnostic:
        raise ValueError("ForestDiffusion blocked: generated target is discarded and external labels are assigned")
    train, _ = split.frames()
    if model in {"tabpfgen", "forestdiffusion"}:
        return [(None, train.rename(columns={"class": "__target__"}))]
    return [(label, train.loc[train["class"] == label, split.names].reset_index(drop=True)) for label in (0, 1)]


def validate_synthetic(frame, names, n):
    require(list(frame.columns) == names, "Synthetic column names/order mismatch")
    require(len(frame) == n, "Synthetic sample shortfall/excess")
    require(all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes), "Nonnumeric synthetic data")
    require(np.isfinite(frame.to_numpy()).all(), "Nonfinite synthetic data")
    return frame.copy()


def sample_split(run, split, model, factory):
    inputs = generator_inputs(split, model)  # Reject blocked model before creating output.
    identity, model_dir, output = run.paths(split, model)
    require(not model_dir.exists() and not output.exists(), "A1 never reuses/overwrites existing models or samples")
    output.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    write_json(output / "identity.json", identity)
    n = run.config["experiment"]["n_samples_per_class"]
    records = []
    sample_hashes = {}
    for label, frame in inputs:
        class_dir = model_dir if label is None else model_dir / f"class_{label}"
        generator = factory(model, run.config["model_params"][model])
        records.append({"label": label, "fit": frame_record(frame)})
        generator.fit(frame.copy(), class_dir)
        for requested in ((0, 1) if label is None else (label,)):
            generated = generator.sample(n, class_dir)
            if label is None:
                require(list(generated.columns) == split.names + ["__label__"], "Missing generated labels")
                require(generated["__label__"].isin([0, 1]).all(), "Invalid generated labels")
                generated = generated.loc[generated["__label__"] == requested, split.names].reset_index(drop=True)
            generated = validate_synthetic(generated, split.names, n)
            generated["class"] = np.full(n, requested, dtype="<i4")
            path = output / f"synthetic_class_{requested}.csv"
            generated.to_csv(path, index=False, mode="x")
            sample_hashes[path.name] = digest(path)
    write_json(output / "samples_manifest.json", {"identity": identity, "inputs": records,
               "files": sample_hashes, "postprocessing": "none; label selection/assignment only"})


def load_samples(run, split, model):
    generator_inputs(split, model)  # Also block evaluation of invalid ForestDiffusion output.
    identity, _, output = run.paths(split, model)
    require(read_json(output / "identity.json") == identity, "Experiment identity mismatch")
    manifest = read_json(output / "samples_manifest.json")
    require(manifest["identity"] == identity, "Samples identity mismatch")
    require(set(manifest["files"]) == {f"synthetic_class_{k}.csv" for k in (0, 1)}, "Sample inventory mismatch")
    frames = []
    for label in (0, 1):
        name = f"synthetic_class_{label}.csv"
        require(digest(output / name) == manifest["files"][name], "Synthetic CSV hash mismatch")
        frame = pd.read_csv(output / name, float_precision="round_trip")
        require(list(frame.columns) == split.names + ["class"], "Synthetic schema mismatch")
        require((frame["class"] == label).all(), "Synthetic label mismatch")
        validate_synthetic(frame[split.names], split.names, run.config["experiment"]["n_samples_per_class"])
        frames.append(frame)
    return frames


def evaluate_split(run, split, model, output_irs, evaluate, logger, *, save_classifiers=False):
    from src.evaluation.builder import build_synthetic_dataset
    identity, _, output = run.paths(split, model)
    synthetic = load_samples(run, split, model)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(exist_ok=False)
    train, test = split.frames()
    majority = int(np.argmax(np.bincount(split.y_train)))
    evidence = []
    for ir in output_irs:
        final = build_synthetic_dataset(synthetic[majority], synthetic[1-majority], ir, split.seed,
                                        maj_label=majority, min_label=1-majority)
        calls = []

        def strict_call(fn, label, prefix, log, *args, **kwargs):
            calls.append({"component": fn.__name__, "inputs": [frame_record(f) for f in args if isinstance(f, pd.DataFrame)]})
            result = fn(*args, **kwargs)
            require(all(r.get("status") != "not_computable" for r in result), f"Metric failed: {label}")
            return result

        # _evaluate_combination accepts an explicit dispatcher in revised mode.
        rows = evaluate(dataset=split.dataset, model=model, seed=split.seed, output_ir=ir,
                        synth_final=final, real_train=train.copy(), real_test=test.copy(),
                        target="class", logger=logger,
                        classifiers_dir=(metrics_dir / f"ir_{ir}" / "classifiers" if save_classifiers else None),
                        audit_path=metrics_dir / f"ir_{ir}" / "degenerate.csv", timed_block=strict_call)
        pd.DataFrame(rows).to_csv(metrics_dir / f"ir_{ir}.csv", index=False, mode="x")
        evidence.append({"ir": ir, "calls": calls, "synthetic": frame_record(final)})
    write_json(metrics_dir / "input_hashes.json", {"identity": identity, "calls": evidence,
                                                  "save_classifiers": save_classifiers})


def selection(args, run):
    datasets = args.datasets or list(run.policy["datasets"])
    seeds = args.seeds or run.policy["seeds"]
    models = args.models or run.config["experiment"]["models"]
    require(set(datasets) <= set(run.policy["datasets"]) and set(seeds) <= set(run.policy["seeds"]), "Unknown A1 dataset/seed")
    require(set(models) <= set(SUPPORTED_MODELS), "Unknown model")
    require(all(len(values) == len(set(values)) for values in (datasets, seeds, models)), "Duplicate selection")
    return datasets, seeds, models


def run_sampler(args):
    require(not getattr(args, "force_retrain", False), "A1 does not support --force-retrain; existing outputs are never reused")
    run = RevisedRun(args.revision_a1_run, config_path=args.config)
    datasets, seeds, models = selection(args, run)
    # Preflight everything before any training or output writes.
    splits = [run.load(ds, seed) for ds in datasets for seed in seeds]
    for split in splits:
        for model in models:
            generator_inputs(split, model, diagnostic=args.dry_run)
            _, mdir, out = run.paths(split, model)
            if not args.dry_run:
                require(not mdir.exists() and not out.exists(), "Existing A1 output; no reuse allowed")
    if args.dry_run:
        for split in splits:
            for model in models:
                print(f"A1 verified {split.dataset} seed={split.seed} model={model}" +
                      (" BLOCKED: invalid ForestDiffusion labeling" if model == "forestdiffusion" else ""))
        return
    from src.generators import get_generator
    from src.utils.seed_utils import set_global_seed
    for split in splits:
        for model in models:
            set_global_seed(split.seed)
            sample_split(run, split, model, get_generator)


def run_evaluator(args, config, logger, evaluate):
    run = RevisedRun(args.revision_a1_run, config_path=args.config)
    datasets, seeds, models = selection(args, run)
    require(all(0 < ir <= 1 for ir in args.ir) and len(set(args.ir)) == len(args.ir), "Invalid/duplicate IR")
    splits = [run.load(ds, seed) for ds in datasets for seed in seeds]
    if args.dry_run:
        for split in splits:
            for model in models:
                print(f"A1 verified evaluator {split.dataset} seed={split.seed} model={model}" +
                      (" BLOCKED: invalid ForestDiffusion labeling" if model == "forestdiffusion" else ""))
        return
    for split in splits:
        for model in models:
            load_samples(run, split, model)
            require(not (run.paths(split, model)[2] / "metrics").exists(), "Existing A1 metrics")
    for split in splits:
        for model in models:
            evaluate_split(run, split, model, args.ir, evaluate, logger, save_classifiers=args.save_classifiers)
