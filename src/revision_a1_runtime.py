"""Strict A1 artifact consumer and isolated sampler/evaluator entry points.

This module never loads historical data, splits, imputes or postprocesses samples.
Source replay is validation only: components always receive the saved NPY arrays.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.const import (DATA_REVISION_A1_DIR, MODELS_DIR, PARAMS_FILE,
                          RESULTS_DIR, TRAIN_MODEL_INFO_DIR, REVISION_A1_SOURCE_FILES, ROOT_DIR,
                          SUPPORTED_MODELS)
from src.revision_a1 import Preprocessor, SourceData, digest, encode, load_source, write_json
from src.timing import GENERATION_TIMING_FILENAME, PhaseTiming

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


def generation_phase(run, split, model, label, phase, output, **details):
    """One common CSV for the real revised entry point, including ForestDiffusion.

    Joint fits/returns (TabPFGen) are not divided into invented per-class times.
    The requested class identifies which sampler request consumed that return.
    """
    if not hasattr(run, '_timing_execution_id'):
        import uuid
        run._timing_execution_id = uuid.uuid4().hex
    return PhaseTiming(RESULTS_DIR / GENERATION_TIMING_FILENAME,
        execution_id=run._timing_execution_id, run_id=run.run_id, experiment_id=output.name,
        generator=model, dataset=split.dataset, seed=split.seed,
        **{'class': 'joint' if label is None else label}, phase=phase,
        artifact_dir=str(output), **details)


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
                 "src/training_workspace.py", "src/utils/seed_utils.py", "src/revision_utility.py", "src/revision_validity.py",
                 "src/timing.py")]
        for folder in ("src/generators", "src/evaluation"):
            paths.extend(sorted((ROOT_DIR / folder).glob("*.py")))
        if model == "forestdiffusion":
            paths.append(ROOT_DIR / "src/revision_forest.py")
        identity = {"protocol": "revision_a1_v1", "run_id": self.run_id,
                "run_manifest_sha256": digest(self.directory / "run_manifest.json"),
                "dataset": split.dataset, "seed": split.seed, "model": model,
                "artifact_hashes": split.artifact_hashes, "config": self.config,
                "runtime": self.runtime,
                "code_sha256": {str(p.relative_to(ROOT_DIR)): digest(p) for p in paths}}
        if model == 'forestdiffusion':
            from src.revision_forest import parameter_record
            identity['generator_parameters'] = parameter_record(self.config['model_params'][model])
        else:
            from src.revision_generator_parameters import parameter_plan
            identity["generator_parameters"] = parameter_plan(model, self.config["model_params"][model], split.seed)
            identity["code_sha256"]["src/revision_generator_parameters.py"] = digest(ROOT_DIR / "src/revision_generator_parameters.py")
            if model == "tabula" and any(k in self.config['model_params'][model] for k in ('pilot_sampling_limits', 'sampling_limits')):
                identity["code_sha256"]["src/revision_tabula_budget.py"] = digest(ROOT_DIR / "src/revision_tabula_budget.py")
            if model == "tabula" and self.config['model_params'][model].get('sampling_strategy') == 'schema_constrained':
                identity["code_sha256"]["src/revision_tabula_schema.py"] = digest(ROOT_DIR / "src/revision_tabula_schema.py")
            if model in ("ctgan", "tvae", "copulagan", "gaussian_copula"):
                from src.revision_sdv import schema_record
                parameters = identity["generator_parameters"]["effective_wrapper_parameters"]
                parameters["_revision_parameters"]["sdv_semantic_schema"] = schema_record(
                    split.names, self.policy["datasets"][split.dataset]["types"])
                identity["code_sha256"]["src/revision_sdv.py"] = digest(ROOT_DIR / "src/revision_sdv.py")
        return identity

    def paths(self, split, model):
        identity = self.identity(split, model)
        experiment = value_hash(identity)
        relative = Path("revision_a1") / self.run_id / split.dataset / f"seed_{split.seed}" / model / experiment
        return identity, MODELS_DIR / relative, RESULTS_DIR / relative


def generator_inputs(split, model, *, diagnostic=False):
    """Classwise predictors; only supervised TabPFGen receives the full target."""
    require(model in SUPPORTED_MODELS, f"Unknown model: {model}")
    train = pd.DataFrame(split.X_train.copy(), columns=split.names)
    train["class"] = split.y_train.copy()
    if model == "tabpfgen":
        return [(None, train.rename(columns={"class": "__target__"}))]
    return [(label, train.loc[train["class"] == label, split.names].reset_index(drop=True)) for label in (0, 1)]


def real_numeric_matrix(frame):
    """A checked temporary view, never a coercion of text or persisted samples.

    Nullable numeric columns can yield an object ndarray despite numeric column
    dtypes. Only that validated case uses float64, with exact value checks.
    NumPy numeric arrays retain their existing representation.
    """
    require(all(pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_complex_dtype(t)
                for t in frame.dtypes), "Nonnumeric or complex synthetic data")
    require(not frame.isna().to_numpy().any(), "Nonfinite synthetic data: missing values")
    matrix = frame.to_numpy(copy=True)
    if matrix.dtype.kind == "O":
        matrix = frame.to_numpy(dtype=np.float64, copy=True)
        require(np.isfinite(matrix).all(), "Nonfinite synthetic data")
        for j, dtype in enumerate(frame.dtypes):
            original = frame.iloc[:, j].tolist()
            if pd.api.types.is_integer_dtype(dtype):
                same = all(int(actual) == int(expected) for actual, expected in zip(matrix[:, j], original))
            else:
                same = all(actual == expected for actual, expected in zip(matrix[:, j].tolist(), original))
            require(same, "Numeric representation would change synthetic values")
    require(matrix.dtype.kind in "biuf" and np.isfinite(matrix).all(), "Nonfinite synthetic data")
    return matrix


def validate_synthetic(frame, names, n):
    require(list(frame.columns) == names, "Synthetic column names/order mismatch")
    require(len(frame) == n, "Synthetic sample shortfall/excess")
    real_numeric_matrix(frame)
    return frame.copy()


def sample_split(run, split, model, factory, *, forest_simulation_backend=None, simulation_root=None):
    if model == "forestdiffusion":
        from src.revision_forest import sample_forest_split
        return sample_forest_split(run, split, backend_factory=forest_simulation_backend, base=simulation_root)
    from src.revision_validity import diagnose, table_record, training_reference
    inputs = generator_inputs(split, model)  # Reject blocked model before creating output.
    identity, model_dir, output = run.paths(split, model)
    require(not model_dir.exists() and not output.exists(), "A1 never reuses/overwrites existing models or samples")
    output.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    write_json(output / "identity.json", identity)
    parameters = identity["generator_parameters"]
    write_json(output / "generator_parameters.json", parameters)
    n = run.config["experiment"]["n_samples_per_class"]
    records = []
    sample_hashes = {}
    reference = training_reference(split.X_train, split.names, run.policy["datasets"][split.dataset],
                                   split.preprocessing.train_categories)
    write_json(output / "validity_reference.json", reference)
    for label, frame in inputs:
        class_dir = model_dir if label is None else model_dir / f"class_{label}"
        generator = factory(model, parameters["effective_wrapper_parameters"])
        records.append({"label": label, "fit": frame_record(frame)})
        with generation_phase(run, split, model, label, 'fit', output, input_rows=len(frame)):
            generator.fit(frame.copy(), class_dir)
        for requested in ((0, 1) if label is None else (label,)):
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with generation_phase(run, split, model, label, 'sampling', output,
                                          requested_class=requested, requested_per_class=n) as timing:
                        generated = generator.sample(n, class_dir)
                        timing['returned_rows'] = len(generated)
                        observation = getattr(generator, 'sampling_observations', None)
                        if isinstance(observation, dict) and 'cache_reused' in observation:
                            timing['cache_reused'] = observation['cache_reused']
            except Exception as exc:
                write_json(output / f"sampling_class_{requested}_execution.json", {
                    "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
                    "warnings": [str(w.message) for w in caught], "produced_rows_observed": None})
                raise
            raw = generated.copy()
            raw_path = output / f"raw_return_class_{requested}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(raw_path, index=False, mode="x")
            internal = getattr(generator, "sampling_observations", None)
            discarded = 0
            boundary = diagnose(raw.drop(columns="__label__", errors="ignore"), split.X_train, split.names,
                                reference, requested=n, stage="wrapper_return_before_sampler_selection",
                                internal_observation=internal)
            boundary["wrapper_return_hash"] = table_record(raw)
            boundary["warnings"] = [str(w.message) for w in caught]
            if "__label__" in raw:
                boundary["observed_label_counts"] = {str(k): int(v) for k, v in raw["__label__"].value_counts(dropna=False).items()}
            write_json(output / f"wrapper_diagnostic_class_{requested}.json", boundary)
            if label is None:
                require(list(generated.columns) == split.names + ["__label__"], "Missing generated labels")
                require(generated["__label__"].isin([0, 1]).all(), "Invalid generated labels")
                generated = generated.loc[generated["__label__"] == requested, split.names].reset_index(drop=True)
                discarded = len(raw) - len(generated)
            diagnostic = diagnose(generated, split.X_train, split.names, reference, requested=n,
                                  stage="wrapper_return_after_label_selection_before_validation",
                                  discarded_observed=discarded, delivered=None, internal_observation=internal)
            diagnostic["wrapper_return_rows"] = len(raw)
            diagnostic["produced_rows_observed"] = len(raw)
            diagnostic["diagnosed_rows_after_label_selection"] = len(generated)
            diagnostic["wrapper_return_hash"] = table_record(raw)
            diagnostic["raw_return_csv_sha256"] = digest(raw_path)
            diagnostic["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
            try:
                generated = validate_synthetic(generated, split.names, n)
            except Exception as exc:
                diagnostic["status"] = "failed_validation"
                diagnostic["delivered_rows"] = 0
                diagnostic["withheld_rows_due_to_failed_validation"] = len(generated)
                diagnostic["error"] = {"type": type(exc).__name__, "message": str(exc)}
                write_json(output / f"validity_class_{requested}.json", diagnostic)
                raise
            diagnostic["status"] = "delivered_with_observational_diagnostics"
            diagnostic["delivered_rows"] = len(generated)
            write_json(output / f"validity_class_{requested}.json", diagnostic)
            generated["class"] = np.full(n, requested, dtype="<i4")
            path = output / f"synthetic_class_{requested}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            generated.to_csv(path, index=False, mode="x")
            sample_hashes[path.name] = digest(path)
    write_json(output / "samples_manifest.json", {"identity": identity, "inputs": records,
               "files": sample_hashes, "postprocessing": "none; label selection/assignment only",
               "observability_files": {p.name: digest(p) for p in output.iterdir()
                    if p.name.startswith(("validity_", "wrapper_diagnostic_", "raw_return_"))}})


def load_samples(run, split, model):
    generator_inputs(split, model)
    identity, _, output = run.paths(split, model)
    require(read_json(output / "identity.json") == identity, "Experiment identity mismatch")
    require(read_json(output / "generator_parameters.json") == identity["generator_parameters"],
            "Effective generator parameter record mismatch")
    manifest = read_json(output / "samples_manifest.json")
    require(manifest["identity"] == identity, "Samples identity mismatch")
    if model == 'forestdiffusion':
        from src.revision_forest import check_revised_delivery
        check_revised_delivery(run, split, manifest)
    for name, expected in manifest.get("observability_files", {}).items():
        require(Path(name).name == name and digest(output / name) == expected, "Diagnostic/raw return hash mismatch")
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


def evaluate_split(run, split, model, output_irs, evaluate, logger, *, save_classifiers=False, record_utility=False):
    from src.evaluation.builder import build_synthetic_dataset
    identity, _, output = run.paths(split, model)
    synthetic = load_samples(run, split, model)
    if record_utility:
        require(all(ir == 1. for ir in output_irs), "Fixed balanced utility comparison requires IR=1 and effective quota equality")
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
        options = {}
        if record_utility:
            from src.revision_utility import evaluate_utility_conditions
            def utility_block(real_train, real_test, synth, **kwargs):
                records = evaluate_utility_conditions(split, synth, run.config["experiment"]["n_samples_per_class"],
                    metrics_dir / f"ir_{ir}" / "utility", {"dataset": split.dataset, "seed": split.seed, "model": model,
                    "identity": identity}, execution_kind="benchmark", allow_training=True, save_estimators=save_classifiers)
                require(all(r["status"] == "completed" for r in records), "Utility execution failed; see execution.json")
                result = []
                for r in records:
                    for name, value in r["metrics"].items():
                        result.append({"block": "utility_ml", "metric": f"{r['condition']}_{name}",
                                       "classifier": r["context"]["classifier"], "value": value, "status": "ok", "reason": None})
                return result
            options["utility_block"] = utility_block
        rows = evaluate(dataset=split.dataset, model=model, seed=split.seed, output_ir=ir,
                        synth_final=final, real_train=train.copy(), real_test=test.copy(),
                        target="class", logger=logger,
                        classifiers_dir=(metrics_dir / f"ir_{ir}" / "classifiers" if save_classifiers else None),
                        audit_path=metrics_dir / f"ir_{ir}" / "degenerate.csv", timed_block=strict_call, **options)
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


def overwrite_directories(run, split, model):
    """Only the selected revised combination, including its previous identities.

    Validate every component before destructive operations. Symlinks may be
    descriptive aliases elsewhere; never follow one while replacing artifacts.
    """
    require(run.run_id == PINNED_RUN and split.dataset in run.policy['datasets'] and
            split.seed in run.policy['seeds'] and model in SUPPORTED_MODELS, 'Invalid overwrite scope')
    relative = Path('revision_a1') / run.run_id / split.dataset / f'seed_{split.seed}' / model
    directories = []
    for root in (MODELS_DIR, RESULTS_DIR, TRAIN_MODEL_INFO_DIR):
        directory = Path(root).absolute()
        require(not directory.is_symlink(), f'Overwrite refuses symbolic link: {directory}')
        for part in relative.parts:
            directory = directory / part
            require(not directory.is_symlink(), f'Overwrite refuses symbolic link: {directory}')
            require(not directory.exists() or directory.is_dir(), f'Overwrite path is not a directory: {directory}')
        require(directory.resolve().is_relative_to(Path(root).resolve()), 'Overwrite path escapes artifact root')
        directories.append(directory)
    return directories


def run_sampler(args):
    force_retrain = getattr(args, "force_retrain", False)
    run = RevisedRun(args.revision_a1_run, config_path=args.config)
    datasets, seeds, models = selection(args, run)
    # Preflight everything before any training or output writes.
    splits = [run.load(ds, seed) for ds in datasets for seed in seeds]
    for split in splits:
        for model in models:
            generator_inputs(split, model, diagnostic=args.dry_run)
            if model == 'forestdiffusion':
                from src.revision_forest import revised_plans
                revised_plans(run, split)
            if model == 'tabula' and not args.dry_run:
                from src.revision_tabula_budget import validate_limits
                params = run.config['model_params'][model]
                require(not ('sampling_limits' in params and 'pilot_sampling_limits' in params), 'Ambiguous TabuLa limits')
                validate_limits(params.get('sampling_limits', params.get('pilot_sampling_limits', {})))
            _, mdir, out = run.paths(split, model)
            if force_retrain:
                overwrite_directories(run, split, model)  # Validate all cleanup scopes before changing anything.
            elif not args.dry_run:
                require(not mdir.exists() and not out.exists(), "Existing A1 output; no reuse allowed")
    if args.dry_run:
        for split in splits:
            for model in models:
                if force_retrain:
                    for directory in overwrite_directories(run, split, model):
                        print(f'DRY-RUN --force-retrain would replace: {directory}')
                if model == "forestdiffusion":
                    from src.revision_forest import revised_plans
                    for label, frame, plan in revised_plans(run, split):
                        print(f"DRY-RUN ForestDiffusion class={label} rows={len(frame)} "
                              f"variables={len(plan['codec']['variable_columns'])} "
                              f"constants={len(plan['codec']['constants'])} all_constant={plan['codec']['all_constant']} "
                              f"plan={value_hash(plan)}")
                print(f"A1 verified {split.dataset} seed={split.seed} model={model}" +
                      (" DRY-RUN ONLY; revised classwise adapter connected" if model == "forestdiffusion" else ""))
        return
    from src.generators import get_generator
    from src.utils.seed_utils import set_global_seed
    for split in splits:
        for model in models:
            if force_retrain:
                for directory in overwrite_directories(run, split, model):
                    if directory.exists():
                        print(f'--force-retrain replacing: {directory}', flush=True)
                        shutil.rmtree(directory)
            set_global_seed(split.seed)
            sample_split(run, split, model, get_generator)


def run_evaluator(args, config, logger, evaluate):
    run = RevisedRun(args.revision_a1_run, config_path=args.config)
    datasets, seeds, models = selection(args, run)
    require(all(0 < ir <= 1 for ir in args.ir) and len(set(args.ir)) == len(args.ir), "Invalid/duplicate IR")
    if not args.dry_run:
        require(all(ir == 1. for ir in args.ir), "Revised fixed balanced utility comparison requires --ir 1.0")
    splits = [run.load(ds, seed) for ds in datasets for seed in seeds]
    if args.dry_run:
        for split in splits:
            for model in models:
                print(f"A1 verified evaluator {split.dataset} seed={split.seed} model={model}" +
                      (" DRY-RUN ONLY; revised classwise reader connected" if model == "forestdiffusion" else ""))
        return
    for split in splits:
        for model in models:
            load_samples(run, split, model)
            require(not (run.paths(split, model)[2] / "metrics").exists(), "Existing A1 metrics")
    for split in splits:
        for model in models:
            evaluate_split(run, split, model, args.ir, evaluate, logger, save_classifiers=args.save_classifiers, record_utility=True)
