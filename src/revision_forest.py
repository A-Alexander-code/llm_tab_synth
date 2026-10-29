"""Classwise ForestDiffusion protocol, shared by revised execution and validation.

The historical wrapper is deliberately independent. No ForestDiffusion import or
constructor is needed to inspect plans or exercise the explicitly simulated path.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle
import threading
import time
import warnings

import numpy as np
import pandas as pd

from src.revision_a1 import digest, encode, write_json
from src.revision_a1_runtime import frame_record, generation_phase, require, runtime_record, value_hash

PROTOCOL = "revision_forest_classwise_v1"
BLOCK_REASON = "ForestDiffusion blocked: classwise protocol awaits real fit/sample/serialize/reload validation"
REVISED_SCOPE = "forest_revised_classwise_run_v1"
_RNG_LOCK = threading.RLock()
VALIDATION_SCOPE = "forest_minimal_real_validation_v1"
VALIDATION_PARAMS = {"n_t": 4, "n_estimators": 5, "max_depth": 2, "duplicate_K": 2,
                     "n_jobs": 1, "nthread": 1, "gpu_hist": False, "model": "xgboost",
                     "diffusion_type": "flow", "n_batch": 1}


def backend_record(factory):
    require(getattr(factory, "_revision_simulation_backend", False) is True,
            "Only an explicitly marked simulation backend is allowed")
    path = inspect.getsourcefile(factory)
    return {"module": factory.__module__, "name": factory.__qualname__,
            "source_sha256": digest(Path(path)) if path else None}


def validation_backend_record():
    library = library_record()
    return {"module": "ForestDiffusion.diffusion_with_trees_class", "name": "ForestDiffusionModel",
            "version": library["version"], "source_sha256": library["source_sha256"]}


def parameter_record(params):
    return {"protocol": PROTOCOL, "scope": REVISED_SCOPE, "configured": params,
            "library": library_record(), "models": "two independent train-class X-only fits",
            "labels": "assigned from class model identity", "padding": False}


def validate_revised_scope(plan):
    """Bind the general adapter to the saved train partition, not an arbitrary X."""
    from config.const import DATA_REVISION_A1_DIR, ROOT_DIR
    from src.revision_a1_runtime import PINNED_RUN
    context = plan['context']
    require(context.get('protocol') == 'revision_a1_v1' and context.get('run_id') == PINNED_RUN,
            'Explicit pinned revised Forest scope required')
    require(plan['execution_kind'] == 'real_revised' and plan.get('execution_scope') == REVISED_SCOPE,
            'Wrong revised Forest execution scope')
    config = context['config']
    require(context['dataset'] in config['revision_a1']['datasets'] and
            context['seed'] in config['revision_a1']['seeds'], 'Unknown revised Forest dataset/seed')
    require(context['generator_parameters'] == parameter_record(config['model_params']['forestdiffusion']),
            'Forest parameter identity mismatch')
    require(all(digest(ROOT_DIR / p) == h for p, h in context['code_sha256'].items()),
            'Changed pipeline code invalidates revised Forest scope')
    directory = DATA_REVISION_A1_DIR / PINNED_RUN / context['dataset'] / f"seed_{context['seed']}"
    for name in ('X_train.npy', 'y_train.npy', 'feature_meta.json'):
        require(digest(directory / name) == context['artifact_hashes'][name], 'Forest train artifact hash mismatch')
    names = json.loads((directory / 'feature_meta.json').read_text())['feature_names']
    X = np.load(directory / 'X_train.npy', allow_pickle=False)
    y = np.load(directory / 'y_train.npy', allow_pickle=False)
    label = plan['training_class']
    require(label in (0, 1), 'Invalid revised Forest class')
    expected = pd.DataFrame(X[y == label], columns=names)
    require(frame_record(expected) == plan['input'], 'Forest must receive exactly its saved train class')
    require(plan['codec'] == ClassCodec.fit(expected, config['revision_a1']['datasets'][context['dataset']]['types']).record,
            'Forest codec differs from saved train class')
    library = library_record()
    wanted = {**library['defaults'], **config['model_params']['forestdiffusion']}
    for name in ('label_y', 'X_covs'):
        wanted.pop(name)
    wanted.update(plan['codec']['indices'], seed=derived_seed(context, label, 'fit')['seed'])
    require(plan['effective_parameters'] == wanted and plan['library'] == library,
            'Revised Forest effective parameters/library differ from identity')
    require(plan['backend'] == validation_backend_record(), 'Unexpected revised Forest backend')


def validate_real_scope(context, rows, label, effective):
    if context.get("validation_scope") == "revision_common_pilot_pima13_v1":
        require(context.get("run_id") == "33a9b4fce40753d94f4a3e2bf0c193f129e8edb5b6a19bb393544f747e2f3543" and
                context.get("dataset") == "diabetes_prima_indian" and context.get("seed") == 13 and
                rows == {0: 400, 1: 214}.get(label), "Outside full-train Pima common pilot")
        config = context["config"]
        require(config.get("technical_pilot", {}).get("scope") == "revision_common_pilot_pima13_v1" and
                config["experiment"]["n_samples_per_class"] == 20, "Explicit quota-20 common pilot required")
        wanted = {**library_record()["defaults"], **config["model_params"]["forestdiffusion"]}
        controlled = {"seed", "bin_indexes", "cat_indexes", "int_indexes", "label_y", "X_covs"}
        require(all(effective.get(k) == v for k, v in wanted.items() if k not in controlled),
                "Common Forest pilot parameters differ from frozen configuration")
        require(effective["n_t"] == 4 and effective["n_estimators"] == 5 and effective["duplicate_K"] == 100 and
                effective["n_jobs"] == 1 and effective["nthread"] == 1 and effective["max_depth"] == 7 and
                not effective["gpu_hist"], "Outside approved bounded CPU Forest pilot")
        return
    require(context.get("validation_scope") == VALIDATION_SCOPE, "Explicit real-validation scope required")
    allowed = {"cervical_class_0": (32, 0), "cervical_class_1": (44, 1),
               "technical_fixture": (12, 0), "all_constant": (3, 0)}
    require(allowed.get(context.get("validation_case")) == (rows, label), "Outside minimal validation case/size")
    require(all(effective.get(k) == v for k, v in VALIDATION_PARAMS.items()), "Outside minimal validation parameters")
    # Match all other API defaults; this path cannot become a general real backend.
    controlled = set(VALIDATION_PARAMS) | {"seed", "bin_indexes", "cat_indexes", "int_indexes", "label_y", "X_covs"}
    require(all(effective.get(k) == v for k, v in library_record()["defaults"].items() if k not in controlled),
            "Unapproved real-validation default override")


def library_record():
    """Read installed sources/defaults, without importing the training constructor."""
    dist = importlib.metadata.distribution("ForestDiffusion")
    root = Path(dist.locate_file("ForestDiffusion")).resolve()
    source = root / "diffusion_with_trees_class.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ForestDiffusionModel")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(default) for arg, default in
                zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}
    return {"name": "ForestDiffusion", "version": dist.version, "path": str(root),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py"))},
            "direct_url": dist.read_text("direct_url.json"), "defaults": defaults}


def derived_seed(context, label, purpose):
    material = {"protocol": PROTOCOL, "run_id": context["run_id"], "dataset": context["dataset"],
                "experimental_seed": int(context["seed"]), "class": int(label), "purpose": purpose}
    fingerprint = value_hash(material)
    return {"material": material, "sha256": fingerprint, "seed": int(fingerprint[:8], 16)}


def pack_rng(state):
    return {"algorithm": state[0], "keys": state[1].tolist(), "position": int(state[2]),
            "has_gauss": int(state[3]), "cached_gaussian": float(state[4])}


def unpack_rng(state):
    return (state["algorithm"], np.asarray(state["keys"], dtype=np.uint32), state["position"],
            state["has_gauss"], state["cached_gaussian"])


@contextmanager
def isolated_rng(state):
    """Cooperating adapter calls only; unrelated global NumPy users must be idle."""
    with _RNG_LOCK:
        outer = np.random.get_state()
        np.random.set_state(unpack_rng(state))
        try:
            yield
        finally:
            np.random.set_state(outer)


class ClassCodec:
    """Train-class-only ordinal mapping and constant extraction, no scaling here."""
    def __init__(self, record):
        self.record = record

    @classmethod
    def fit(cls, frame, types):
        require(len(frame) > 0 and frame.columns.is_unique, "Empty class or duplicate predictors")
        require(set(frame) <= set(types), "Predictors absent from explicit schema")
        require(all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes), "Nonnumeric prepared train")
        require(np.isfinite(frame.to_numpy()).all(), "Nonfinite prepared train")
        constants, categories, variable = {}, {}, []
        indices = {"bin_indexes": [], "cat_indexes": [], "int_indexes": []}
        fractional = {}
        for name in frame:
            kind = types[name]
            require(kind in {"binary", "categorical", "continuous", "count"}, "Unsupported schema type")
            values = np.unique(frame[name].to_numpy())
            if kind == "binary":
                require(np.isin(values, [0, 1]).all(), "Prepared binary outside 0/1")
            if kind == "count":
                fractional[name] = int((frame[name] != np.trunc(frame[name])).sum())
            if len(values) == 1:
                constants[name] = float(values[0])
                continue
            j = len(variable)
            variable.append(name)
            if kind == "categorical":
                categories[name] = [float(v) for v in values]
                indices["cat_indexes"].append(j)
            elif kind == "binary":
                indices["bin_indexes"].append(j)
            elif kind == "count":
                indices["int_indexes"].append(j)
        # 1.0.6 matches dummy prefixes by substring; fail explicitly for ambiguous indices.
        cats = indices["cat_indexes"]
        require(not any(str(a) + "_" in str(b) + "_" for a in cats for b in cats if a != b),
                "Installed categorical decoder has ambiguous index prefixes")
        flat = sum(indices.values(), [])
        require(len(set(flat)) == len(flat) and all(0 <= j < len(variable) for j in flat),
                "Overlapping or invalid library treatment indices")
        return cls({"columns": list(frame), "dtypes": {n: str(frame[n].dtype) for n in frame},
                    "types": {n: types[n] for n in frame}, "variable_columns": variable,
                    "constants": constants, "categories_ordinal_to_original": categories,
                    "indices": indices, "train_fractional_count_rows": fractional,
                    "all_constant": not variable,
                    "count_policy": "native int_indexes on nonconstant outputs only; input medians untouched; constants exact"})

    def encode(self, frame):
        r = self.record
        require(list(frame) == r["columns"], "Codec input column order mismatch")
        array = frame[r["variable_columns"]].to_numpy(dtype=np.float64, copy=True)
        for name, codes in r["categories_ordinal_to_original"].items():
            j = r["variable_columns"].index(name)
            require(np.isin(array[:, j], codes).all(), "Unknown training category")
            array[:, j] = np.searchsorted(codes, array[:, j])
        return array

    def decode(self, returned):
        r = self.record
        a = np.asarray(returned)
        require(a.ndim == 2 and a.shape[1] == len(r["variable_columns"]), "Library return shape/columns incompatible")
        require(np.issubdtype(a.dtype, np.number), "Library return nonnumeric")
        columns = {name: a[:, j].copy() for j, name in enumerate(r["variable_columns"])}
        for name, codes in r["categories_ordinal_to_original"].items():
            v = columns[name]
            # No rounding or invented mapping for an undecodable return. Preserve raw first.
            require(np.isfinite(v).all() and np.isin(v, np.arange(len(codes))).all(),
                    "Library categorical inverse contract violated")
            columns[name] = np.asarray(codes)[v.astype(int)]
        for name, value in r["constants"].items():
            columns[name] = np.full(len(a), value)
        frame = pd.DataFrame(columns)[r["columns"]]
        for name, dtype in r["dtypes"].items():
            if name in r["constants"]:
                frame[name] = frame[name].astype(dtype)
        return frame


def make_plan(frame, types, context, label, params, *, library=None, backend_factory=None, real_validation=False,
              revised_run=False):
    require(label in (0, 1), "Invalid training class")
    library = library or library_record()
    require(library["version"] == "1.0.6", "ForestDiffusion version requires protocol re-audit")
    codec = ClassCodec.fit(frame, types)
    X = codec.encode(frame)
    controlled = {"X", "X_covs", "label_y", "seed", "bin_indexes", "cat_indexes", "int_indexes",
                  "true_min_max_values"}
    require(not set(params) & controlled, "Parameters attempt to override class protocol")
    require(set(params) <= set(library["defaults"]) | {"nthread"}, "Unreviewed ForestDiffusion parameter")
    effective = {**library["defaults"], **params}
    for key in ("X_covs", "label_y"):
        effective.pop(key)
    require(effective["n_batch"] == 1 and effective["model"] == "xgboost" and
            effective["diffusion_type"] == "flow", "Only audited XGBoost flow/n_batch=1 supported")
    require(effective["n_t"] >= 2 and effective["n_estimators"] >= 1 and effective["duplicate_K"] >= 1,
            "Invalid ForestDiffusion size parameters")
    seeds = {p: derived_seed(context, label, p) for p in ("fit", "sample")}
    effective.update(codec.record["indices"], seed=seeds["fit"]["seed"])
    backend = None
    if backend_factory is not None:
        backend = backend_record(backend_factory)
    if real_validation:
        require(backend_factory is None, "Real validation cannot use a simulated factory")
        validate_real_scope(context, len(frame), label, effective)
        backend = validation_backend_record()
    require(not (revised_run and (real_validation or backend_factory is not None)), 'Mixed Forest execution modes')
    if revised_run:
        backend = validation_backend_record()
    kind = 'real_revised' if revised_run else ("real_validation" if real_validation else ("simulated" if backend else "blocked_real"))
    result = {"protocol": PROTOCOL, "execution_kind": kind,
            "real_validation": "pending", "context": context, "training_class": int(label),
            "input": frame_record(frame), "encoded_input": {"shape": list(X.shape), "dtype": X.dtype.str,
                       "sha256": hashlib.sha256(X.tobytes()).hexdigest()},
            "codec": codec.record, "library": library, "effective_parameters": effective,
            "native_encoded_train_min": X.min(axis=0).tolist(), "native_encoded_train_max": X.max(axis=0).tolist(),
            "native_transform": "class-only get_dummies(drop_first)/MinMaxScaler(-1,1); inverse/argmax; int+bin round then empirical clip",
            "native_pre_inverse_output_observable": False,
            "seeds": seeds, "seed_derivation": "SHA256 canonical JSON, first 32 bits unsigned",
            "sampling_policy": {"max_generate_calls_per_request": 1, "batch_size": "requested quota",
                                "excess": "first requested rows, audit remainder", "padding": False,
                                "domain_and_empirical_diagnostics": "observational"},
            "backend": backend, "wrapper_sha256": digest(Path(__file__)),
            "runtime": runtime_record(), "numpy_version": np.__version__, "pandas_version": pd.__version__}
    if revised_run:
        result['execution_scope'] = REVISED_SCOPE
        result['real_validation'] = 'prior minimal validation reused; benchmark quota/quality not demonstrated'
        validate_revised_scope(result)
    return result


class RevisedForestGenerator:
    def __init__(self, plan, model_dir, *, backend_factory=None, real_validation=False, revised_run=False):
        self.plan = plan
        self.directory = Path(model_dir).resolve()
        self.backend_factory = backend_factory
        self.real_validation = real_validation
        self.revised_run = revised_run
        require(not (revised_run and (real_validation or backend_factory is not None)), 'Mixed Forest execution modes')
        require(self.directory.name == f"class_{plan['training_class']}", "Wrong class model path")
        require(self.directory.parent.name == value_hash(plan), "Wrong model identity path")
        self.codec = ClassCodec(plan["codec"])
        self.sampling_observations = None

    def _authorize(self):
        if self.revised_run:
            validate_revised_scope(self.plan)
        elif self.real_validation:
            require(self.backend_factory is None and self.plan["execution_kind"] == "real_validation",
                    "Real validation must be explicitly identified")
            validate_real_scope(self.plan["context"], self.plan["encoded_input"]["shape"][0],
                                self.plan["training_class"], self.plan["effective_parameters"])
        else:
            require(self.backend_factory is not None and self.plan["execution_kind"] == "simulated", BLOCK_REASON)

    def fit(self, frame):
        self._authorize()
        # Recompute the complete identity before constructor use; stale plans cannot train.
        expected = make_plan(frame, self.plan["codec"]["types"], self.plan["context"],
                             self.plan["training_class"], self._original_params(),
                             backend_factory=self.backend_factory, real_validation=self.real_validation,
                             revised_run=self.revised_run)
        require(expected == self.plan, "Stale class plan/cache identity")
        require(not self.plan["codec"]["all_constant"], "All-constant class: no ordinary ForestDiffusion fit; no fallback")
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(self.directory / "plan.json", self.plan)
        X = self.codec.encode(frame)
        np.save(self.directory / "constructor_input.npy", X, allow_pickle=False)
        write_json(self.directory / "constructor_arguments.json", {"positional": ["X"],
                   "keyword_arguments": self.plan["effective_parameters"], "input": self.plan["encoded_input"],
                   "execution_kind": self.plan["execution_kind"]})
        state = pack_rng(np.random.RandomState(self.plan["seeds"]["fit"]["seed"]).get_state())
        started = time.perf_counter()
        fit_record = {"execution_kind": self.plan["execution_kind"], "status": "started", "constructor_calls": 0}
        write_json(self.directory / "fit_started.json", fit_record)
        caught = []
        try:
            with isolated_rng(state), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if self.real_validation or self.revised_run:
                    from ForestDiffusion import ForestDiffusionModel
                    fit_record["constructor_calls"] = 1
                    model = ForestDiffusionModel(X.copy(), **self.plan["effective_parameters"])
                else:
                    fit_record["constructor_calls"] = 1
                    model = self.backend_factory(X.copy(), **self.plan["effective_parameters"])
            fit_record["status"] = "completed"
        except Exception as exc:
            fit_record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            fit_record.update(seconds=time.perf_counter()-started, warnings=[str(w.message) for w in caught])
            write_json(self.directory / "fit_execution.json", fit_record)
        model_path = self.directory / "model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(model, stream)
        np.save(self.directory / "reference_train.npy", frame.to_numpy(), allow_pickle=False)
        write_json(self.directory / "metadata.json", {"identity_sha256": value_hash(self.plan),
                   "training_class": self.plan["training_class"], "model_sha256": digest(model_path),
                   "reference_train_sha256": digest(self.directory / "reference_train.npy"),
                   "execution_kind": self.plan["execution_kind"], "status": self.plan["execution_kind"] + "_fit"})
        write_json(self.directory / "sampling_state.json", {"identity_sha256": value_hash(self.plan), "calls": 0,
                   "rng": pack_rng(np.random.RandomState(self.plan["seeds"]["sample"]["seed"]).get_state())})

    def _original_params(self):
        # Reconstruct overrides, omitting values controlled by codec/seed policy.
        controlled = {"seed", "bin_indexes", "cat_indexes", "int_indexes", "true_min_max_values"}
        return {k: v for k, v in self.plan["effective_parameters"].items() if k not in controlled and
                (k not in self.plan["library"]["defaults"] or v != self.plan["library"]["defaults"][k])}

    def _check(self, requested_class):
        require(requested_class == self.plan["training_class"], "Requested class differs from training class")
        require(self.directory.name == f"class_{requested_class}", "Wrong class model path")
        saved = json.loads((self.directory / "plan.json").read_text())
        metadata = json.loads((self.directory / "metadata.json").read_text())
        require(saved == self.plan and metadata["identity_sha256"] == value_hash(self.plan), "Class model cache identity mismatch")
        require(metadata["training_class"] == requested_class and metadata["execution_kind"] == self.plan["execution_kind"],
                "Model metadata class/execution mismatch")
        require(self.plan["wrapper_sha256"] == digest(Path(__file__)) and self.plan["library"] == library_record(),
                "Changed library/wrapper invalidates cache")
        current_backend = validation_backend_record() if self.real_validation or self.revised_run else backend_record(self.backend_factory)
        require(self.plan["runtime"] == runtime_record() and self.plan["backend"] == current_backend,
                "Changed runtime/backend invalidates cache")
        from config.const import ROOT_DIR
        require(all(digest(ROOT_DIR / name) == sha for name, sha in
                    self.plan["context"].get("code_sha256", {}).items()), "Changed pipeline code invalidates cache")
        require(metadata["model_sha256"] == digest(self.directory / "model.pkl"), "Model bytes mismatch")
        require(metadata["reference_train_sha256"] == digest(self.directory / "reference_train.npy"), "Reference train bytes mismatch")
        return metadata

    def sample(self, n, *, requested_class):
        self._authorize()
        require(isinstance(n, int) and n > 0, "Invalid quota")
        if self.real_validation:
            expected_n = 20 if self.plan['context'].get('validation_scope') == 'revision_common_pilot_pima13_v1' else 8
            require(n == expected_n, "Real validation quota differs from authorized scope")
        if self.revised_run:
            require(n == self.plan['context']['config']['experiment']['n_samples_per_class'],
                    'Revised Forest quota differs from identity')
        self._check(requested_class)  # Reject routing/metadata BEFORE deserialize/generate.
        # File lock serializes continuation of one class across cooperating processes.
        with (self.directory / "sampling.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_path = self.directory / "sampling_state.json"
            state = json.loads(state_path.read_text())
            require(state["identity_sha256"] == value_hash(self.plan), "Sampling state identity mismatch")
            call = state["calls"] + 1
            audit = self.directory / f"sample_{call:04d}"
            audit.mkdir(exist_ok=False)
            started = time.perf_counter()
            observation = {"execution_kind": self.plan["execution_kind"], "requested_rows": n, "requested_class": requested_class,
                           "generate_calls": 1, "produced_rows": None, "selected_rows": 0,
                           "excess_discarded_rows": 0, "delivered_rows": 0, "deficit_rows": None,
                           "padding_rows": 0, "status": "started", "warnings": [],
                           "rng_before_sha256": value_hash(state["rng"])}
            self.sampling_observations = observation
            caught = []
            try:
                with (self.directory / "model.pkl").open("rb") as stream:
                    model = pickle.load(stream)
                if self.real_validation:
                    write_json(audit / "rng_before.json", state["rng"])
                    observation["in_memory_model_before_sha256"] = hashlib.sha256(pickle.dumps(model)).hexdigest()
                with isolated_rng(state["rng"]):
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            returned = model.generate(batch_size=n)
                    finally:
                        state["rng"] = pack_rng(np.random.get_state())
                if self.real_validation:
                    with (audit / "model_after_generate.pkl").open("xb") as stream:
                        pickle.dump(model, stream)
                    observation["in_memory_model_after_sha256"] = digest(audit / "model_after_generate.pkl")
                raw = np.asarray(returned)
                produced = len(raw) if raw.ndim else None
                observation.update(produced_rows=produced,
                                   deficit_rows=max(0, n-produced) if produced is not None else None,
                                   return_shape=list(raw.shape), return_dtype=str(raw.dtype),
                                   nonfinite_values=int((~np.isfinite(raw)).sum()) if
                                   np.issubdtype(raw.dtype, np.number) else None)
                # Preserve the exact observable API return, including invalid numeric values.
                np.save(audit / "library_return.npy", raw, allow_pickle=False)
                observation["library_return_sha256"] = digest(audit / "library_return.npy")
                decoded = self.codec.decode(raw)
                np.save(audit / "decoded_return.npy", decoded.to_numpy(), allow_pickle=False)
                decoded.to_csv(audit / "decoded_return.csv", index=False)
                from src.revision_validity import diagnose, training_reference
                reference_frame = pd.DataFrame(np.load(self.directory / "reference_train.npy", allow_pickle=False),
                                               columns=self.plan["codec"]["columns"])
                reference = training_reference(reference_frame.to_numpy(), list(reference_frame),
                            {"types": self.plan["codec"]["types"], "categorical_domains": {}},
                            {k: v for k, v in self.plan["codec"]["categories_ordinal_to_original"].items()})
                diagnostic = diagnose(decoded, reference_frame.to_numpy(), list(reference_frame), reference,
                                      requested=n, stage="native_return_after_reversible_decode", delivered=0)
                write_json(audit / "diagnostic.json", diagnostic)
                selected = decoded.iloc[:n].copy()
                observation.update(selected_rows=len(selected), excess_discarded_rows=max(0, len(decoded)-n),
                                   deficit_rows=max(0, n-len(selected)))
                require(np.isfinite(decoded.to_numpy()).all(), "Nonfinite library return; entire request withheld")
                require(len(selected) == n, "ForestDiffusion sample deficit; no padding or retries")
                observation.update(status=self.plan["execution_kind"] + "_delivered", delivered_rows=n)
                return selected
            except Exception as exc:
                observation.update(status="failed", withheld_rows_due_to_error=observation["produced_rows"],
                                   error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                observation["warnings"] = [str(w.message) for w in caught]
                observation["seconds"] = time.perf_counter()-started
                state["calls"] = call
                observation["rng_after_sha256"] = value_hash(state["rng"])
                if self.real_validation:
                    write_json(audit / "rng_after.json", state["rng"])
                write_json(audit / "execution.json", observation)
                temporary = self.directory / "sampling_state.next.json"
                temporary.write_bytes(encode(state))
                temporary.replace(state_path)
                fcntl.flock(lock, fcntl.LOCK_UN)


def sample_forest_split(run, split, *, backend_factory=None, base=None):
    """Specialized revised sampler; no entry into the legacy full-table wrapper."""
    if backend_factory is None:
        require(base is None, 'Real revised Forest cannot use a simulation output root')
        return _sample_revised_split(run, split)
    from src.revision_a1_runtime import generator_inputs
    from src.revision_validity import diagnose, training_reference
    context = run.identity(split, "forestdiffusion")
    require(base is not None, "Simulation requires an explicit isolated output root")
    root = Path(base) / split.dataset / f"seed_{split.seed}"
    require(not root.exists(), "Simulated ForestDiffusion output already exists; no overwrite/reuse")
    inputs = generator_inputs(split, "forestdiffusion", diagnostic=True)
    plans = [(label, frame, make_plan(frame, run.policy["datasets"][split.dataset]["types"], context, label,
              run.config["model_params"]["forestdiffusion"], backend_factory=backend_factory)) for label, frame in inputs]
    root.mkdir(parents=True, exist_ok=False)
    n = run.config["experiment"]["n_samples_per_class"]
    reference = training_reference(split.X_train, split.names, run.policy["datasets"][split.dataset],
                                   split.preprocessing.train_categories)
    write_json(root / "identity.json", {"execution_kind": "simulated", "context": context, "quota": n,
                                       "real_validation": "pending"})
    records = []
    for label, frame, plan in plans:
        model_dir = root / "models" / value_hash(plan) / f"class_{label}"
        generator = RevisedForestGenerator(plan, model_dir, backend_factory=backend_factory)
        record = {"training_class": label, "identity": value_hash(plan), "directory": str(model_dir),
                  "status": "started", "execution_kind": "simulated"}
        records.append(record)
        try:
            generator.fit(frame)
            # Reload plan/codec/model rather than relying on a live fitted instance.
            generator = RevisedForestGenerator(json.loads((model_dir / "plan.json").read_text()), model_dir,
                                               backend_factory=backend_factory)
            generated = generator.sample(n, requested_class=label)
            diagnostic = diagnose(generated, split.X_train, split.names, reference, requested=n,
                                   stage="simulated_revised_sampler", delivered=n,
                                   internal_observation=generator.sampling_observations)
            write_json(root / f"validity_class_{label}.json", diagnostic)
            generated["class"] = np.full(n, label, dtype=np.int32)
            path = root / f"simulated_class_{label}.csv"
            generated.to_csv(path, index=False, mode="x")
            record.update(status="simulated_delivered", rows=n, sha256=digest(path),
                          observations=generator.sampling_observations)
        except Exception as exc:
            record.update(status="failed", delivered_rows=0, error={"type": type(exc).__name__, "message": str(exc)})
            write_json(root / "execution.json", {"execution_kind": "simulated", "status": "failed", "classes": records})
            raise
    result = {"execution_kind": "simulated", "status": "simulated_completed", "real_validation": "pending", "classes": records}
    write_json(root / "execution.json", result)
    return result


def revised_plans(run, split):
    from src.revision_a1_runtime import generator_inputs
    context = run.identity(split, 'forestdiffusion')
    result = [(label, frame, make_plan(frame, run.policy['datasets'][split.dataset]['types'], context,
               label, run.config['model_params']['forestdiffusion'], revised_run=True))
              for label, frame in generator_inputs(split, 'forestdiffusion')]
    require(all(not p['codec']['all_constant'] for _, _, p in result),
            'All-constant class: no ordinary ForestDiffusion fit; no fallback')
    return result


def _sample_revised_split(run, split):
    from src.revision_a1_runtime import validate_synthetic
    from src.revision_validity import diagnose, training_reference
    identity, models, output = run.paths(split, 'forestdiffusion')
    require(not models.exists() and not output.exists(), 'A1 never reuses/overwrites existing models or samples')
    plans = revised_plans(run, split)  # Both codecs/scopes checked before any constructor or output.
    models.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'identity.json', identity)
    write_json(output / 'generator_parameters.json', identity['generator_parameters'])
    reference = training_reference(split.X_train, split.names, run.policy['datasets'][split.dataset],
                                   split.preprocessing.train_categories)
    write_json(output / 'validity_reference.json', reference)
    records, files, class_models = [], {}, []
    quota = run.config['experiment']['n_samples_per_class']
    for label, frame, plan in plans:
        directory = models / value_hash(plan) / f'class_{label}'
        record = {'label': label, 'fit': frame_record(frame), 'status': 'fit_started'}
        records.append(record)
        _update_execution_record(output / 'forest_execution.json', {'status': 'started', 'classes': records})
        try:
            adapter = RevisedForestGenerator(plan, directory, revised_run=True)
            with generation_phase(run, split, 'forestdiffusion', label, 'fit', output, input_rows=len(frame)):
                adapter.fit(frame)
            with generation_phase(run, split, 'forestdiffusion', label, 'sampling', output,
                                  requested_class=label, requested_per_class=quota) as timing:
                adapter = RevisedForestGenerator(json.loads((directory / 'plan.json').read_text()), directory, revised_run=True)
                returned = adapter.sample(quota, requested_class=label)
                timing['returned_rows'] = len(returned)
            returned.to_csv(output / f'raw_return_class_{label}.csv', index=False, mode='x')
            diagnostic = diagnose(returned, split.X_train, split.names, reference, requested=quota,
                stage='revised_forest_return_after_codec', delivered=None, internal_observation=adapter.sampling_observations)
            _update_execution_record(output / f'validity_class_{label}.json', diagnostic)
            generated = validate_synthetic(returned, split.names, quota)
            diagnostic.update(delivered_rows=quota, status='delivered_without_padding')
            _update_execution_record(output / f'validity_class_{label}.json', diagnostic)
            generated['class'] = np.full(quota, label, dtype=np.int32)
            path = output / f'synthetic_class_{label}.csv'
            generated.to_csv(path, index=False, mode='x')
            files[path.name] = digest(path)
            class_models.append({'label': label, 'relative_directory': str(directory.relative_to(models)),
                'plan_sha256': digest(directory / 'plan.json'), 'metadata_sha256': digest(directory / 'metadata.json')})
            record.update(status='delivered', delivered_rows=quota, observations=adapter.sampling_observations)
        except Exception as exc:
            record.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
            _update_execution_record(output / 'forest_execution.json', {'status': 'failed', 'classes': records})
            raise
    _update_execution_record(output / 'forest_execution.json', {'status': 'completed', 'classes': records})
    write_json(output / 'samples_manifest.json', {'identity': identity, 'inputs': records, 'files': files,
        'forest_models': class_models, 'postprocessing': 'approved train-class codec inverse; labels from model identity; no padding',
        'observability_files': {p.name: digest(p) for p in output.iterdir()
            if p.name.startswith(('validity_', 'raw_return_', 'forest_execution'))}})


def check_revised_delivery(run, split, manifest):
    """General reader verifies both class models without deserializing or sampling."""
    _, models, _ = run.paths(split, 'forestdiffusion')
    entries = manifest.get('forest_models', [])
    require([entry['label'] for entry in entries] == [0, 1], 'Forest delivery needs exactly both class models')
    for (label, _, plan), entry in zip(revised_plans(run, split), entries):
        expected_relative = Path(value_hash(plan)) / f'class_{label}'
        require(entry['relative_directory'] == str(expected_relative), 'Forest class model routing mismatch')
        directory = models / expected_relative
        require(digest(directory / 'plan.json') == entry['plan_sha256'] and
                digest(directory / 'metadata.json') == entry['metadata_sha256'], 'Forest delivery model hash mismatch')
        adapter = RevisedForestGenerator(plan, directory, revised_run=True)
        adapter._authorize()
        adapter._check(label)


def _update_execution_record(path, value):
    """Atomic progress updates inside a new run; immutable artifacts still use xb."""
    temporary = path.with_suffix(path.suffix + '.next')
    with temporary.open('xb') as stream:
        stream.write(encode(value))
    temporary.replace(path)
