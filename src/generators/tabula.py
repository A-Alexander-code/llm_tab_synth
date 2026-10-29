"""TabuLa generator wrapper."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from src.generators.base import BaseGenerator

_TABULA_REPO = "https://github.com/zhao-zilong/Tabula"
_TABULA_DIR  = Path(__file__).resolve().parent.parent.joinpath("models").joinpath("tabula")


def _ensure_tabula():
    if not _TABULA_DIR.exists():
        subprocess.run(
            ["git", "clone", _TABULA_REPO, str(_TABULA_DIR)],
            check=True,
        )
    if str(_TABULA_DIR) not in sys.path:
        sys.path.insert(0, str(_TABULA_DIR))


def _save_tabula(model, model_dir: Path) -> None:
    """Save a TabuLa model with JSON-safe metadata."""
    import numpy as np

    model_dir.mkdir(parents=True, exist_ok=True)

    attributes = model.__dict__.copy()
    for key in ["tokenizer", "model", "config"]:
        attributes.pop(key, None)

    if isinstance(attributes.get("conditional_col_dist"), np.ndarray):
        attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

    safe_attributes = {}
    for k, v in attributes.items():
        try:
            json.dumps(v)
            safe_attributes[k] = v
        except (TypeError, ValueError):
            pass

    with model_dir.joinpath("config.json").open("w", encoding="utf-8") as f:
        json.dump(safe_attributes, f)

    torch.save(model.model.state_dict(), str(model_dir.joinpath("model.pt")))


def _load_tabula(model_dir: Path, llm: str):
    """Load a saved TabuLa model."""
    _ensure_tabula()
    from tabula.tabula import Tabula  # noqa: PLC0415

    config_path = model_dir.joinpath("config.json")
    model_pt    = model_dir.joinpath("model.pt")

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found in {model_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = Tabula(llm=llm)

    for key, value in config.items():
        try:
            setattr(model, key, value)
        except AttributeError:
            pass

    buffer     = io.BytesIO(model_pt.read_bytes())
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state_dict)

    model.model.config.pad_token_id = model.tokenizer.eos_token_id

    return model


def _compute_max_length(df: pd.DataFrame, margin: int = 20) -> int:
    """Estimate a sampling length from the dataset rows."""
    _ensure_tabula()
    from tabula.tabula import Tabula  # noqa: PLC0415

    tokenizer = Tabula(llm="distilgpt2").tokenizer
    sample    = df.head(min(10, len(df)))
    lengths   = []

    for _, row in sample.iterrows():
        text   = ", ".join(f"{col} is {val}" for col, val in row.items())
        tokens = tokenizer(text, return_tensors="pt")
        lengths.append(tokens["input_ids"].shape[1])

    estimated = max(lengths) + margin
    return max(80, min(estimated, 300))


def _revised_sampling_config(df: pd.DataFrame, model, margin: int = 20) -> dict:
    """Measure complete class-train rows with the native `column value` format.

    Reuse the model's tokenizer/configuration; do not construct another model,
    sample permutations, fit transformations or access test data.
    """
    if df.empty or len(df.columns) == 0:
        raise ValueError('TabuLa length estimation requires nonempty class train input')
    if type(margin) is not int or margin < 0:
        raise ValueError('TabuLa length margin must be a nonnegative integer')
    limits = {}
    config = model.model.config
    for name in ('max_position_embeddings', 'n_positions', 'n_ctx'):
        value = getattr(config, name, None)
        if type(value) is int and value > 0:
            limits[name] = value
    tokenizer_limit = getattr(model.tokenizer, 'model_max_length', None)
    if type(tokenizer_limit) is int and 0 < tokenizer_limit < sys.maxsize:
        limits['tokenizer_model_max_length'] = tokenizer_limit
    if not limits:
        raise ValueError('TabuLa model/tokenizer does not expose a finite context window')
    window = min(limits.values())
    lengths = []
    texts = []
    for row in df.itertuples(index=False, name=None):
        texts.append(', '.join(f'{column} {str(value).strip()}' for column, value in zip(df.columns, row)))
        if len(texts) == 128:
            lengths.extend(len(ids) for ids in model.tokenizer(texts)['input_ids'])
            texts = []
    if texts:
        lengths.extend(len(ids) for ids in model.tokenizer(texts)['input_ids'])
    if len(lengths) != len(df) or not all(length > 0 for length in lengths):
        raise ValueError('TabuLa tokenizer returned incompatible train row lengths')
    longest = max(lengths)
    if longest >= window:
        raise ValueError(f'TabuLa train row needs {longest} tokens; context window {window} leaves no completion room')
    requested = max(80, longest+margin)
    return {'estimated_max_length': min(requested, window),
            'source': 'complete class train input only',
            'serialization': 'column value; comma-space separator',
            'column_order': 'prepared train order; native training permutations unchanged',
            'train_rows_measured': len(lengths),
            'train_token_lengths': {'min': min(lengths), 'max': longest},
            'margin_tokens': margin, 'minimum_length': 80,
            'context_window': window, 'context_limits': limits,
            'requested_max_length': requested, 'context_limited': requested > window}


class TabulaGenerator(BaseGenerator):

    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        from src.revision_generator_parameters import revised_context
        revised = revised_context(self.params, "tabula")
        if revised and self.params.get('sampling_strategy') == 'schema_constrained':
            from src.revision_tabula_schema import validate_schema
            validate_schema(list(df.columns), list(df.select_dtypes(include='number').columns))
        import warnings  # noqa: PLC0415
        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")
        warnings.filterwarnings("ignore", message=".*loss_type=None.*")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        _ensure_tabula()
        from tabula.tabula import Tabula  # noqa: PLC0415

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model = Tabula(
            llm=self.params["llm"],
            experiment_dir=str(model_dir.joinpath("training")),
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
        )
        # Reject an incompatible context before starting an expensive fit.
        sampling_config = _revised_sampling_config(df, model) if revised else None
        model.fit(df)
        _save_tabula(model, model_dir)

        if sampling_config is None:
            sampling_config = {"estimated_max_length": _compute_max_length(df)}
        with model_dir.joinpath("sampling_config.json").open("w") as f:
            json.dump(sampling_config, f)

    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        from src.revision_generator_parameters import revised_context
        revised = revised_context(self.params, "tabula")
        if revised:
            strategy = self.params.get('sampling_strategy', 'native')
            if strategy not in ('native', 'schema_constrained'):
                raise ValueError('Unknown revised TabuLa sampling_strategy')
            if 'sampling_limits' in self.params and 'pilot_sampling_limits' in self.params:
                raise ValueError('Ambiguous TabuLa limits')
            limits = self.params.get('sampling_limits', self.params.get('pilot_sampling_limits'))
            if limits is not None:
                from src.revision_tabula_budget import validate_limits
                validate_limits(limits)
        import warnings  # noqa: PLC0415
        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*kernel version.*")
        warnings.filterwarnings("ignore", message=".*GPT2TokenizerFast.*")

        cuda_device = self.params.get("cuda_device", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            device = f"cuda:{cuda_device}" if cuda_device.isdigit() else "cuda"
        else:
            device = "cpu"

        sampling_config_path = model_dir.joinpath("sampling_config.json")
        sampling_config = None
        if sampling_config_path.exists():
            with sampling_config_path.open("r") as f:
                sampling_config = json.load(f)
            max_length = sampling_config.get(
                "estimated_max_length",
                self.params.get("max_length_sample", 200)
            )
        else:
            max_length = self.params.get("max_length_sample", 200)

        model = _load_tabula(model_dir, llm=self.params["llm"])
        kwargs = dict(device=device, max_length=max_length,
                      temperature=self.params.get("temperature", 0.7),
                      k=self.params["sampling_batch_size"] if revised else self.params.get("top_k", 100))
        if not revised:
            return model.sample(n_samples, **kwargs)
        generation_config = getattr(getattr(model, "model", None), "generation_config", None)
        self.sampling_observations = {
            "execution_kind": "revision_a1", "library_sample_arguments": {"n_samples": n_samples, **kwargs},
            "length_source": "train-derived sampling_config.json" if sampling_config_path.exists() else "config fallback",
            "length_estimation": sampling_config,
            "library_generate_overrides": {"do_sample": True, "temperature": kwargs["temperature"],
                                           "max_length": max_length, "pad_token_id": 50256},
            "inherited_generation_config": (json.loads(generation_config.to_json_string(use_diff=False))
                                             if generation_config is not None else None),
            "k_semantics": "candidate batch size; does not override token top_k",
            "internal_generate_calls_observed": None, "internal_candidates_observed": None,
            "internal_discarded_rows_observed": None, "status": "started",
            "limits_policy": "unchanged; bounded-loop proposal pending decision",
        }
        if revised and limits is not None:
            self.sampling_observations['limits_policy'] = 'explicit call/candidate/no-progress bounds; native batch semantics unchanged'
        audit_path = model_dir / "revision_sampling_parameters.json"
        audit_path.write_text(json.dumps(self.sampling_observations, indent=2) + "\n")
        try:
            from contextlib import nullcontext
            from src.revision_tabula_budget import attention_masked_sampling, bounded_sample
            schema_context = nullcontext()
            if strategy == 'schema_constrained':
                if limits is None:
                    raise ValueError('Schema-constrained sampling requires explicit generation limits')
                from src.revision_tabula_schema import schema_constrained_sampling
                schema_context = schema_constrained_sampling(model, self.sampling_observations, audit_path)
            with schema_context, attention_masked_sampling(model, self.sampling_observations):
                if limits is not None:
                    result = bounded_sample(model, n_samples, kwargs, limits,
                                            self.sampling_observations, audit_path,
                                            scope='revised' if 'sampling_limits' in self.params else 'pilot')
                else:
                    result = model.sample(n_samples, **kwargs)
            self.sampling_observations.update(status="returned", wrapper_return_rows=len(result))
            return result
        except Exception as exc:
            self.sampling_observations.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            audit_path.write_text(json.dumps(self.sampling_observations, indent=2) + "\n")
