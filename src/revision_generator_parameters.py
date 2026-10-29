"""Unambiguous parameters for the nine revised generators; no model imports.

Legacy config aliases are migrated explicitly and recorded. The historical
factory/wrappers keep their old parameter semantics unless this marker is present.
"""
from __future__ import annotations

import copy
import math

PROTOCOL = "revision_generator_parameters_v1"
MARKER = "_revision_parameters"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def renamed(params, old, new, default=None):
    require(not (old in params and new in params), f"Ambiguous parameters: use only {new}, never both {old} and {new}")
    if old in params:
        params[new] = params.pop(old)
        return {"from": old, "to": new, "value": params[new], "behavior_changed": False}
    if new not in params and default is not None:
        params[new] = default
    require(new in params, f"Missing explicit parameter {new}")
    return None


def parameter_plan(model, configured, experimental_seed):
    require(model != "forestdiffusion", "ForestDiffusion is outside this parameter policy")
    require(type(experimental_seed) is int and experimental_seed >= 0, "Invalid experimental seed")
    require(MARKER not in configured, "Reserved revised parameter marker")
    before, params = copy.deepcopy(configured), copy.deepcopy(configured)
    aliases, units, randomness = [], {}, {"experimental_seed": experimental_seed,
        "global_seed_setup": "CLI set_global_seed before class fits; Python/NumPy/torch/CUDA and configured backend flags",
        "determinism_verified": False}
    if model == "tabula":
        require(params.get('sampling_strategy', 'native') in ('native', 'schema_constrained'),
                'Unknown TabuLa sampling_strategy')
        alias = renamed(params, "top_k", "sampling_batch_size", default=100)
        if alias:
            aliases.append(alias)
        require(type(params["sampling_batch_size"]) is int and params["sampling_batch_size"] > 0,
                "sampling_batch_size must be a positive integer")
        units["sampling_batch_size"] = "candidate rows per library iteration; not token top-k"
        if params.get('sampling_strategy') == 'schema_constrained':
            units['sampling_strategy'] = ('native classwise training and start feature; remaining predictor names '
                                         'forced in prepared order; numeric values sampled; no padding or imputation')
    elif model == "realtabformer":
        if "random_state" in params:
            require(type(params["random_state"]) is int and params["random_state"] == experimental_seed,
                    "REaLTabFormer random_state contradicts experimental seed")
        params["random_state"] = experimental_seed
        randomness.update(library_random_state=experimental_seed,
                          previous_library_random_state_default=1029,
                          trainer_seed="not overridden; installed TrainingArguments default 42",
                          data_seed="not overridden; installed TrainingArguments default None",
                          initialization="GPT2Config(n_layer=6)/GPT2LMHeadModel(config), unchanged")
    elif model == "mostlyai":
        alias = renamed(params, "max_training_time", "max_training_time_minutes")
        if alias:
            aliases.append(alias)
        value = params["max_training_time_minutes"]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                "max_training_time_minutes must be positive; API maps zero to its default")
        units["max_training_time_minutes"] = {"library_argument": "max_training_time", "unit": "minutes",
                                               "library_internal_seconds": value*60}
        randomness.update(library_seed_argument=None,
                          limitation="installed split/analyze/encode/train/generate expose no RNG seed; seed_data is conditioning data")
    params[MARKER] = {"protocol": PROTOCOL, "model": model, "experimental_seed": experimental_seed}
    if model == "great":
        from src.revision_great import correction_record
        params[MARKER]["great_correction"] = correction_record()
    if model == "tabpfgen":
        from src.revision_tabpfgen import policy_record
        params[MARKER]["completion_policy"] = policy_record()
    return {"protocol": PROTOCOL, "model": model, "configured": before, "effective_wrapper_parameters": params,
            "alias_migrations": aliases, "units": units, "randomness": randomness,
            "sampling_budgets": "unchanged; definitive limits require protocol decision"}


def revised_context(params, model):
    """Validate wrapper activation; return None for the untouched historical path."""
    if MARKER not in params:
        return None
    context = params[MARKER]
    require(context.get("protocol") == PROTOCOL and context.get("model") == model,
            "Invalid revised generator parameter context")
    require(type(context.get("experimental_seed")) is int, "Missing experimental seed")
    if model == "tabula":
        require("top_k" not in params and type(params.get("sampling_batch_size")) is int and
                params["sampling_batch_size"] > 0, "Ambiguous/invalid revised TabuLa batch size")
    elif model == "realtabformer":
        require(type(params.get("random_state")) is int and params["random_state"] == context["experimental_seed"],
                "REaLTabFormer random_state contradicts experimental seed")
    elif model == "mostlyai":
        require("max_training_time" not in params and "max_training_time_minutes" in params,
                "Ambiguous revised MOSTLY training-time units")
        value = params["max_training_time_minutes"]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                "max_training_time_minutes must be positive; API maps zero to its default")
    return context
