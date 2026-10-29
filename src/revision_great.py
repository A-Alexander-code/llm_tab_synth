"""Versioned, instance-local GReaT guided parser; only the revised wrapper binds it.

No dependency edits or model construction. The public upstream sample dispatcher
is retained, while its guided method is replaced on one restored instance only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import shutil
import types

VERSION = "great-guided-parser-a1/1.0.0"
FIELD_SEPARATOR = ", "  # be_great.great_dataset.GReaTDataset._getitem serialization.
UPSTREAM_SHA256 = "43b63b02ba98e9c233c5dd501f7b8775321ae035315911fc268097b20cfda0a3"
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
IDENTITY_FILE = "revision_great_identity.json"
ROW_ATTEMPT_FACTOR = 2
MAX_CONSECUTIVE_REJECTIONS = 10
PARSING_REJECTIONS = frozenset({"unclosed_numeric_field", "numeric_grammar", "nonfinite_literal",
                                "numeric_overflow", "numeric_underflow"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


class GreatCorrectionError(ValueError):
    def __init__(self, reason, message, evidence=None):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.evidence = evidence or {}


def correction_record():
    dist = importlib.metadata.distribution("be-great")
    path = Path(dist.locate_file("be_great/great.py")).resolve()
    if dist.version != "0.0.13" or sha(path) != UPSTREAM_SHA256:
        raise GreatCorrectionError("upstream_mismatch", "GReaT source/version requires re-audit")
    return {"version": VERSION, "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": sha(__file__), "upstream_version": dist.version,
            "upstream_path": str(path), "upstream_sha256": sha(path),
            "wrapper_sha256": sha(Path(__file__).parent / "generators/great.py"),
            "dependency_modified": False, "binding": "instance-local _guided_sample; upstream sample dispatcher",
            "numeric_grammar": NUMBER.pattern, "closure_required": True,
            "generation_policy": "guided unchanged; complete-delimiter stopping; native EOS; existing single fallback"}


def save_model_identity(model_dir, context):
    record = correction_record()
    if context.get("great_correction") != record:
        raise GreatCorrectionError("correction_identity_mismatch", "Parameter correction identity differs")
    write(Path(model_dir) / IDENTITY_FILE, {"correction": record, "parameter_context": context})


def check_parameter_identity(context):
    if context.get("great_correction") != correction_record():
        raise GreatCorrectionError("correction_identity_mismatch", "Stale correction parameters; construction prohibited")


def check_model_identity(model_dir, context):
    path = Path(model_dir) / IDENTITY_FILE
    if not path.exists():
        raise GreatCorrectionError("missing_correction_identity", "Historical/unidentified artifact cannot be used in revised mode")
    current = correction_record()
    expected = {"correction": current, "parameter_context": context}
    if context.get("great_correction") != current or json.loads(path.read_text()) != expected:
        raise GreatCorrectionError("correction_identity_mismatch", "Version/hash/context incompatible; no artifact reuse")


def ids(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return value


def eos_ids(model):
    config = getattr(getattr(model, "model", None), "generation_config", None)
    value = getattr(config, "eos_token_id", None)
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value)]


def output_sequences(output):
    if hasattr(output, "sequences"):
        value = output.sequences
    elif isinstance(output, dict) and "sequences" in output:
        value = output["sequences"]
    else:
        value = output
    rows = ids(value)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise GreatCorrectionError("output_structure", "Expected one decoder-only sequence", {"sequences": rows})
    return [int(t) for t in rows[0]]


def field_delimiter(text, known_columns=()):
    """Find the first canonical separator or compact header of a known column.

    A bare comma is insufficient: the compact form must contain the literal
    column name and the complete ` is` marker. Escape names with punctuation.
    """
    position, next_column = text.find(FIELD_SEPARATOR), None
    for column in known_columns:
        match = re.search(',' + re.escape(column) + r' is(?=\s|$)', text)
        if match is not None and (position < 0 or match.start() < position):
            position, next_column = match.start(), column
    return position, next_column


def continuation(output, input_ids, tokenizer, *, active_eos, max_length, attention_mask=None,
                 max_new_tokens=None, known_columns=()):
    prompt_rows = ids(input_ids)
    if len(prompt_rows) != 1:
        raise GreatCorrectionError("input_structure", "Guided field requires one input row")
    prompt = [int(t) for t in prompt_rows[0]]
    sequence = output_sequences(output)
    if len(sequence) < len(prompt) or sequence[:len(prompt)] != prompt:
        raise GreatCorrectionError("output_prefix_mismatch", "Returned sequence does not preserve padded input IDs",
                                   {"input_ids": prompt, "sequence_ids": sequence})
    new = sequence[len(prompt):]  # Full input width INCLUDING padding, never decoded-string lengths.
    eos_position = next((j for j, token in enumerate(new) if token in active_eos), None)
    inspection = new if eos_position is None else new[:eos_position+1]
    text = tokenizer.decode(inspection, skip_special_tokens=False)
    delimiter_position, next_column = field_delimiter(text, known_columns)
    limit_hit = len(new) >= max_new_tokens if max_new_tokens is not None else len(sequence) >= max_length
    if delimiter_position >= 0:
        reason, raw = "delimiter", text[:delimiter_position]
    elif eos_position is not None:
        reason, raw = "eos", tokenizer.decode(new[:eos_position], skip_special_tokens=False)
    else:
        reason, raw = ("length" if limit_hit else "unknown"), text
    mask = ids(attention_mask) if attention_mask is not None else None
    evidence = {"input_ids": prompt, "input_width": len(prompt), "attention_mask": mask,
                "nonpadding_input_tokens": sum(mask[0]) if mask else None,
                "sequence_ids": sequence, "new_token_ids": new, "decoded_continuation": text,
                "raw_field": raw, "termination": reason, "active_eos_ids": list(active_eos),
                "eos_position": eos_position, "delimiter_character_position": delimiter_position if delimiter_position >= 0 else None,
                "delimiter_next_column": next_column,
                "requested_max_length": max_length, "inherited_max_new_tokens": max_new_tokens,
                "length_limit_reached": limit_hit, "field_closed": reason in ("delimiter", "eos")}
    return raw, evidence


def numeric_value(text, termination):
    if termination not in ("delimiter", "eos"):
        raise GreatCorrectionError("unclosed_numeric_field", "Digits alone do not establish completion")
    value = text.strip()
    if value.lower().lstrip("+-") in {"nan", "inf", "infinity"}:
        raise GreatCorrectionError("nonfinite_literal", "NaN/infinity are not finite numbers")
    if not NUMBER.fullmatch(value):
        raise GreatCorrectionError("numeric_grammar", "Expected a complete numeric literal, without units or substrings")
    number = float(value)
    if not math.isfinite(number):
        raise GreatCorrectionError("numeric_overflow", "Literal cannot be represented as a finite float")
    # Inspect the validated mantissa: Decimal itself has exponent size limits.
    if number == 0 and any(digit in '123456789' for digit in value.lower().split('e', 1)[0]):
        raise GreatCorrectionError("numeric_underflow", "Nonzero literal underflows to zero")
    return number, value  # Keep the validated ORIGINAL lexeme for subsequent prompt context.


def categorical_value(raw, feature, cat_cols, categorical_values, *, delimiter_closed=False):
    """Original nonnumeric cleanup/matching, deliberately without numeric parsing."""
    if delimiter_closed:
        value = raw.strip()
    elif ";" in raw:
        value = raw.split(";", 1)[0].strip()
    else:
        for delimiter in [",", ".", "\n", " "]:
            if delimiter in raw:
                value = raw.split(delimiter)[0].strip()
                break
        else:
            value = raw[:30].strip()
    while value and not (value[-1].isalnum() or value[-1] in ['.', '-']):
        value = value[:-1]
    if feature in cat_cols and feature in categorical_values:
        valid = categorical_values[feature]
        for cat in valid:
            if cat.lower() == value.lower():
                return cat
        for cat in valid:
            if cat.lower() in value.lower():
                return cat
    return value


def delimiter_stopping(tokenizer, input_width, known_columns=()):
    # Decoding the complete continuation handles multi-token and context-merged delimiters.
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch
    class CompleteDelimiter(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.tensor([field_delimiter(tokenizer.decode(row[input_width:], skip_special_tokens=False),
                                                known_columns)[0] >= 0
                                 for row in ids(input_ids)], dtype=torch.bool, device=input_ids.device)
    return StoppingCriteriaList([CompleteDelimiter()])


class SamplingAudit:
    def __init__(self, root, requested, correction, observations):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / f"call_{len(list(root.glob('call_*')))+1:04d}"
        self.directory.mkdir(exist_ok=False)
        self.fields = []
        self.summary = observations
        self.summary.update(execution_kind="revision_a1", correction=correction, requested_rows=requested,
                            audit_directory=str(self.directory), generate_calls=0, fallback_calls=0,
                            attempted_rows=0, completed_rows=0, discarded_rows=0, delivered_rows=0,
                            consecutive_rejections=0, rejection_reasons={}, padding_rows=0, status="started")
        self.save()

    def save(self):
        write(self.directory / "execution.json", self.summary)

    def field(self, record):
        self.fields.append(record)
        write(self.directory / f"field_{len(self.fields):05d}.json", record)


def guided_sample(model, audit, n_samples=10, temperature=.7, max_length=100, device="cuda",
                  random_feature_order=True, conditions=None):
    import pandas as pd
    if conditions:
        raise GreatCorrectionError("unsupported_column_conditions", "v1 covers the revised wrapper, which passes no column conditions")
    if model.columns is None:
        raise GreatCorrectionError("unfitted_model", "Missing columns")
    if getattr(getattr(model.model, "config", None), "is_encoder_decoder", False):
        raise GreatCorrectionError("unsupported_model_structure", "Audited model is decoder-only")
    model._resolve_device(device)
    model.model.eval()
    cat_cols = [c for c in model.columns if c.startswith('cat_')]
    categories = {}
    for col in cat_cols:
        if col == model.conditional_col and isinstance(model.conditional_col_dist, dict):
            categories[col] = list(model.conditional_col_dist)
    max_attempts = ROW_ATTEMPT_FACTOR*n_samples
    audit.summary.update(row_sampling_policy="discard incomplete row; keep complete rows; bounded replacement attempts",
                         limits={"max_row_attempts": max_attempts,
                                 "max_consecutive_rejections": MAX_CONSECUTIVE_REJECTIONS,
                                 "max_generate_calls": 2*max_attempts*len(model.columns)},
                         generate_call_limit_basis="at most primary plus existing fallback per field per row attempt")
    audit.save()
    rows = []
    while len(rows) < n_samples:
        exhausted = ("max_consecutive_rejections" if audit.summary['consecutive_rejections'] >= MAX_CONSECUTIVE_REJECTIONS
                     else "max_row_attempts" if audit.summary['attempted_rows'] >= max_attempts else None)
        if exhausted:
            audit.summary['termination'] = exhausted
            raise GreatCorrectionError('sampling_budget_exhausted', f'{exhausted}; retained {len(rows)} of {n_samples} complete rows')
        i = audit.summary['attempted_rows']
        audit.summary['attempted_rows'] += 1
        first_field = len(audit.fields)+1
        names = model.columns.copy()
        if random_feature_order:
            random.shuffle(names)  # Same Python RNG use as upstream.
        context, row = "", {}
        for feature in names:
            prompt = f"{context}{feature} is"
            field = {"row": i, "feature": feature, "prompt": prompt, "numeric_column": feature in model.num_cols,
                     "attempts": [], "status": "started"}
            try:
                inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
                width = inputs['input_ids'].shape[1]
                field.update(input_width=width, input_ids=ids(inputs['input_ids'])[0],
                             attention_mask=ids(inputs.get('attention_mask')),
                             delimiter_strategy="full decoded continuation; no first-token EOS heuristic")
                kwargs = dict(max_length=width+30, temperature=temperature,
                              pad_token_id=model.tokenizer.eos_token_id, do_sample=True)
                active_eos = eos_ids(model)
                config = getattr(model.model, 'generation_config', None)
                inherited_new = getattr(config, 'max_new_tokens', None)
                field['inherited_generation_config'] = {key: getattr(config, key, None) for key in
                    ('eos_token_id', 'max_new_tokens', 'top_k', 'top_p', 'num_beams', 'num_return_sequences',
                     'repetition_penalty', 'min_length', 'min_new_tokens')}
                # Preserve the existing single fallback, only for these generate exceptions.
                try:
                    field['attempts'].append({"path": "primary", "arguments": kwargs,
                                              "delimiter_stopping": True, "active_eos_ids": active_eos,
                                              "generation_invoked": False})
                    delimiter_ids = model.tokenizer.encode(FIELD_SEPARATOR, add_special_tokens=False)
                    # Context is a deterministic tokenizer probe, not a generated/imputed value.
                    contextual_ids = model.tokenizer.encode(prompt + ' 17.25' + FIELD_SEPARATOR, add_special_tokens=False)
                    field.update(delimiter_isolated_ids=ids(delimiter_ids), delimiter_context_probe_ids=ids(contextual_ids),
                                 delimiter_context_probe_text=prompt + ' 17.25' + FIELD_SEPARATOR)
                    primary = {**kwargs, 'stopping_criteria': delimiter_stopping(model.tokenizer, width, model.columns)}
                    audit.summary['generate_calls'] += 1
                    field['attempts'][-1]['generation_invoked'] = True
                    output = model.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **primary)
                except (ValueError, RuntimeError, IndexError) as exc:
                    field['attempts'][-1]['error'] = {"type": type(exc).__name__, "message": str(exc)}
                    audit.summary['generate_calls'] += 1
                    audit.summary['fallback_calls'] += 1
                    field['attempts'].append({"path": "existing_fallback", "arguments": kwargs,
                                              "delimiter_stopping": False, "active_eos_ids": active_eos,
                                              "generation_invoked": True})
                    output = model.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **kwargs)
                raw, evidence = continuation(output, inputs['input_ids'], model.tokenizer, active_eos=active_eos,
                                             max_length=width+30, attention_mask=inputs.get('attention_mask'),
                                             max_new_tokens=inherited_new, known_columns=model.columns)
                field.update(evidence)
                if feature in model.num_cols:
                    number, lexical = numeric_value(raw, evidence['termination'])
                    row[feature] = number
                else:
                    lexical = categorical_value(raw.strip(), feature, cat_cols, categories,
                                                delimiter_closed=evidence['termination'] == 'delimiter')
                    row[feature] = lexical
                field.update(status="accepted", value=row[feature], context_lexeme=lexical)
                context += f"{feature} is {lexical}{FIELD_SEPARATOR}"
            except Exception as exc:
                if isinstance(exc, GreatCorrectionError):
                    field.update(exc.evidence)
                recoverable = isinstance(exc, GreatCorrectionError) and exc.reason in PARSING_REJECTIONS
                field.update(status="rejected" if recoverable else "failed", reason=getattr(exc, 'reason', 'generation_error'),
                             error={"type": type(exc).__name__, "message": str(exc)})
                audit.field(field)
                write(audit.directory / f'row_{i+1:05d}.json', {
                    "row_attempt": i, "status": "rejected" if recoverable else "failed",
                    "first_field": first_field, "last_field": len(audit.fields), "partial_values": row,
                    "failed_feature": feature, "reason": field['reason'], "output_row": None})
                if recoverable:
                    audit.summary['discarded_rows'] += 1
                    audit.summary['consecutive_rejections'] += 1
                    reasons = audit.summary['rejection_reasons']
                    reasons[exc.reason] = reasons.get(exc.reason, 0)+1
                    audit.save()
                    break  # Discard the entire incomplete row; next attempt starts with empty context.
                raise
            audit.field(field)
        else:
            rows.append({feature: row[feature] for feature in model.columns})
            checkpoint = audit.directory / 'completed_rows.csv'
            pd.DataFrame([rows[-1]], columns=model.columns).to_csv(
                checkpoint, index=False, mode='a', header=not checkpoint.exists())
            write(audit.directory / f'row_{i+1:05d}.json', {
                "row_attempt": i, "status": "accepted", "first_field": first_field,
                "last_field": len(audit.fields), "output_row": len(rows)-1, "values": rows[-1]})
            audit.summary.update(completed_rows=len(rows), consecutive_rejections=0)
            audit.save()
    return pd.DataFrame(rows, columns=model.columns)


def sample_with_correction(model, n_samples, model_dir, observations):
    audit = SamplingAudit(Path(model_dir) / 'revision_great_sampling', n_samples, correction_record(), observations)
    existed = '_guided_sample' in vars(model)
    previous = vars(model).get('_guided_sample')
    def bound(instance, *args, **kwargs):
        return guided_sample(instance, audit, *args, **kwargs)
    model._guided_sample = types.MethodType(bound, model)
    try:
        # Execute the actual public library dispatcher, with the same wrapper kwargs.
        result = model.sample(n_samples, guided_sampling=True, max_length=500)
        if len(result) != n_samples:
            raise GreatCorrectionError('sample_shortfall', 'Public sample dispatcher returned an incompatible quota')
        audit.summary.update(status='delivered', termination='quota_reached', delivered_rows=len(result),
                             produced_rows=len(result), deficit_rows=0, remaining_valid_rows=0)
        return result
    except Exception as exc:
        audit.summary.update(status='failed', delivered_rows=0, produced_rows=audit.summary['completed_rows'],
                             deficit_rows=n_samples, completed_rows_withheld=audit.summary['completed_rows'],
                             remaining_valid_rows=n_samples-audit.summary['completed_rows'],
                             reason=getattr(exc,'reason','generation_error'), error={"type":type(exc).__name__,"message":str(exc)})
        audit.summary.setdefault('termination', 'structural_or_generation_error')
        checkpoint = audit.directory / 'completed_rows.csv'
        if checkpoint.exists():
            shutil.copyfile(checkpoint, audit.directory / 'completed_rows_withheld.csv')
        raise
    finally:
        if existed:
            model._guided_sample = previous
        else:
            del model._guided_sample
        audit.save()
