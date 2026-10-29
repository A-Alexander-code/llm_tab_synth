"""Explicit schema-constrained TabuLa sampling; training and legacy sampling stay native.

The native start feature/distribution is retained. Remaining predictor names are
forced in prepared column order. Numeric literals are sampled, never imputed.
This is a different sampling procedure from TabuLa's free text generation.
"""
from contextlib import contextmanager
import json
import math
import re
from pathlib import Path
from types import MethodType

import pandas as pd
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

STRATEGY = 'schema_constrained'
NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
TERMINAL = frozenset(('integer', 'decimal', 'fraction', 'exponent_digits'))


def validate_schema(columns, numerical_columns):
    if (not columns or len(set(columns)) != len(columns)
            or set(columns) != set(numerical_columns)
            or any(not isinstance(c, str) or not c or not c.isascii() or ',' in c for c in columns)):
        raise ValueError('Schema-constrained TabuLa requires unique ASCII numeric predictor names without commas')


def numeric_step(state, character):
    if character in '0123456789':
        if state in ('start', 'sign', 'integer'):
            return 'integer'
        if state in ('leading_decimal', 'decimal', 'fraction'):
            return 'fraction'
        if state in ('exponent', 'exponent_sign', 'exponent_digits'):
            return 'exponent_digits'
    if character in '+-' and state in ('start', 'exponent'):
        return 'sign' if state == 'start' else 'exponent_sign'
    if character == '.' and state in ('start', 'sign', 'integer'):
        return 'decimal' if state == 'integer' else 'leading_decimal'
    if character in 'eE' and state in ('integer', 'decimal', 'fraction'):
        return 'exponent'
    return None


class SchemaLogitsProcessor(LogitsProcessor):
    """Token-level finite grammar over literal names and complete numeric values.

    Token transitions consume *all* characters, including tokens crossing a
    comma/name/value boundary. No substring extraction or value substitution.
    EOS is permitted only after the last complete numeric literal.
    """
    def __init__(self, tokenizer, columns, start_column, prompt_width, batch_size, token_tables=None):
        self.tokenizer = tokenizer
        self.remaining = [c for c in columns if c != start_column]
        self.literals = [' '+c+' ' for c in self.remaining]
        self.prompt_width = prompt_width
        self.consumed = prompt_width
        self.eos = tokenizer.eos_token_id
        self.states = [(0, 0, None)]*batch_size
        self.special = set(tokenizer.all_special_ids)
        if token_tables is None:
            self.pieces, self.by_initial, self.cache = {}, {}, {}
            for token in range(len(tokenizer)):
                if token in self.special:
                    continue
                piece = tokenizer.decode([token], clean_up_tokenization_spaces=False)
                if piece and piece.isascii():
                    self.pieces[token] = piece
                    self.by_initial.setdefault(piece[0], []).append(token)
        else:
            self.pieces, self.by_initial, self.cache = token_tables

    def transition(self, state, piece):
        index, offset, numeric = state
        for character in piece:
            if index >= len(self.literals):
                return None
            if numeric is None:
                literal = self.literals[index]
                if character != literal[offset]:
                    return None
                offset += 1
                if offset == len(literal):
                    numeric = 'start'
                continue
            updated = numeric_step(numeric, character)
            if updated is not None:
                numeric = updated
            elif character == ',' and numeric in TERMINAL:
                if index+1 == len(self.literals):
                    index, offset, numeric = index+1, 0, 'closed'
                else:
                    index, offset, numeric = index+1, 0, None
            else:
                return None
        return index, offset, numeric

    def eos_allowed(self, state):
        index, _, numeric = state
        return not self.literals or numeric == 'closed' or (index == len(self.literals)-1 and numeric in TERMINAL)

    def allowed(self, state):
        if state not in self.cache:
            index, offset, numeric = state
            initials = ('0123456789+-.eE,' if numeric is not None else
                        self.literals[index][offset] if self.literals else '')
            allowed = []
            for first in initials:
                for token in self.by_initial.get(first, ()):
                    if self.transition(state, self.pieces[token]) is not None:
                        allowed.append(token)
            if self.eos_allowed(state):
                allowed.append(self.eos)
            if not allowed:
                raise RuntimeError('TabuLa schema grammar has no available token')
            self.cache[state] = allowed
        return self.cache[state]

    def __call__(self, input_ids, scores):
        if len(input_ids) != len(self.states) or input_ids.shape[1] < self.consumed:
            raise RuntimeError('TabuLa schema processor batch/token order changed')
        for row, suffix in enumerate(input_ids[:, self.consumed:].tolist()):
            for token in suffix:
                if token == self.eos and self.eos_allowed(self.states[row]):
                    # Finished sequences are padded with EOS by generate.
                    self.states[row] = (len(self.literals), 0, 'finished')
                elif self.states[row][2] == 'finished' and token == self.eos:
                    continue
                else:
                    updated = self.transition(self.states[row], self.pieces.get(token, ''))
                    if not self.pieces.get(token) or updated is None:
                        raise RuntimeError('TabuLa emitted a token outside the schema grammar')
                    self.states[row] = updated
        self.consumed = input_ids.shape[1]
        constrained = torch.full_like(scores, -float('inf'))
        for row, state in enumerate(self.states):
            allowed = [self.eos] if state[2] == 'finished' else self.allowed(state)
            constrained[row, allowed] = scores[row, allowed]
        if not torch.isfinite(constrained).any(dim=1).all():
            raise RuntimeError('TabuLa has no finite score for schema-compatible tokens')
        return constrained


def parse_row(text, columns, start_column):
    """Validate the complete emitted row; do not recover partial substrings."""
    expected = [start_column]+[c for c in columns if c != start_column]
    fields = text.split(',')
    if len(fields) == len(expected)+1 and fields[-1] == '':
        fields.pop()  # A closing delimiter is valid only after all known fields.
    if len(fields) != len(expected):
        raise ValueError('field_count')
    values = {}
    for column, field in zip(expected, fields):
        prefix = column+' '
        field = field.strip()
        if not field.startswith(prefix):
            raise ValueError('column_name_or_order')
        literal = field[len(prefix):]
        if NUMBER.fullmatch(literal) is None:
            raise ValueError('numeric_grammar')
        number = float(literal)
        if not math.isfinite(number):
            raise ValueError('nonfinite')
        values[column] = number
    return [values[c] for c in columns]


def schema_sample(self, n_samples, start_col='', start_col_dist=None,
                  temperature=.7, k=100, max_length=100, device='cuda'):
    """Same candidate batch semantics and native start, with explicit grammar."""
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError('TabuLa requires a positive integer sample quota')
    validate_schema(self.columns, self.num_cols)
    if self.categorical_columns:
        raise ValueError('Schema-constrained sampling does not support native category codecs')
    start_column = start_col or self.conditional_col
    if start_column not in self.columns:
        raise ValueError('Unknown schema start column')
    start = self._get_start_sampler(start_col, start_col_dist)
    self.model.to(device)
    df_gen = pd.DataFrame(columns=self.columns, dtype=float)
    observations, audit_path = self._schema_observations, self._schema_audit_path
    batch_number = 0
    token_tables = None
    observations.update(sampling_strategy=STRATEGY,
        schema_column_order=[start_column]+[c for c in self.columns if c != start_column],
        schema_policy='native start; remaining names in prepared order; model-generated numeric literals; EOS after complete row, optional final comma',
        schema_batches=[])
    with tqdm(total=n_samples) as pbar:
        while len(df_gen) < n_samples:
            start_tokens = torch.tensor(start.get_start_tokens(k), device=device)
            # Check the untouched native prompts before applying completion grammar.
            for row in start_tokens:
                prompt = self.tokenizer.decode(row[row != self.tokenizer.eos_token_id],
                                               clean_up_tokenization_spaces=False)
                if not prompt.endswith(','):
                    raise RuntimeError('TabuLa native start no longer ends in comma')
                parse_row(prompt, [start_column], start_column)
            processor = SchemaLogitsProcessor(self.tokenizer, self.columns, start_column,
                                              start_tokens.shape[1], k, token_tables)
            token_tables = processor.pieces, processor.by_initial, processor.cache
            tokens = self.model.generate(input_ids=start_tokens, max_length=max_length,
                do_sample=True, temperature=temperature, pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([processor]))
            batch_number += 1
            records, accepted = [], []
            for position, row in enumerate(tokens):
                suffix = row[start_tokens.shape[1]:].tolist()
                closed = self.tokenizer.eos_token_id in suffix
                end = start_tokens.shape[1]+suffix.index(self.tokenizer.eos_token_id) if closed else len(row)
                ids = row[:end].tolist()
                # Remove only native left padding, not generated text.
                while ids and ids[0] == self.tokenizer.eos_token_id:
                    ids.pop(0)
                text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
                reason, values = None, None
                try:
                    if not closed:
                        raise ValueError('length_limit_without_eos')
                    values = parse_row(text, self.columns, start_column)
                    accepted.append(values)
                except ValueError as exc:
                    reason = str(exc)
                records.append({'generate_call': batch_number, 'candidate_in_batch': position,
                    'text': text, 'accepted': reason is None, 'rejection': reason,
                    'termination': 'eos' if closed else 'length_limit',
                    'last_field_closure': ('comma' if text.endswith(',') else 'eos') if closed else None,
                    'accepted_origin': len(df_gen)+len(accepted)-1 if reason is None else None,
                    'selected_for_delivery': len(df_gen)+len(accepted) <= n_samples if reason is None else False,
                    'values': dict(zip(self.columns, values)) if values is not None else None})
            path = audit_path.with_name('sampling_candidates.jsonl')
            with path.open('x' if batch_number == 1 else 'a', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            observations['schema_batches'].append({'call': batch_number, 'candidates': len(records),
                'accepted': len(accepted), 'rejected': len(records)-len(accepted),
                'rejection_counts': {r: sum(item['rejection'] == r for item in records)
                                     for r in sorted({item['rejection'] for item in records if item['rejection']})}})
            observations.update(schema_candidates_path=path.name,
                internal_rejected_rows_observed=sum(batch['rejected'] for batch in observations['schema_batches']))
            if accepted:
                accepted_frame = pd.DataFrame(accepted, columns=self.columns)
                retained = audit_path.with_name('sampling_accepted_before_selection.csv')
                accepted_frame.to_csv(retained, index=False, mode='x' if len(df_gen) == 0 else 'a',
                                      header=len(df_gen) == 0)
                observations['accepted_before_selection_path'] = retained.name
                df_gen = pd.concat([df_gen, accepted_frame], ignore_index=True)
            pbar.update(len(accepted))
    observations.update(schema_candidates_path=path.name, accepted_before_quota_selection=len(df_gen),
                        excess_rows=len(df_gen)-n_samples, copies_added=0)
    return df_gen.reset_index(drop=True).head(n_samples)


@contextmanager
def schema_constrained_sampling(model, observations, audit_path):
    original = model.sample
    had_sample = 'sample' in vars(model)
    if any(name in vars(model) for name in ('_schema_observations', '_schema_audit_path')):
        raise RuntimeError('Nested TabuLa schema sampling is unsupported')
    model._schema_observations, model._schema_audit_path = observations, Path(audit_path)
    model.sample = MethodType(schema_sample, model)
    try:
        yield
    finally:
        if had_sample:
            model.sample = original
        else:
            del model.sample
        del model._schema_observations, model._schema_audit_path
