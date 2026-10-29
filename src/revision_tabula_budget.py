"""Bounds and instance-local prompt masks around the native TabuLa sampler."""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

VERSION = 'tabula-pilot-generation-bounds/1'


@contextmanager
def attention_masked_sampling(model, observations):
    """Keep native prompts/RNG/padding, retaining masks before padding loses lengths.

    Installed outside bounded_sample so the bounds still inspect the native
    sample frame. No global tokenizer, vendor method or generation setting is changed.
    """
    original_start = model._get_start_sampler
    original_generate = model.model.generate
    had_start = '_get_start_sampler' in vars(model)
    had_generate = 'generate' in vars(model.model)
    pending = None
    observations.update(attention_mask_policy='original tokenizer masks plus zeros for native left padding',
                        attention_mask_batches=[])

    class MaskedStart:
        def __init__(self, start):
            self.start = start

        def get_start_tokens(self, n):
            nonlocal pending
            captured = []
            tokenizer = self.start.tokenizer

            class CaptureTokenizer:
                def __call__(self, *args, **kwargs):
                    result = tokenizer(*args, **kwargs)
                    captured.append(result)
                    return result

                def __getattr__(self, name):
                    return getattr(tokenizer, name)

            self.start.tokenizer = CaptureTokenizer()
            try:
                padded = self.start.get_start_tokens(n)
            finally:
                self.start.tokenizer = tokenizer
            if len(captured) != 1:
                raise RuntimeError('TabuLa prompt tokenization changed; cannot establish attention mask')
            original = captured[0]['input_ids']
            masks = captured[0].get('attention_mask', [[1]*len(row) for row in original])
            if len(original) != n or len(padded) != n or len(masks) != n:
                raise RuntimeError('TabuLa prompt batch/mask count mismatch')
            widths = {len(row) for row in padded}
            if len(widths) != 1:
                raise RuntimeError('TabuLa prompts must have a common padded width')
            width = widths.pop()
            attention = []
            for ids, mask, row in zip(original, masks, padded):
                left = width-len(ids)
                if (left < 0 or len(ids) != len(mask) or any(value not in (0, 1) for value in mask)
                        or list(row) != [50256]*left+list(ids)):
                    raise RuntimeError('TabuLa native left padding does not match original prompt')
                attention.append([0]*left+list(mask))
            pending = {'input_ids': padded, 'attention_mask': attention,
                       'prompt_lengths': [len(row) for row in original], 'input_width': width}
            return padded

    def get_start(*args, **kwargs):
        return MaskedStart(original_start(*args, **kwargs))

    def generate(*args, **options):
        nonlocal pending
        import torch
        batch, pending = pending, None
        if batch is None:
            raise RuntimeError('TabuLa generate has no matching original prompt mask')
        tokens = options.get('input_ids', args[0] if args else None)
        if tokens is None or tokens.ndim != 2:
            raise RuntimeError('TabuLa generate requires a two-dimensional input token batch')
        expected = torch.tensor(batch['input_ids'], dtype=tokens.dtype, device=tokens.device)
        if not torch.equal(tokens, expected):
            raise RuntimeError('TabuLa generate tokens differ from captured prompts')
        mask = torch.tensor(batch['attention_mask'], dtype=torch.long, device=tokens.device)
        supplied = options.get('attention_mask')
        if supplied is not None and not torch.equal(supplied, mask):
            raise RuntimeError('TabuLa existing attention mask differs from original prompts')
        options['attention_mask'] = mask
        observations['attention_mask_batches'].append({
            'call': len(observations['attention_mask_batches'])+1,
            'prompt_lengths': batch['prompt_lengths'], 'input_width': batch['input_width'],
            'attention_mask': batch['attention_mask'], 'device': str(tokens.device),
            'padding_positions': sum(batch['input_width']-length for length in batch['prompt_lengths']),
            'masked_positions': sum(row.count(0) for row in batch['attention_mask'])})
        return original_generate(*args, **options)

    model._get_start_sampler = get_start
    model.model.generate = generate
    try:
        yield
    finally:
        if had_start:
            model._get_start_sampler = original_start
        else:
            del model._get_start_sampler
        if had_generate:
            model.model.generate = original_generate
        else:
            del model.model.generate


def validate_limits(limits):
    required = ('max_generate_calls', 'max_candidates', 'max_no_progress')
    if set(limits) != set(required) or any(type(limits[k]) is not int or limits[k] < 1 for k in required):
        raise ValueError('Explicit TabuLa limits required: max_generate_calls, max_candidates, max_no_progress (positive integers)')


def bounded_sample(model, n, kwargs, limits, observations, audit_path, *, scope='pilot'):
    validate_limits(limits)
    if scope not in ('pilot', 'revised'):
        raise ValueError('Unknown TabuLa limits scope')
    original = model.model.generate
    calls, candidates, stalled, previous_rows = 0, 0, 0, None
    caller = None
    observations.update(budget_version=VERSION if scope == 'pilot' else 'tabula-revised-generation-bounds/1',
                        limits_scope=scope, limits=limits, batches=[],
                        internal_generate_calls_observed=0, internal_candidates_observed=0,
                        accepted_rows_observed=0, consecutive_no_progress=0)
    def save():
        Path(audit_path).write_text(json.dumps(observations, indent=2)+'\n')
    def generate(*args, **options):
        nonlocal calls, candidates, stalled, previous_rows, caller
        caller = inspect.currentframe().f_back
        frame = caller.f_locals.get('df_gen')
        if frame is None:
            raise RuntimeError('TabuLa native sampler no longer exposes progress; stop')
        accepted = len(frame)
        if previous_rows is not None:
            stalled = stalled+1 if accepted <= previous_rows else 0
        previous_rows = accepted
        tensor = options.get('input_ids', args[0] if args else None)
        batch = int(tensor.shape[0])
        reason = ('no_progress' if stalled >= limits['max_no_progress'] else
                  'generate_calls' if calls >= limits['max_generate_calls'] else
                  'candidates' if candidates+batch > limits['max_candidates'] else None)
        observations.update(accepted_rows_observed=accepted, consecutive_no_progress=stalled)
        if reason:
            observations.update(termination=scope+'_budget_'+reason)
            save()
            raise RuntimeError('TabuLa '+scope+' sampling budget exhausted: '+reason)
        calls += 1
        candidates += batch
        observations['batches'].append({'call': calls, 'candidates': batch, 'accepted_before': accepted})
        observations.update(internal_generate_calls_observed=calls, internal_candidates_observed=candidates)
        save()
        return original(*args, **options)
    model.model.generate = generate
    try:
        result = model.sample(n, **kwargs)
        observations.update(termination='quota_reached' if len(result) == n else 'native_shortfall',
                            wrapper_return_rows=len(result))
        return result
    except Exception:
        if caller is not None and 'df_gen' in caller.f_locals:
            retained = caller.f_locals['df_gen']
            if len(retained):
                path = Path(audit_path).with_name('sampling_accepted_withheld.csv')
                retained.to_csv(path, index=False, mode='x')
                observations['accepted_withheld'] = {'path': path.name, 'rows': len(retained),
                                                     'delivered': False}
        raise
    finally:
        model.model.generate = original
        if caller is not None and 'df_gen' in caller.f_locals:
            observations['accepted_rows_observed'] = len(caller.f_locals['df_gen'])
        observations.update(internal_generate_calls_observed=calls, internal_candidates_observed=candidates)
        save()
