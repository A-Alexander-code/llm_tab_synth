"""Auditable balanced repetition for revised TabPFGen only; no model imports."""
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import uuid
import warnings

import numpy as np
import pandas as pd

VERSION = 'tabpfgen-balanced-repetition-a1/1.0.0'
BINDING = 'revision_tabpfgen_identity.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


class CompletionError(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(f'{reason}: {message}')


def require(condition, reason, message):
    if not condition:
        raise CompletionError(reason, message)


def policy_record():
    dist = importlib.metadata.distribution('tabpfgen')
    source = Path(dist.locate_file('tabpfgen/tabpfgen.py')).resolve()
    return {'version': VERSION, 'implementation_sha256': sha(__file__),
        'wrapper_sha256': sha(Path(__file__).parent/'generators/tabpfgen.py'),
        'library_version': dist.version, 'library_source': str(source), 'library_sha256': sha(source),
        'tabpfn_version': importlib.metadata.version('tabpfn'),
        'name': 'TabPFGen con completado por repetición',
        'selection': 'first accepted origins in library return order; never deduplicate',
        'padding': 'retain each selected origin once, then cyclic copies from first origin; multiplicities differ by <=1',
        'empty_class': 'fail the request; retain successful other-class output for audit only',
        'budget': 'unchanged: max_iterations (default 200), max(4*total_missing,100) candidates per call',
        'cache': 'retain unpadded sources; recompute each delivery; reuse when quota<=generation quota, regenerate otherwise',
        'raw_output_scope': 'generate_classification return, after native transforms/label refinement; earlier internal stages not exposed',
        'checkpoint': 'unchanged library default, no new model_path or checkpoint selection'}


def validate_policy(params):
    context = params.get('_revision_parameters', {})
    require(context.get('completion_policy') == policy_record(), 'policy_identity_mismatch', 'Stale or missing revised completion policy')
    limit = params.get('max_iterations', 200)
    require(type(limit) is int and limit >= 0, 'invalid_budget', 'max_iterations must be a nonnegative integer')


def check_fit(frame, model_dir, params):
    validate_policy(params)
    require(not (model_dir/'train_data.pkl').exists() and not (model_dir/BINDING).exists(),
            'existing_training_artifact', 'Revised fit never overwrites/rebinds existing training data')
    require(frame.columns.is_unique and '__target__' in frame, 'training_schema', 'Unique predictors and target required')
    features = frame.drop(columns='__target__')
    require(len(features.columns) > 0 and all(pd.api.types.is_numeric_dtype(t) and
            not pd.api.types.is_complex_dtype(t) for t in features.dtypes), 'training_types', 'Real numeric predictors required')
    require(np.isfinite(features.to_numpy()).all(), 'training_nonfinite', 'Training predictors must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        require(np.isfinite(features.to_numpy().astype(np.float32)).all(),
                'training_float32_nonfinite', 'Existing float32 conversion would produce nonfinite training values')
    require('__label__' not in features, 'training_schema', 'Reserved label cannot be a predictor')
    target = frame['__target__']
    require(pd.api.types.is_numeric_dtype(target) and not pd.api.types.is_complex_dtype(target) and np.isfinite(target).all() and
            set(target.tolist()) == {0, 1}, 'training_labels', 'Revised runtime requires both binary classes, coded 0/1')


def bind_training(model_dir, params, feature_cols):
    write(model_dir/BINDING, {'policy': policy_record(), 'effective_parameters': params,
          'features': feature_cols, 'classes': [0, 1], 'train_data_sha256': sha(model_dir/'train_data.pkl')})


def check_binding(model_dir, params):
    validate_policy(params)
    file = model_dir/BINDING
    require(file.is_file(), 'missing_revised_identity', 'Historical/unidentified data cannot be reused in revised mode')
    binding = json.loads(file.read_text())
    require(binding['policy'] == policy_record() and binding['effective_parameters'] == params,
            'artifact_identity_mismatch', 'Policy, seed or effective parameters differ')
    require(binding['train_data_sha256'] == sha(model_dir/'train_data.pkl'), 'training_hash_mismatch', 'Training payload changed')
    require(not (model_dir/'samples_cache.pkl').exists(), 'historical_cache_present', 'Historical padded cache is incompatible')
    return binding


def label_of(value):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'biuf':
        return None, 'label_type_or_shape'
    number = array.item()
    if not math.isfinite(number):
        return None, 'label_nonfinite'
    if number not in (0, 1):
        return None, 'label_outside_binary_domain'
    return int(number), None


def complete_class(sources, quota, columns, label):
    """Origin identity is positional; content equality never reduces the source pool."""
    require(type(quota) is int and quota > 0, 'invalid_quota', 'Quota must be positive integer')
    chosen = sources[:quota]
    count = len(chosen)
    indices = list(range(count))
    if count:
        indices += [i % count for i in range(quota-count)]
    multiplicities = Counter(indices)
    final = pd.DataFrame([chosen[i]['values'] for i in indices], columns=columns)
    before = pd.DataFrame([s['values'] for s in chosen], columns=columns)
    lineage = [{'class': label, 'class_row': j, 'combined_row': label*quota+j,
                'origin_id': chosen[i]['origin_id'], 'batch': chosen[i]['batch'],
                'batch_row': chosen[i]['batch_row'], 'padding_copy': j >= count}
               for j, i in enumerate(indices)]
    duplicates = int(before.duplicated().sum())
    copies = len(final)-count
    stats = {'quota': quota, 'selected_before_padding': count, 'padding_copies': copies,
        'final_rows': len(final), 'final_unique_contents': len(final.drop_duplicates()),
        'selected_unique_contents_before_padding': len(before.drop_duplicates()),
        'natural_duplicate_rows_before_padding': duplicates,
        'final_duplicate_rows_extra': int(final.duplicated().sum()),
        'duplicate_decomposition': {'natural_in_selected_rows': duplicates, 'added_padding_copies': copies},
        'padding_fraction': copies/quota, 'quota_covered_fraction': len(final)/quota,
        'origin_multiplicities': [{'origin_id': s['origin_id'], 'multiplicity': multiplicities[i]}
                                  for i, s in enumerate(chosen)],
        'status': 'ready' if count else 'failed_empty_class'}
    require(stats['final_duplicate_rows_extra'] == duplicates+copies,
            'lineage_invariant', 'Duplicate accounting inconsistent')
    if count:
        require(max(multiplicities.values())-min(multiplicities.values()) <= 1,
                'lineage_invariant', 'Unbalanced repetition')
    return before, final, lineage, stats


def new_directory(root, prefix):
    numbers = [int(p.name[len(prefix):]) for p in root.glob(prefix+'*') if p.name[len(prefix):].isdigit()]
    directory = root/f'{prefix}{max(numbers, default=0)+1:04d}'
    directory.mkdir(exist_ok=False)
    return directory


def generate_pool(root, params, state, quota, factory, binding, observations):
    directory = new_directory(root, 'generation_')
    generation_id = uuid.uuid4().hex  # Audit namespace only; does not consume the experimental RNG.
    pools = {0: [], 1: []}
    seen = {0: set(), 1: set()}
    by_class = {c: {'candidate_rows': 0, 'accepted_rows': 0, 'structural_rejections': 0,
                    'rejection_reasons': Counter()} for c in (0, 1)}
    record = {'generation_id': generation_id, 'generation_quota': quota, 'iterations': 0,
        'max_iterations': params.get('max_iterations', 200), 'candidates_requested': 0,
        'X_rows_received': 0, 'y_rows_received': 0, 'unattributed_rejections': Counter(),
        'structural_rejected_rows': 0, 'unattributed_rejected_rows': 0,
        'batches': [], 'termination': 'started', 'status': 'started'}
    observations.update(generation_directory=str(directory), generation_context=record, cache_reused=False)
    write(directory/'generation.json', record)
    try:
        model = factory(n_sgld_steps=params['n_sgld_steps'], sgld_step_size=params['sgld_step_size'],
                        sgld_noise_scale=params['sgld_noise_scale'], device=params['device'])
        with (directory/'row_dispositions.csv').open('x', newline='') as stream:
            events = csv.DictWriter(stream, fieldnames=['origin_id', 'batch', 'batch_row', 'class', 'disposition', 'reasons'])
            events.writeheader()
            while record['iterations'] < record['max_iterations']:
                missing = sum(max(0, quota-len(pools[c])) for c in (0, 1))
                if not missing:
                    break
                requested = max(missing*4, 100)  # Existing budget, unchanged.
                batch_index = record['iterations']+1
                batch = {'batch': batch_index, 'requested': requested, 'status': 'started'}
                record['batches'].append(batch)
                record['candidates_requested'] += requested
                record['iterations'] += 1
                observations.update(current_library_calls=record['iterations'], current_candidates_requested=record['candidates_requested'])
                write(directory/'generation.json', record)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.filterwarnings('ignore', message='.*Running on CPU with more than 200 samples.*',
                                            category=UserWarning, module='tabpfn')
                    result = model.generate_classification(state['X'], state['y'], n_samples=requested, balance_classes=True)
                file = directory/f'batch_{batch_index:04d}.pkl'
                with file.open('xb') as output:
                    pickle.dump(result, output)
                batch.update(raw_file=file.name, raw_sha256=sha(file), warnings=[str(w.message) for w in caught])
                require(isinstance(result, (tuple, list)) and len(result) == 2, 'library_return_structure', 'Expected X,y pair')
                X, y = result
                require(hasattr(X, '__len__') and hasattr(y, '__len__'), 'library_return_structure', 'Expected row containers')
                x_count, y_count = len(X), len(y)
                record['X_rows_received'] += x_count
                record['y_rows_received'] += y_count
                observations['current_X_rows_received'] = record['X_rows_received']
                batch.update(X_rows=x_count, y_rows=y_count)
                schema_error = []
                if isinstance(X, pd.DataFrame):
                    if list(X.columns) != state['feature_cols']:
                        schema_error.append('feature_schema_mismatch')
                    rows = list(X.itertuples(index=False, name=None))
                else:
                    rows = X
                labels = np.asarray(y)
                aligned = labels.ndim == 1 and x_count == y_count
                batch.update(structural_errors=[] if aligned else ['row_label_alignment_mismatch'],
                             unmatched_label_rows=max(0, y_count-x_count))
                for j, x in enumerate(rows):
                    origin = f'{generation_id}:batch_{batch_index}:row_{j}'
                    label, error = label_of(labels[j]) if aligned else (None, 'row_label_alignment_mismatch')
                    errors = list(schema_error)+([error] if error else [])
                    array = np.asarray(x)
                    if array.ndim != 1 or len(array) != len(state['feature_cols']):
                        errors.append('feature_shape')
                    elif array.dtype.kind not in 'biuf':
                        errors.append('feature_type')
                    elif not np.isfinite(array).all():
                        errors.append('feature_nonfinite')
                    disposition = 'rejected'
                    if errors:
                        record['structural_rejected_rows'] += 1
                    if label is not None:
                        info = by_class[label]
                        info['candidate_rows'] += 1
                        if errors:
                            info['structural_rejections'] += 1
                            info['rejection_reasons'].update(errors)
                        else:
                            info['accepted_rows'] += 1
                            values = array.tolist()
                            seen[label].add(tuple(values))
                            disposition = 'accepted_surplus'
                            if len(pools[label]) < quota:
                                pools[label].append({'origin_id': origin, 'batch': batch_index, 'batch_row': j, 'values': values})
                                disposition = 'accepted_retained_source'
                    else:
                        record['unattributed_rejected_rows'] += 1
                        record['unattributed_rejections'].update(errors)
                    events.writerow({'origin_id': origin, 'batch': batch_index, 'batch_row': j,
                        'class': label, 'disposition': disposition, 'reasons': json.dumps(errors)})
                stream.flush()
                batch['status'] = 'processed'
                write(directory/'generation.json', record)
        record.update(status='completed', termination='quota_reached' if all(len(pools[c]) >= quota for c in (0, 1)) else 'max_iterations')
        record['classes'] = {str(c): {**by_class[c], 'accepted_unique_contents': len(seen[c]),
            'natural_duplicate_rows_all_accepted': by_class[c]['accepted_rows']-len(seen[c]),
            'retained_source_rows': len(pools[c])} for c in (0, 1)}
        with (directory/'unpadded_sources.pkl').open('xb') as stream:
            pickle.dump(pools, stream)
        write(directory/'generation.json', record)
        manifest = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
        write(directory/'manifest.json', manifest)
        write(root/'cache.json', {'binding_sha256': binding, 'directory': directory.name,
            'generation_quota': quota, 'manifest_sha256': sha(directory/'manifest.json')})
        return directory, record, pools
    except Exception as exc:
        record.update(status='failed', termination='structural_or_library_failure',
                      error={'type': type(exc).__name__, 'message': str(exc)})
        record['classes'] = {str(c): {**by_class[c], 'retained_source_rows': len(pools[c])} for c in (0, 1)}
        write(directory/'generation.json', record)
        raise


def sample_revised(params, quota, model_dir, factory, observations):
    require(type(quota) is int and quota > 0, 'invalid_quota', 'Quota must be a positive integer')
    binding = check_binding(model_dir, params)
    root = model_dir/'revision_tabpfgen'
    root.mkdir(exist_ok=True)
    call = new_directory(root, 'call_')
    observations.update(policy=policy_record(), requested_per_class=quota, audit_directory=str(call),
                        status='started', returned_rows=0, current_library_calls=0,
                        current_candidates_requested=0, current_X_rows_received=0)
    write(call/'execution.json', observations)
    try:
        key = sha(model_dir/BINDING)
        cache_file = root/'cache.json'
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else None
        if cache:
            require(cache['binding_sha256'] == key, 'cache_identity_mismatch', 'Cache does not match training/parameters')
            name = cache['directory']
            require(Path(name).name == name and name.startswith('generation_'), 'cache_identity_mismatch', 'Invalid cache path')
            cached = root/name
            require(sha(cached/'manifest.json') == cache['manifest_sha256'], 'cache_hash_mismatch', 'Cache manifest changed')
            for file, expected in json.loads((cached/'manifest.json').read_text()).items():
                require(Path(file).name == file and sha(cached/file) == expected, 'cache_hash_mismatch', 'Cached raw source changed')
            context = json.loads((cached/'generation.json').read_text())
            require(context['status'] == 'completed' and context['generation_quota'] == cache['generation_quota'],
                    'cache_identity_mismatch', 'Generation context differs from cache quota/status')
        reused = cache is not None and quota <= cache['generation_quota']
        if reused:
            generation = cached
            with (generation/'unpadded_sources.pkl').open('rb') as stream:
                pools = pickle.load(stream)
        else:
            with (model_dir/'train_data.pkl').open('rb') as stream:
                state = pickle.load(stream)
            require(state['feature_cols'] == binding['features'] and state['X'].shape[1] == len(binding['features']) and
                    set(np.unique(state['y'])) == {0, 1}, 'training_schema', 'Training state differs from binding')
            generation, context, pools = generate_pool(root, params, state, quota, factory, key, observations)
            observations.update(current_library_calls=context['iterations'],
                current_candidates_requested=context['candidates_requested'], current_X_rows_received=context['X_rows_received'])
        observations.update(cache_reused=reused, generation_directory=str(generation), generation_context=context,
            candidates_requested_per_class=None,
            candidate_request_scope='API requests a total batch with balance_classes=True; per-class requests not observable', classes={})
        request_identity = {'training_binding_sha256': key, 'quota': quota, 'policy_version': VERSION,
                            'generation_id': context['generation_id']}
        observations['request_identity'] = request_identity
        observations['request_identity_sha256'] = hashlib.sha256(json.dumps(request_identity, sort_keys=True).encode()).hexdigest()
        output, failed = [], []
        for label in (0, 1):
            before, final, lineage, stats = complete_class(pools[label], quota, binding['features'], label)
            available = context['classes'][str(label)]
            stats.update(accepted_rows_from_generation=available['accepted_rows'],
                candidate_rows_received_from_generation=available['candidate_rows'],
                structural_rejections=available['structural_rejections'], rejection_reasons=available['rejection_reasons'],
                natural_duplicate_rows_all_accepted=available['natural_duplicate_rows_all_accepted'],
                retained_sources_available=len(pools[label]),
                surplus_accepted_rows=available['accepted_rows']-len(before),
                generation_termination=context['termination'])
            before.to_pickle(call/f'class_{label}_before_padding.pkl')
            pd.DataFrame(pools[label]).to_pickle(call/f'class_{label}_source_pool.pkl')
            final.to_pickle(call/f'class_{label}_final.pkl')
            pd.DataFrame(lineage, columns=['class', 'class_row', 'combined_row', 'origin_id', 'batch', 'batch_row', 'padding_copy']).to_csv(
                call/f'class_{label}_lineage.csv', index=False)
            observations['classes'][str(label)] = stats
            if stats['padding_copies']:
                warnings.warn(f'TabPFGen con completado por repetición: class {label}, quota={quota}, '
                    f'generated origins used={len(before)}, added copies={stats["padding_copies"]}; '
                    'copies are not independent generated samples.', UserWarning, stacklevel=2)
            if not len(before):
                failed.append(label)
            labelled = final.copy()
            labelled['__label__'] = label
            output.append(labelled)
        observations['failed_classes'] = failed
        for label in (0, 1):
            item = observations['classes'][str(label)]
            item['delivered_rows'] = 0 if failed else item['final_rows']
            item['withheld_rows'] = item['final_rows'] if failed else 0
        require(not failed, 'empty_generated_class', f'No accepted source rows for classes {failed}; audit={call}')
        result = pd.concat(output, ignore_index=True)
        require(list(result.columns) == binding['features']+['__label__'] and len(result) == 2*quota and
                all(pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_complex_dtype(t) for t in result.dtypes) and
                np.isfinite(result.to_numpy()).all() and result['__label__'].isin([0, 1]).all(),
                'final_structure', 'Completed output must preserve schema, real numeric types, finiteness and class')
        observations.update(status='delivered', returned_rows=len(result))
        return result
    except Exception as exc:
        observations.update(status='failed', returned_rows=0, reason=getattr(exc, 'reason', 'library_or_audit_failure'),
                            error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        observations['files_sha256'] = {p.name: sha(p) for p in call.iterdir() if p.is_file() and p.name != 'execution.json'}
        write(call/'execution.json', observations)
        write(root/'latest_sampling_audit.json', observations)
