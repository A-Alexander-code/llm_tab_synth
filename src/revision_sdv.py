"""Explicit semantic SDV metadata for revised classwise generators only."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

VERSION = 'sdv-semantic-metadata-a1/1.0.0'
MODELS = ('ctgan', 'tvae', 'copulagan', 'gaussian_copula')
SDTYPES = {'continuous': 'numerical', 'count': 'numerical',
           'categorical': 'categorical', 'binary': 'categorical'}
RESERVED = {'class', '__target__', '__label__'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def policy_record():
    return {'version': VERSION, 'implementation_sha256': sha(__file__),
        'library_versions': {p: importlib.metadata.version(p) for p in ('sdv', 'rdt')},
        'mapping': SDTYPES, 'keys': 'none; no detection',
        'categories': 'learned separately from each model input; no external domains supplied',
        'values': 'unchanged; numerical Float representation permits fractional counts',
        'transformer_parameters': 'existing synthesizer defaults unchanged'}


def schema_record(names, semantic_types):
    names = list(names)
    require(names and len(set(names)) == len(names) and not RESERVED.intersection(names),
            'SDV requires unique predictors without target/label columns')
    for name in names:
        require(semantic_types.get(name) in SDTYPES, f'Ambiguous or missing SDV semantic type: {name}')
    selected = {name: semantic_types[name] for name in names}
    columns = {name: {'sdtype': SDTYPES[selected[name]]} for name in names}
    return {'policy': policy_record(), 'column_order': names, 'semantic_types': selected,
            'metadata': {'METADATA_SPEC_VERSION': 'V1', 'tables': {'table': {'columns': columns}}}}


def context(params, model):
    from src.revision_generator_parameters import revised_context
    revised = revised_context(params, model)
    if revised is None:
        return None
    require(model in MODELS, 'Not a revised SDV generator')
    schema = revised.get('sdv_semantic_schema')
    require(isinstance(schema, dict), 'Missing revised SDV semantic schema; legacy metadata is incompatible')
    expected = schema_record(schema['column_order'], schema['semantic_types'])
    require(schema == expected, 'Incompatible revised SDV schema/policy/runtime')
    return expected


def metadata_from_schema(schema, frame=None, metadata_class=None):
    if metadata_class is None:
        from sdv.metadata import Metadata
        metadata_class = Metadata
    expected = schema_record(schema['column_order'], schema['semantic_types'])
    require(schema == expected, 'Incompatible SDV metadata schema')
    metadata = metadata_class.load_from_dict(expected['metadata'])
    metadata.validate()
    if frame is not None:
        require(list(frame.columns) == expected['column_order'], 'SDV predictor names/order differ from revised schema')
        metadata.validate_table(frame, table_name='table')
    return metadata


def fit_metadata(params, model, frame, directory, legacy_class):
    schema = context(params, model)
    if schema is None:
        return legacy_class.detect_from_dataframe(frame)
    require(not any((directory/name).exists() for name in
                    ('metadata.json', 'revision_sdv_metadata.json', 'revision_sdv_binding.json', f'{model}_model.pkl')),
            'Revised SDV fit cannot overwrite/rebind existing artifacts')
    metadata = metadata_from_schema(schema, frame, legacy_class)
    from src.revision_a1_runtime import frame_record
    write(directory/'revision_sdv_metadata.json', {'schema': schema, 'effective_metadata': metadata.to_dict(),
          'effective_parameters': params, 'model': model, 'train_input': frame_record(frame),
          'transformation_fit_scope': 'this model input only; metadata shares types, never fitted transformers'})
    return metadata


def bind_model(params, model, directory):
    if context(params, model) is None:
        return
    write(directory/'revision_sdv_binding.json', {'model': model, 'effective_parameters': params,
          'files_sha256': {name: sha(directory/name) for name in
                          ('revision_sdv_metadata.json', 'metadata.json', f'{model}_model.pkl')}})


def check_sample(params, model, directory):
    schema = context(params, model)
    if schema is None:
        return
    file = directory/'revision_sdv_binding.json'
    require(file.is_file(), 'Missing revised SDV binding; historical/unidentified artifact rejected')
    binding = json.loads(file.read_text())
    require(binding['model'] == model and binding['effective_parameters'] == params,
            'Incompatible revised SDV artifact parameters or schema')
    expected_names = {'revision_sdv_metadata.json', 'metadata.json', f'{model}_model.pkl'}
    require(set(binding['files_sha256']) == expected_names, 'Incomplete SDV artifact binding')
    for name, expected in binding['files_sha256'].items():
        require((directory/name).is_file() and sha(directory/name) == expected, 'SDV artifact hash mismatch: '+name)
    record = json.loads((directory/'revision_sdv_metadata.json').read_text())
    require(record['schema'] == schema and record['effective_parameters'] == params,
            'Effective SDV metadata differs from requested schema')
    from sdv.metadata import Metadata
    require(Metadata.load_from_json(directory/'metadata.json').to_dict() == record['effective_metadata'],
            'Saved SDV metadata differs from effective metadata')
