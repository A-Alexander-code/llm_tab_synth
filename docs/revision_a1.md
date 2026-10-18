# A1 revised data preparation

Only the four files in `data/original_source/` configured in `config/const.py`
are read. No historical prepared data, partitions, models, or results are used.
All experiment settings and explicit per-column mappings/types are in the
`revision_a1` section of `config/params.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_revision_a1.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 main_prepare_revision_a1.py
PYTHONDONTWRITEBYTECODE=1 python3 main_prepare_revision_a1.py --verify-only
```

Python 3.10+ and the project's PyYAML logging dependency suffice. Preparation
uses decimal arithmetic and the standard library, without model dependencies.
Matrices use standard NPY v1.0 little-endian float64 (`X`) and int32 (`y`). No
scaling, clipping, rounding of observed values, or count-median rounding is done.
Fractional count medians remain floating-point values; the schema retains their
count semantics. The fitted preprocessing object is JSON, reloadable through
`Preprocessor.from_dict`; no executable pickle is necessary.

Sources are read with explicit comma-separated headers and exact column order.
Framingham recognizes only `NA`, Cervical only `?`; other missing surprises
stop preparation. Pima maps zeros to missing only in its five configured
measurements. Target codes 0/1 are preserved. Their clinical meanings have not
been authenticated against source codebooks in this checkout, especially the
generic Retinopathy `class`; metadata does not invent meanings.

All rows are retained. Group identity hashes the ordered original predictors
after name/missing normalization using exact canonical decimal values. It never
includes target. Contradictory labels in a group stop that dataset. Row IDs use
source SHA256 and one-based data-record position (header excluded). They never
enter X. A source hash is checked before, during, and after preparation.

For each seed, sort group hashes and shuffle with `random.Random(seed)`.
Independently within each class, subset-sum over whole group sizes chooses the
reachable test row count nearest `floor(n_class * 0.2 + 0.5)`, keeping both
classes on both sides. Ties prefer fewer test rows; first discovered subset-sum
predecessors resolve equal-size subset choices. Class 0 is processed first.
This is approximate group stratification, not a claim of identical proportions.
Only row counts and labels choose the split, never predictive performance.
Rows within each matrix retain source-position order. Groups are never discarded.

Preprocessing receives X_train only. Cervical drops columns with train missing
fraction >= 0.60 separately per seed. Entirely missing retained train columns
stop preparation. Medians fill continuous/count columns; modes fill explicit
binary/categorical columns, breaking ties with the smallest numeric value.
Train-observed categories are recorded; no categories are inferred from test.
The fixed education domain is a schema validation rule, not a fitted encoder;
unseen-in-train numeric categories are retained without remapping. No encoding
or scaling is applied. Cervical's other diagnostic predictors remain, marked
as requiring documentation of availability at prediction time.

Outputs live under `data/processed_revision_a1/<fingerprint>/<dataset>/seed_N/`:

- X/y matrices, column metadata and effective class/group counts/proportions;
- `partition.json`: matrix positions, source rows and groups;
- `preprocessing.json`: retained names, train missing counts/fractions, fitted
  values, strategies and train-observed categories;
- `schema.json`, `source_audit.json`, `verification.json`, `provenance.json`;
- `artifact_hashes.json`: SHA256 of all per-combination data/metadata artifacts.

The run root contains `summary.csv`, `summary.json`, `failures.json`,
`run_manifest.json`, and operational logs. The manifest records dependency/runtime
versions, Git HEAD, working-code hashes, configuration, source hashes before/after,
and hashes of summary/failure artifacts. Manifest/hash inventory files do not
hash themselves; operational logs are not data artifacts.

The output fingerprint includes the relevant working source/test code, configuration,
source content, Python version and implementation. Changed inputs/code/configuration
create a different destination. An existing destination is never overwritten or
silently reused. Use `--verify-only` to reproduce it entirely in memory, compare
all artifact bytes/hashes, reload preprocessing and replay saved source positions.
This verification mode performs no writes. Incomplete/failed runs are not accepted.

Verification includes independent train-statistic calculations, fixed-split test
feature interventions, test-label flips/shuffles, matrix alignment/finiteness,
repeatability, preservation of all rows, disjoint source rows/groups and both
classes. Counts of identical predictor patterns after selection/imputation are
reported separately, without changing the split retrospectively. Adversarial
unit tests cover failure paths, fractional medians, mode ties, threshold equality,
invalid headers, contradictory targets and artifact corruption.

The existing sampler/evaluator are deliberately not connected to these outputs.
All models and downstream metrics must later consume the same revised partitions.
No generator/classifier is trained here. A1 is not closed: experimental reruns,
clinical target codebooks, subject-level independence (no IDs), and temporal
availability of Cervical diagnostic predictors remain pending.
