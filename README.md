# Synthetic Tabular Data Generation Benchmark

This repository benchmarks generative models for **class-conditional synthetic
tabular data generation**, with particular emphasis on imbalanced binary
classification datasets. It provides an end-to-end workflow to:

1. create reproducible stratified train/test splits;
2. train ten tabular generative models and generate samples for both classes;
3. evaluate fidelity, downstream utility, and privacy at several output
   imbalance ratios; and
4. produce publication-ready utility, privacy, and privacy-utility trade-off
   figures.

The imbalance ratio (IR) is defined as
`minority samples / majority samples`. Models are trained on the original
training split; the requested IR is applied only when the generated class
samples are assembled for evaluation.

## Supported models

| Model | Training strategy |
| --- | --- |
| GReaT (`great`) | One model per class |
| TabuLa (`tabula`) | One model per class |
| REaLTabFormer (`realtabformer`) | One model per class |
| CTGAN (`ctgan`) | One model per class |
| TVAE (`tvae`) | One model per class |
| Gaussian Copula (`gaussian_copula`) | One model per class |
| CopulaGAN (`copulagan`) | One model per class |
| TabPFGen (`tabpfgen`) | One model trained on the full labelled dataset |
| ForestDiffusion (`forestdiffusion`) | One model trained on the full labelled dataset |
| MOSTLY AI (`mostlyai`) | One model per class |

## Repository structure

Revised A1 data preparation (four immutable original sources, group splits and
train-only preprocessing) is documented in [docs/revision_a1.md](docs/revision_a1.md).
It has a separate CLI/output tree and does not train models or overwrite prior runs.

```text
config/
  const.py                    Project paths and supported model names
  params.json                 Experiments, model hyperparameters, and plots
  datasets_description.json   Dataset column-type specifications
  logging.yaml                Sampler logging configuration
data/raw/                     Input CSV datasets
data/processed/               Reusable stratified splits
models/                       Persisted trained models
results/samples/              Generated class-conditional CSV files
results/metrics_raw/          Per-run evaluation metrics and audits
results/metrics_aggregated/   Metrics aggregated across seeds
figures/evaluation_profiles/  Generated evaluation figures
src/generators/               Generator implementations
src/evaluation/               Utility, fidelity, and privacy evaluation
src/plots/                    Plot preparation and rendering
main_sampler.py               Training and sampling entry point
main_evaluator.py             Evaluation entry point
main_plotter.py               Plotting entry point
```

## Requirements and setup

- Python 3.10 or newer
- Git (TabuLa is cloned automatically on first use)
- A CUDA-capable environment for models configured to use CUDA
- Enough disk space for model checkpoints, samples, metrics, and figures

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

This repository currently does not include a pinned dependency file. The core
pipeline uses NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML, PyTorch,
and XGBoost. Generator-specific dependencies include SDV, GReaT, TabuLa,
REaLTabFormer, TabPFGen, ForestDiffusion, and MOSTLY AI Engine. Install only the
generator packages required for the models you intend to run, together with
their compatible PyTorch/CUDA versions.

Create any missing project directories after cloning:

```bash
python scripts/setup_dirs.py
python scripts/setup_dirs.py --verbose
```

## Data and configuration

Place each input dataset at `data/raw/<dataset>.csv`. A dataset must:

- contain the target column configured by `experiment.target_column`
  (`class` by default);
- use binary integer labels `0` and `1`;
- contain numeric or numerically encoded feature columns, because the splitter
  converts the feature matrix to `float32`; and
- have enough examples of both classes for a stratified train/test split.

Register the dataset name in `experiment.datasets` in `config/params.json`.
For reliable synthetic-value post-processing, also add its numeric, categorical,
binary, and target columns to `config/datasets_description.json`.

`config/params.json` controls all experiment settings:

| Key | Meaning |
| --- | --- |
| `experiment.datasets` | Dataset names, without `.csv` |
| `experiment.models` | Models run by default |
| `experiment.seeds` | Split and experiment random seeds |
| `experiment.imbalance_ratios` | Output IR values evaluated |
| `experiment.n_samples_per_class` | Samples generated for each class |
| `experiment.target_column` | Binary target column |
| `experiment.test_split` | Fraction reserved for the test set |
| `model_params.<model>` | Model-specific training and sampling settings |
| `plots.*` | Plot datasets, model order, classifiers, titles, and PNG DPI |

## Running the pipeline

Run the commands from the repository root and in this order:

```bash
python main_sampler.py
python main_evaluator.py
python main_plotter.py all
```

The following sections describe every command-line possibility.

### 1. Training and sampling: `main_sampler.py`

The sampler creates one stratified split per dataset and seed, trains the
selected models, and writes `synthetic_class_0.csv` and
`synthetic_class_1.csv`. With no arguments it runs all datasets, models, and
seeds listed in `params.json`.

```bash
# Run the complete configured experiment
python main_sampler.py

# Run one or more models
python main_sampler.py --models great ctgan tvae

# Run one or more datasets (names do not include .csv)
python main_sampler.py --datasets heart_framingham cervical_cancer

# Run one or more seeds
python main_sampler.py --seeds 13 42

# Combine filters
python main_sampler.py \
  --models ctgan gaussian_copula \
  --datasets diabetes_prima_indian \
  --seeds 42 69

# Load experiment settings from another JSON file
python main_sampler.py --config config/params.json

# Ignore the model hash/cache and train the selected models again
python main_sampler.py --models ctgan --force-retrain

# Print all resolved model/dataset/seed combinations without running them
python main_sampler.py --dry-run

# Display CLI help
python main_sampler.py --help
```

| Option | Values and behavior |
| --- | --- |
| `--models MODEL [MODEL ...]` | Any supported model names; default: `experiment.models` |
| `--datasets DATASET [DATASET ...]` | Dataset stems from `data/raw`; default: `experiment.datasets` |
| `--seeds SEED [SEED ...]` | One or more integers; default: `experiment.seeds` |
| `--config PATH` | Configuration JSON; default: `config/params.json` |
| `--force-retrain` | Retrain even when the dataset/parameter hash and model files match |
| `--dry-run` | Preview combinations without splitting, training, or sampling |

Existing stratified splits are reused. Trained models are also reused when the
stored experiment hash matches and the expected model files exist. Sampling is
still performed when training is skipped. Each completed combination writes:

```text
data/processed/<dataset>/seed_<seed>/
models/<model>/<dataset>/seed_<seed>/...
results/samples/<model>/<dataset>/seed_<seed>/
  synthetic_class_0.csv
  synthetic_class_1.csv
  timing.json
  params_snapshot.json
logs/sampler_<timestamp>.log
```

### 2. Evaluation: `main_evaluator.py`

The evaluator loads the real split and the two synthetic class files, builds a
synthetic dataset for each requested IR, and computes:

- fidelity: univariate similarity, multivariate similarity, and real-vs-synthetic
  detectability;
- utility: TRTR and TSTR performance with logistic regression, random forest,
  and XGBoost; and
- privacy: exact matches, distance to closest record (DCR), holdout DCR,
  membership inference, attribute inference, nearest-neighbor adversarial
  accuracy (NNAA), and k-anonymity diagnostics.

With no arguments, it evaluates all datasets, models, seeds, and IR values in
the `experiment` section of `params.json`.

```bash
# Evaluate the complete configured experiment
python main_evaluator.py

# Select datasets and models
python main_evaluator.py \
  --datasets heart_framingham diabetic_retinopathy \
  --models ctgan great

# Select seeds
python main_evaluator.py --models great --seeds 13 42 88

# Evaluate one or more output imbalance ratios
python main_evaluator.py --models ctgan --ir 0.5 0.75 1.0

# Combine every selector
python main_evaluator.py \
  --datasets kidney_disease \
  --models tvae copulagan \
  --seeds 23 42 \
  --ir 0.6 0.8 1.0

# Save fitted TSTR classifiers for later inspection
python main_evaluator.py --models ctgan --save-classifiers true

# Other accepted true values are 1 and yes; false values include false, 0, and no
python main_evaluator.py --save-classifiers yes

# Preview all combinations without evaluating them
python main_evaluator.py --dry-run

# Display CLI help
python main_evaluator.py --help
```

| Option | Values and behavior |
| --- | --- |
| `--datasets DATASET [DATASET ...]` | Default: `experiment.datasets` |
| `--models MODEL [MODEL ...]` | Default: `experiment.models` |
| `--seeds SEED [SEED ...]` | One or more integers; default: `experiment.seeds` |
| `--ir IR [IR ...]` | One or more floating-point output ratios; default: `experiment.imbalance_ratios` |
| `--save-classifiers BOOL` | Save TSTR (not TRTR) `.pkl` classifiers; default: `false` |
| `--dry-run` | Log resolved combinations without computing metrics |
| `--config PATH` | Accepted by the current CLI, but see the note below |

> **Current `--config` limitation:** `main_evaluator.py` loads
> `config/params.json` before parsing arguments and does not apply the parsed
> `--config` path. Until that implementation is changed, edit the default file
> or use the selection flags above. The sampler's `--config` option does work.

The evaluator skips missing or empty synthetic sample files and records
individual metric-block failures without aborting every combination. At the
start of each dataset it replaces that dataset's master CSV, then aggregates
valid results across the selected seeds. Outputs include:

```text
results/metrics_raw/<dataset>/<model>/seed_<seed>/ir_<IR>/metrics.csv
results/metrics_raw/<dataset>/<model>/seed_<seed>/ir_<IR>/degenerate.csv
results/metrics_raw/<dataset>_master.csv
results/metrics_aggregated/<dataset>_aggregated.csv
results/classifiers/<dataset>/<model>/seed_<seed>/ir_<IR>/  # optional
logs/evaluator_<timestamp>.log
```

### 3. Plotting: `main_plotter.py`

The plotter reads `results/metrics_aggregated/<dataset>_aggregated.csv`. It
requires one of four subcommands: `utility`, `privacy`, `tradeoff`, or `all`.
Dataset availability, defaults, model order, supported classifiers, default IR,
title behavior, and PNG resolution come from the `plots` section of
`config/params.json`.

```bash
# Generate the downstream-utility figure with default settings
python main_plotter.py utility

# Choose the classifier used in utility metrics
python main_plotter.py utility --classifier random_forest

# Generate the privacy figure
python main_plotter.py privacy

# Generate the trade-off figure for one classifier and one output IR
python main_plotter.py tradeoff --classifier xgboost --output-ir 0.8

# Generate utility, privacy, and trade-off figures together
python main_plotter.py all --classifier logistic_regression --output-ir 1.0

# Restrict any subcommand to configured datasets
python main_plotter.py all \
  --datasets heart_framingham cervical_cancer kidney_disease

# Include automatically generated titles (titles are disabled by default)
python main_plotter.py utility --title

# Explicitly suppress titles, even if plots.default_title is changed to true
python main_plotter.py all --no-title

# Enable debug-level application logging
python main_plotter.py privacy --verbose

# Display general or subcommand-specific help
python main_plotter.py --help
python main_plotter.py tradeoff --help
```

Options shared by all subcommands:

| Option | Values and behavior |
| --- | --- |
| `--datasets DATASET [DATASET ...]` | Configured datasets; default: `plots.datasets` |
| `--title` / `--no-title` | Enable or disable the generated figure title |
| `--verbose` | Enable debug logging for the plotting application |

Subcommand-specific options:

| Subcommand | Additional options |
| --- | --- |
| `utility` | `--classifier {logistic_regression,random_forest,xgboost}` |
| `privacy` | None |
| `tradeoff` | `--classifier ...`, `--output-ir FLOAT` (must be greater than zero) |
| `all` | `--classifier ...`, `--output-ir FLOAT` (must be greater than zero) |

Figures are saved as both PNG and PDF under:

```text
figures/evaluation_profiles/utility/
figures/evaluation_profiles/privacy/
figures/evaluation_profiles/tradeoff/
```

## Reproducibility and caching

The following caching behavior describes historical mode. The explicit
`--revision-a1-run` mode consumes only the pinned, verified A1 preparation,
uses separate model/result paths, and refuses model reuse or overwrites.
See [A1 integration and no-training verification](docs/revision_a1_integration.md)
for commands, evidence, limitations, and the ForestDiffusion execution block.

- Each dataset/seed split is generated once with stratification and then reused.
- Training IR is never modified; requested output IR values are applied during
  evaluation.
- Model reuse is controlled by a hash of the training data and model parameters.
- `--force-retrain` bypasses the trained-model cache.
- Sampling outputs include a parameter snapshot and per-class timing data.
- Evaluation results retain dataset, model, seed, and output-IR identifiers;
  aggregated files report mean, standard deviation, minimum, maximum, valid
  count, and not-computable count.

## Extending the benchmark

All generators inherit `BaseGenerator` and implement:

```python
fit(df: pandas.DataFrame, model_dir: pathlib.Path) -> None
sample(n_samples: int, model_dir: pathlib.Path) -> pandas.DataFrame
```

To add a model, create its module in `src/generators/`, register it in
`src/generators/__init__.py`, add its name to `SUPPORTED_MODELS` in
`config/const.py`, and define its hyperparameters in `config/params.json`.
