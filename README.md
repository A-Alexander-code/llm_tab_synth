# Synthetic Tabular Data Generation Benchmark

For the eight-generator final campaign and its reproducibility evidence, see the
[R2.G11 central guide](docs/reproducibility_final_campaign.md). The general workflow below also covers historical experiments.

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
| ForestDiffusion (`forestdiffusion`) | Historical: full labelled dataset. Revised A1: two classwise models; real execution blocked pending validation |
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

The plotter replaces the historical aggregated-metric figures with all nine
figures from `plotting_results_ver4.ipynb`. It reads `results/results_long.csv`
and, for `timing`, `results/generation_timing.csv`. Paths are relative to the
repository root, independently of the current working directory.

```bash
# All nine figures, each in PNG and PDF
python main_plotter.py all

# All three utility profiles, or just one classifier
python main_plotter.py utility
python main_plotter.py utility --classifier random_forest

python main_plotter.py privacy
python main_plotter.py tradeoff --classifier logistic_regression
python main_plotter.py detectability
python main_plotter.py timing

# Restrict datasets and classifier
python main_plotter.py all --datasets heart_framingham cervical_cancer --classifier xgboost
```

| Command | Notebook outputs |
| --- | --- |
| `utility` | `utility_profiles_{classifier}` in `figures/utility_profiles_final/` |
| `privacy` | `figure_3_privacy_evaluation` in `figures/privacy_profiles_final/` |
| `tradeoff` | XGBoost: `figure_4_privacy_utility_tradeoff_xgboost` in `figures/privacy_utility_tradeoff_final/`; Random Forest / Logistic Regression: `figure_S3_privacy_utility_tradeoff_random_forest` / `figure_S4_privacy_utility_tradeoff_logistic_regression` in `figures/supplementary_final/` |
| `detectability` | `figure_S5_detectability_profiles` in `figures/supplementary_final/` |
| `timing` | `figure_S6_computational_time` in `figures/supplementary_final/` |
| `all` | All of the above; nine figures with the default classifier selection |

`--classifier {all,xgboost,random_forest,logistic_regression}` is available for
`utility`, `tradeoff`, and `all`, and defaults to `all`. `--datasets` is available
for every command and defaults to the notebook's four datasets, configured in
`plots.datasets`. The eight generators, their order, colors, metrics, reference
lines, and seed statistics reproduce the notebook. Utility scales are calculated
separately for each classifier, as in the notebook code. The historical
`--output-ir` option is removed because these figures do not select an output IR.

All commands accept `--verbose` and `--title` / `--no-title`. By default there is
no overall figure title, matching the notebook; panel titles remain present.
`plots.png_dpi` controls PNG resolution (default 300). Both output formats retain
the notebook filenames, so repeated commands replace matching outputs.

The notebook's data validation is retained, including duplicate detection,
metric-specific status and fold filters, seed pairing for trade-offs, and timing
completeness across both classes and seeds 13, 23, 42, 69, and 88. Timing sums the
two class durations per seed before computing mean and sample standard deviation.
Boxes show mean ± sample standard deviation bounded by extrema, with min–max
whiskers; they are not quartile boxplots. Insufficient-resolution MIA observations
and annotations follow the notebook. No training or evaluation runs are launched.

## Reproducibility and caching

The following caching behavior describes historical mode. The explicit
`--revision-a1-run` mode consumes only the pinned, verified A1 preparation,
uses separate model/result paths, and refuses model reuse or overwrites.
See [A1 integration and no-training verification](docs/revision_a1_integration.md)
for commands, evidence, limitations, and the ForestDiffusion execution block.
The [follow-up audit and utility preparation](docs/revision_followup.md) documents
the nine other generators, fixed-size balanced real baselines, validity reports,
prediction records, and a pilot proposal that has not been executed.
The [revised ForestDiffusion protocol](docs/revision_forest.md) implements classwise
routing, reversible types, isolated RNG continuation, and simulated verification
on all 20 pinned partitions. Real ForestDiffusion execution remains blocked.
The [minimal real validation](docs/revision_forest_real_validation.md) records three
small fits, exact sequence comparisons with reloads, and supervised resource use.
The full ForestDiffusion benchmark remains blocked pending review.
The [nine-generator incident report](docs/revision_generator_incidents.md) records
revised parameter names and REaLTabFormer seed propagation, with parser, sampling
budget, padding, and metadata changes separated as pending decisions.

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
