"""Generate synthetic samples using multiple generative models.

Usage:
  python main_sampler.py                              # Run all models
  python main_sampler.py --models great ctgan         # Specific models
  python main_sampler.py --datasets diabetes-retin    # Specific datasets
  python main_sampler.py --force-retrain              # Force retraining
  python main_sampler.py --dry-run                    # Preview combinations
"""

from __future__ import annotations
 
import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
 
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
 
from config.const import (
    DATA_RAW_DIR,
    LOGGING_CONFIG,
    PARAMS_FILE,
    PARAMS_SNAPSHOT_NAME,
    SUPPORTED_MODELS,
    TIMING_FILENAME,
    get_model_class_dir,
    get_processed_dir,
    get_samples_dir,
    LOGS_DIR,
)
from src.data_splitter import is_split_ready, split_and_save, load_split, reconstruct_dataframe
from src.generators import get_generator, GENERATOR_REGISTRY
from src.hashing import compute_experiment_hash
from src.io_utils import load_params, save_dataframe, save_json
from src.logging_utils import setup_logging
from src.model_registry import is_model_trained, register_trained_model
from src.timing import Timer, save_timing
from src.postprocessing import load_type_spec, postprocess_synthetic, postprocess_summary
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples using multiple generative models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help=(
            f"Models to run. Default: all supported models.\n"
            f"Available: {SUPPORTED_MODELS}"
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="DATASET",
        help="Dataset names (without extension) located in data/raw/. Default: all in params.json.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        metavar="SEED",
        help="Random seeds. Default: all seeds in params.json.",
    )
    parser.add_argument(
        "--config", type=Path, default=PARAMS_FILE,
        metavar="PATH",
        help=f"Path to params.json. Default: {PARAMS_FILE}",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help=("Force retraining even if the model already exists. In revised mode, replace models, samples "
              "and training workspaces for the selected dataset/seed/model combinations, including previous identities."),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned experiment combinations without executing.",
    )
    parser.add_argument("--revision-a1-run", help="Explicit pinned A1 run ID; strict isolated data and outputs")
    return parser.parse_args()
 
 
def resolve_combinations(
    params: dict,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[int]]:
    """Resolve the final model, dataset, and seed lists."""
    models = args.models   or params["experiment"]["models"]
    datasets = args.datasets or params["experiment"]["datasets"]
    seeds = args.seeds    or params["experiment"]["seeds"]
 
    invalid = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}. Valid: {SUPPORTED_MODELS}")
 
    return models, datasets, seeds
 
 
def build_train_df_for_class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label: int,
    meta: dict,
) -> pd.DataFrame:
    """Rebuild the class-specific training DataFrame."""
    mask = y_train == label
    X_class = X_train[mask]
    y_class = y_train[mask]
    df = reconstruct_dataframe(X_class, y_class, meta)
    return df.drop(columns=[meta["dataset"]["target_column"]])
 
 
def build_full_train_df(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta: dict,
) -> pd.DataFrame:
    """Rebuild the full training DataFrame with the target renamed to '__target__'."""
    df = reconstruct_dataframe(X_train, y_train, meta)
    target_col = meta["dataset"]["target_column"]
    return df.rename(columns={target_col: "__target__"})
 
 
_FULL_DATASET_MODELS = {"tabpfgen", "forestdiffusion"}
 
def run(args: argparse.Namespace) -> None:
    if getattr(args, "revision_a1_run", None):
        from src.revision_a1_runtime import run_sampler
        return run_sampler(args)
    params   = load_params(args.config)
    models, datasets, seeds = resolve_combinations(params, args)
 
    n_samples = params["experiment"]["n_samples_per_class"]
    test_size = params["experiment"]["test_split"]
    target_col = params["experiment"]["target_column"]
    labels = [0, 1]
 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR.joinpath(f"sampler_{ts}.log")
    logger = setup_logging(LOGGING_CONFIG, log_file, logger_name="sampler")
 
    logger.info("=" * 60)
    logger.info("main_sampler started at %s", ts)
    logger.info("Config: %s", args.config)
    logger.info("Models: %s", models)
    logger.info("Datasets: %s", datasets)
    logger.info("Seeds: %s", seeds)
    logger.info("n_samples_per_class: %d", n_samples)
    logger.info("force_retrain: %s", args.force_retrain)
    logger.info("dry_run: %s", args.dry_run)
    logger.info("=" * 60)
 
    total_combinations = len(models) * len(datasets) * len(seeds)
    logger.info("Total combinations: %d", total_combinations)
 
    if args.dry_run:
        print("\n[DRY RUN] Planned combinations:")
        for dataset in datasets:
            for seed in seeds:
                for model in models:
                    print(f"  model={model:20s}  dataset={dataset}  seed={seed}")
        print(f"\nTotal: {total_combinations} combinations")
        return
 
    for dataset in datasets:
        raw_path = DATA_RAW_DIR.joinpath(f"{dataset}.csv")
        if not raw_path.exists():
            logger.error("Dataset not found: %s", raw_path)
            continue
 
        from src.io_utils import load_dataset
        df_raw = load_dataset(raw_path)
        logger.info("Loaded dataset '%s': %d rows, %d cols", dataset, *df_raw.shape)
 
        description_path = ROOT_DIR.joinpath("config").joinpath("datasets_description.json")
        type_spec = load_type_spec(dataset, description_path, raw_path)
        logger.info(
            "[%s] Type spec loaded: %s",
            dataset, postprocess_summary(type_spec)
        )
 
        for seed in seeds:
            processed_dir = get_processed_dir(dataset, seed)
 
            if is_split_ready(processed_dir):
                logger.info(
                    "[%s | seed=%d] Split already exists. Loading from %s",
                    dataset, seed, processed_dir,
                )
            else:
                logger.info(
                    "[%s | seed=%d] Generating stratified split (test_size=%.2f)...",
                    dataset, seed, test_size,
                )
                meta = split_and_save(
                    df=df_raw,
                    target_col=target_col,
                    output_dir=processed_dir,
                    test_size=test_size,
                    seed=seed,
                )
                logger.info(
                    "[%s | seed=%d] Split saved. Train: %d | Test: %d | IR_train: %.4f",
                    dataset, seed,
                    meta["split"]["n_train"],
                    meta["split"]["n_test"],
                    meta["class_distribution"]["train"]["imbalance_ratio"],
                )
 
            X_train, X_test, y_train, y_test, meta = load_split(processed_dir)
 
            for model_name in models:
                model_params = params["model_params"][model_name]
                uses_full_df = model_name in _FULL_DATASET_MODELS
 
                logger.info(
                    "[%s | seed=%d | %s] Starting...",
                    dataset, seed, model_name,
                )
 
                exp_hash = compute_experiment_hash(
                    pd.DataFrame(X_train), model_params
                )
 
                samples_dir = get_samples_dir(model_name, dataset, seed)
                samples_dir.mkdir(parents=True, exist_ok=True)
                timing_path = samples_dir.joinpath(TIMING_FILENAME)
 
                if uses_full_df:
                    class_dir = get_model_class_dir(model_name, dataset, seed, label=0)
                    class_dir = class_dir.parent

                    already_trained = (
                        not args.force_retrain
                        and is_model_trained(class_dir, exp_hash)
                    )
 
                    if not already_trained:
                        train_df = build_full_train_df(X_train, y_train, meta)
                        generator = get_generator(model_name, model_params)
 
                        logger.info(
                            "[%s | seed=%d | %s] Training on full dataset (%d rows)...",
                            dataset, seed, model_name, len(train_df),
                        )
                        with Timer("training") as t_train:
                            generator.fit(train_df, class_dir)
 
                        save_timing(
                            {f"training_s": round(t_train.elapsed, 4)},
                            timing_path,
                        )
                        register_trained_model(
                            class_dir, exp_hash,
                            metadata={
                                "model": model_name, "dataset": dataset,
                                "seed": seed, "trained_on": "full_dataset",
                            },
                        )
                        logger.info(
                            "[%s | seed=%d | %s] Training completed in %.2fs",
                            dataset, seed, model_name, t_train.elapsed,
                        )
                    else:
                        generator = get_generator(model_name, model_params)
                        logger.info(
                            "[%s | seed=%d | %s] Model already trained. Skipping.",
                            dataset, seed, model_name,
                        )
 
                    for label in labels:
                        _generate_and_save(
                            generator=generator,
                            model_dir=class_dir,
                            label=label,
                            n_samples=n_samples,
                            samples_dir=samples_dir,
                            timing_path=timing_path,
                            model_name=model_name,
                            dataset=dataset,
                            seed=seed,
                            logger=logger,
                            type_spec=type_spec,
                        )
 
                else:
                    for label in labels:
                        class_dir = get_model_class_dir(model_name, dataset, seed, label)
 
                        already_trained = (
                            not args.force_retrain
                            and is_model_trained(class_dir, exp_hash)
                        )
 
                        generator = get_generator(model_name, model_params)
 
                        if not already_trained:
                            train_df = build_train_df_for_class(
                                X_train, y_train, label, meta
                            )
                            logger.info(
                                "[%s | seed=%d | %s | class=%d] Training (%d rows)...",
                                dataset, seed, model_name, label, len(train_df),
                            )
                            with Timer("training") as t_train:
                                generator.fit(train_df, class_dir)
 
                            save_timing(
                                {f"training_class_{label}_s": round(t_train.elapsed, 4)},
                                timing_path,
                            )
                            register_trained_model(
                                class_dir, exp_hash,
                                metadata={
                                    "model": model_name, "dataset": dataset,
                                    "seed": seed, "label": label,
                                },
                            )
                            logger.info(
                                "[%s | seed=%d | %s | class=%d] Training done in %.2fs",
                                dataset, seed, model_name, label, t_train.elapsed,
                            )
                        else:
                            logger.info(
                                "[%s | seed=%d | %s | class=%d] Already trained. Skipping.",
                                dataset, seed, model_name, label,
                            )
 
                        _generate_and_save(
                            generator=generator,
                            model_dir=class_dir,
                            label=label,
                            n_samples=n_samples,
                            samples_dir=samples_dir,
                            timing_path=timing_path,
                            model_name=model_name,
                            dataset=dataset,
                            seed=seed,
                            logger=logger,
                            type_spec=type_spec,
                        )
 
                snapshot_path = samples_dir.joinpath(PARAMS_SNAPSHOT_NAME)
                save_json(
                    {
                        "timestamp": ts,
                        "model": model_name,
                        "dataset": dataset,
                        "seed": seed,
                        "model_params": model_params,
                        "experiment": params["experiment"],
                    },
                    snapshot_path,
                )
                logger.info(
                    "[%s | seed=%d | %s] Params snapshot saved to %s",
                    dataset, seed, model_name, snapshot_path,
                )
 
    logger.info("main_sampler finished.")
 
 
def _generate_and_save(
    generator,
    model_dir: Path,
    label: int,
    n_samples: int,
    samples_dir: Path,
    timing_path: Path,
    model_name: str,
    dataset: str,
    seed: int,
    logger: logging.Logger,
    type_spec=None,
) -> None:
    """Generate class samples, optionally postprocess them, and save them."""
    logger.info(
        "[%s | seed=%d | %s | class=%d] Generating %d samples...",
        dataset, seed, model_name, label, n_samples,
    )
    with Timer("sampling") as t_sample:
        synth_df = generator.sample(n_samples, model_dir)
 
    if "__label__" in synth_df.columns:
        synth_df = synth_df[synth_df["__label__"] == label].drop(
            columns=["__label__"]
        ).reset_index(drop=True)
 
    if type_spec is not None:
        synth_df = postprocess_synthetic(synth_df, type_spec)
 
    synth_df["class"] = label
    out_path = samples_dir.joinpath(f"synthetic_class_{label}.csv")
    save_dataframe(synth_df, out_path)
 
    save_timing(
        {f"sampling_class_{label}_s": round(t_sample.elapsed, 4)},
        timing_path,
    )
    logger.info(
        "[%s | seed=%d | %s | class=%d] %d samples saved to %s (%.2fs)",
        dataset, seed, model_name, label,
        len(synth_df), out_path, t_sample.elapsed,
    )
 
 
if __name__ == "__main__":
    args = parse_args()
    run(args)
 
