"""Evaluate synthetic tabular samples against utility and privacy metrics.

Usage:
    python main_evaluator.py
    python main_evaluator.py --datasets diabetes-retin --models ctgan
    python main_evaluator.py --models great --seeds 42
    python main_evaluator.py --ir 0.5 0.75 1.0
    python main_evaluator.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.joinpath("config").joinpath("params.json")

sys.path.insert(0, str(ROOT))

from src.utils.seed_utils import set_global_seed
from src.evaluation.builder import build_synthetic_dataset
from src.evaluation.utility import (compute_univariate, 
                                    compute_multivariate, 
                                    compute_detectability, 
                                    compute_ml_efficacy,)
from src.evaluation.privacy import (compute_exact_matches, 
                                    compute_dcr, 
                                    compute_dcr_holdout, 
                                    compute_mia, 
                                    compute_attribute_inference,)
from src.evaluation.aggregator import (build_rows, 
                                       save_combination_csv, 
                                       append_to_master,
                                       save_aggregated,)


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir.joinpath(f"evaluator_{ts}.log")

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("evaluator")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _processed_dir(root: Path, dataset: str, seed: int) -> Path:
    return root.joinpath("data").joinpath("processed").joinpath(dataset).joinpath(f"seed_{seed}")


def _samples_dir(root: Path, model: str, dataset: str, seed: int) -> Path:
    return root.joinpath("results").joinpath("samples").joinpath(model).joinpath(dataset).joinpath(f"seed_{seed}")


def _metrics_combination_path(
    root: Path, dataset: str, model: str, seed: int, output_ir: float
) -> Path:
    return (
        root.joinpath("results")
        .joinpath("metrics_raw")
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
        .joinpath(f"ir_{output_ir:.2f}")
        .joinpath("metrics.csv")
    )


def _master_path(root: Path, dataset: str) -> Path:
    return root.joinpath("results").joinpath("metrics_raw").joinpath(f"{dataset}_master.csv")


def _aggregated_path(root: Path, dataset: str) -> Path:
    return root.joinpath("results").joinpath("metrics_aggregated").joinpath(f"{dataset}_aggregated.csv")


def _load_split(processed_dir: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test arrays and rebuild DataFrames with the target column."""
    meta_path = processed_dir.joinpath("feature_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_meta.json not found in {processed_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["dataset"]["feature_names"]

    X_train = np.load(str(processed_dir.joinpath("X_train.npy")))
    X_test = np.load(str(processed_dir.joinpath("X_test.npy")))
    y_train = np.load(str(processed_dir.joinpath("y_train.npy")))
    y_test = np.load(str(processed_dir.joinpath("y_test.npy")))

    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df  = pd.DataFrame(X_test,  columns=feature_names)
    train_df[target] = y_train.astype(int)
    test_df[target]  = y_test.astype(int)

    return train_df, test_df


def _load_synthetic(samples_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load synthetic class files, or return None if they are missing."""
    path_0 = samples_dir.joinpath("synthetic_class_0.csv")
    path_1 = samples_dir.joinpath("synthetic_class_1.csv")

    if not path_0.exists() or not path_1.exists():
        return None

    synth_0 = pd.read_csv(path_0)
    synth_1 = pd.read_csv(path_1)
    return synth_0, synth_1


def _evaluate_combination(
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    synth_final: pd.DataFrame,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    target: str,
    logger: logging.Logger,
) -> list[dict]:
    """Run all evaluation blocks for one dataset/model/seed/IR combination."""
    all_rows = []

    # Drop target for feature-only metrics.
    real_train_feat = real_train.drop(columns=[target], errors="ignore")
    real_test_feat = real_test.drop(columns=[target],  errors="ignore")
    synth_feat = synth_final.drop(columns=[target], errors="ignore")

    try:
        all_rows += compute_univariate(real_train_feat, synth_feat)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] univariate failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_multivariate(real_train_feat, synth_feat)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] multivariate failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_detectability(real_train_feat, synth_feat, seed=seed)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] detectability failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_ml_efficacy(real_train, real_test, synth_final, target=target, seed=seed)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] ml_efficacy failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_exact_matches(synth_feat, real_train_feat)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] exact_matches failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_dcr(synth_feat, real_train_feat)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] dcr failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_dcr_holdout(real_test_feat, real_train_feat)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] dcr_holdout failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_mia(synth_feat, real_train_feat, real_test_feat, seed=seed)
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] mia failed:\n{traceback.format_exc()}")

    try:
        all_rows += compute_attribute_inference(
            synth_final, real_train, real_test, target=target, seed=seed
        )
    except Exception:
        logger.warning(f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}] attribute_inference failed:\n{traceback.format_exc()}")

    return build_rows(all_rows, dataset=dataset, model=model, seed=seed, output_ir=output_ir)


def parse_args(config: dict) -> argparse.Namespace:
    exp = config["experiment"]
    parser = argparse.ArgumentParser(description="Evaluador de datos sintéticos tabulares")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--datasets", nargs="+", default=exp["datasets"])
    parser.add_argument("--models", nargs="+", default=exp["models"])
    parser.add_argument("--seeds", nargs="+", type=int, default=exp["seeds"])
    parser.add_argument("--ir", nargs="+", type=float, default=exp["imbalance_ratios"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, config: dict, logger: logging.Logger) -> None:
    target = config["experiment"]["target_column"]
    datasets = args.datasets
    models = args.models
    seeds = args.seeds
    output_irs = sorted(args.ir)

    total = len(datasets) * len(seeds) * len(models) * len(output_irs)
    logger.info("=" * 60)
    logger.info("main_evaluator started")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Models: {models}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output IR: {output_irs}")
    logger.info(f"Total combinations: {total}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] No evaluation will be run.")
        for ds in datasets:
            for seed in seeds:
                for model in models:
                    for ir in output_irs:
                        logger.info(f"  {ds} | seed={seed} | {model} | ir={ir:.2f}")
        return

    for dataset in datasets:
        master_path = _master_path(ROOT, dataset)

        # Rebuild the master CSV on each full run.
        if master_path.exists():
            master_path.unlink()
            logger.info(f"[{dataset}] Cleared existing master CSV.")

        n_errors = 0

        for seed in seeds:
            set_global_seed(seed)

            proc_dir = _processed_dir(ROOT, dataset, seed)
            try:
                real_train, real_test = _load_split(proc_dir, target)
                logger.info(
                    f"[{dataset}|seed={seed}] Split loaded — "
                    f"train={len(real_train)}, test={len(real_test)}"
                )
            except Exception:
                logger.error(f"[{dataset}|seed={seed}] Failed to load split:\n{traceback.format_exc()}")
                n_errors += 1
                continue

            for model in models:
                sdir = _samples_dir(ROOT, model, dataset, seed)
                result = _load_synthetic(sdir)

                if result is None:
                    logger.warning(
                        f"[{dataset}|seed={seed}|{model}] Synthetic samples not found in {sdir}. Skipping."
                    )
                    n_errors += 1
                    continue

                synth_0, synth_1 = result
                logger.info(
                    f"[{dataset}|seed={seed}|{model}] Synthetic loaded — "
                    f"class_0={len(synth_0)}, class_1={len(synth_1)}"
                )

                # Match majority/minority classes to the real train split.
                n0 = len(real_train[real_train[target] == 0])
                n1 = len(real_train[real_train[target] == 1])
                if n0 >= n1:
                    synth_maj, synth_min = synth_0, synth_1
                else:
                    synth_maj, synth_min = synth_1, synth_0

                for output_ir in output_irs:
                    logger.info(
                        f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] Evaluating..."
                    )
                    try:
                        synth_final = build_synthetic_dataset(
                            synth_maj=synth_maj,
                            synth_min=synth_min,
                            output_ir=output_ir,
                            seed=seed,
                        )

                        rows = _evaluate_combination(
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            output_ir=output_ir,
                            synth_final=synth_final,
                            real_train=real_train,
                            real_test=real_test,
                            target=target,
                            logger=logger,
                        )

                        comb_path = _metrics_combination_path(ROOT, dataset, model, seed, output_ir)
                        save_combination_csv(rows, comb_path)

                        append_to_master(rows, master_path)

                        logger.info(
                            f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                            f"Done — {len(rows)} metric rows saved."
                        )

                    except Exception:
                        logger.error(
                            f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                            f"Evaluation failed:\n{traceback.format_exc()}"
                        )
                        n_errors += 1

        try:
            agg_path = _aggregated_path(ROOT, dataset)
            agg_df = save_aggregated(master_path, agg_path)
            logger.info(
                f"[{dataset}] Aggregated results saved to {agg_path} "
                f"({len(agg_df)} rows)."
            )
        except Exception:
            logger.error(f"[{dataset}] Aggregation failed:\n{traceback.format_exc()}")
            n_errors += 1

    logger.info("=" * 60)
    logger.info(f"main_evaluator finished. Total errors: {n_errors}")
    logger.info("=" * 60)


if __name__ == "__main__":
    config = _load_config(CONFIG_PATH)
    logger = _setup_logging(ROOT.joinpath("logs"))
    args = parse_args(config)
    run(args, config, logger)
