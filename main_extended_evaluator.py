"""
Run extended evaluation metrics for synthetic tabular data.

Usage examples:
    python main_extended_evaluator.py
    python main_extended_evaluator.py --metrics all
    python main_extended_evaluator.py --metrics llm_privacy cooccurrence tradeoff
    python main_extended_evaluator.py --metrics all --include-temperature
    python main_extended_evaluator.py --llm-only --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.joinpath("config").joinpath("params.json")

sys.path.insert(0, str(ROOT))

from config.const import (
    EXPECTED_METRICS,
    LLM_ONLY_MODELS,
    METRIC_ALIASES,
    METRIC_REGISTRY,
    TEMPERATURE_TRACKING,
)
from src.evaluation.aggregator import append_to_master, build_rows, save_aggregated, save_combination_csv
from src.evaluation.builder import build_synthetic_dataset
from src.evaluation.cooccurrence import evaluate_cooccurrence_metrics
from src.evaluation.llm_privacy import evaluate_llm_privacy_metrics
from src.evaluation.temperature import compute_temperature_sensitivity_from_aggregated
from src.evaluation.tradeoff import compute_tradeoff_metrics_from_rows
from src.utils.seed_utils import set_global_seed


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _setup_logging(log_dir: Path, verbose: bool, dry_run: bool) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path = None
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir.joinpath(f"extended_evaluator_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger("extended_evaluator")
    if log_path is not None:
        logger.info("Logging to %s", log_path)
    return logger


def resolve_metrics(requested_metrics: list[str], include_temperature: bool = False) -> list[str]:
    """Resolve metric aliases with deterministic ordering."""
    resolved: list[str] = []

    for metric_name in requested_metrics:
        if metric_name in METRIC_ALIASES:
            expanded = METRIC_ALIASES[metric_name]
            for item in expanded:
                if item not in resolved:
                    resolved.append(item)
            if metric_name == "all" and include_temperature and "temperature" not in resolved:
                resolved.append("temperature")
            continue

        if metric_name == "temperature" and not include_temperature:
            raise ValueError("temperature metrics require --include-temperature")
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric_name}")
        if metric_name not in resolved:
            resolved.append(metric_name)

    if include_temperature and "all" not in requested_metrics and "temperature" in requested_metrics:
        if "temperature" not in resolved:
            resolved.append("temperature")

    return resolved


def _default_metrics(config: dict[str, Any]) -> list[str]:
    return config["extended_evaluation"]["default_metrics"]


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    experiment = config["experiment"]
    extended = config["extended_evaluation"]

    parser = argparse.ArgumentParser(description="Extended synthetic tabular evaluator")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--datasets", nargs="+", default=experiment["datasets"])
    parser.add_argument("--models", nargs="+", default=experiment["models"])
    parser.add_argument("--seeds", nargs="+", type=int, default=experiment["seeds"])
    parser.add_argument("--ir", nargs="+", type=float, default=experiment["imbalance_ratios"])
    parser.add_argument("--metrics", nargs="+", default=_default_metrics(config))
    parser.add_argument("--llm-only", action="store_true", default=extended.get("llm_only_default", False))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--workers", type=int, default=int(extended.get("workers", 1)))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--include-temperature", action="store_true")
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=extended.get("temperature", {}).get("temperatures", []),
    )
    parser.add_argument("--run-base-first", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config = _load_config(Path(pre_args.config))
    parser = build_parser(config)
    args = parser.parse_args(argv)
    args.metrics = resolve_metrics(args.metrics, include_temperature=args.include_temperature)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args, config


def _processed_dir(root: Path, dataset: str, seed: int) -> Path:
    return root.joinpath("data").joinpath("processed").joinpath(dataset).joinpath(f"seed_{seed}")


def _samples_dir(root: Path, model: str, dataset: str, seed: int) -> Path:
    return root.joinpath("results").joinpath("samples").joinpath(model).joinpath(dataset).joinpath(f"seed_{seed}")


def _temperature_samples_dir(
    root: Path,
    model: str,
    dataset: str,
    seed: int,
    temperature: float,
) -> Path:
    return (
        _samples_dir(root, model, dataset, seed)
        .joinpath(f"temperature_{temperature}")
    )


def _base_metrics_path(root: Path, dataset: str, model: str, seed: int, output_ir: float) -> Path:
    return (
        root.joinpath("results")
        .joinpath("metrics_raw")
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
        .joinpath(f"ir_{output_ir:.2f}")
        .joinpath("metrics.csv")
    )


def _base_master_path(root: Path, dataset: str) -> Path:
    return root.joinpath("results").joinpath("metrics_raw").joinpath(f"{dataset}_master.csv")


def _extended_combination_path(
    root: Path,
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    temperature: float | None = None,
    output_dir: str = "results/metrics_extended_raw",
) -> Path:
    base_dir = (
        root.joinpath(output_dir)
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
    )
    if temperature is not None:
        base_dir = base_dir.joinpath(f"temperature_{temperature}")
    return base_dir.joinpath(f"ir_{output_ir:.2f}").joinpath("extended_metrics.csv")


def _resolve_output_path(root: Path, relative_path: str) -> Path:
    return root.joinpath(relative_path)


def _load_split(processed_dir: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_path = processed_dir.joinpath("feature_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_meta.json not found in {processed_dir}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    feature_names = meta["dataset"]["feature_names"]
    x_train = np.load(str(processed_dir.joinpath("X_train.npy")))
    x_test = np.load(str(processed_dir.joinpath("X_test.npy")))
    y_train = np.load(str(processed_dir.joinpath("y_train.npy")))
    y_test = np.load(str(processed_dir.joinpath("y_test.npy")))

    train_df = pd.DataFrame(x_train, columns=feature_names)
    test_df = pd.DataFrame(x_test, columns=feature_names)
    train_df[target] = y_train.astype(int)
    test_df[target] = y_test.astype(int)
    return train_df, test_df


def _load_synthetic_pair(samples_dir: Path, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path_0 = samples_dir.joinpath("synthetic_class_0.csv")
    path_1 = samples_dir.joinpath("synthetic_class_1.csv")
    if not path_0.exists() or not path_1.exists():
        return None
    if path_0.stat().st_size == 0 or path_1.stat().st_size == 0:
        logger.warning("Empty synthetic file under %s", samples_dir)
        return None
    try:
        synth_0 = pd.read_csv(path_0)
        synth_1 = pd.read_csv(path_1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read synthetic samples from %s: %s", samples_dir, exc)
        return None
    if synth_0.empty or synth_1.empty:
        logger.warning("Synthetic data is empty under %s", samples_dir)
        return None
    return synth_0, synth_1


def _determine_class_layout(
    real_train: pd.DataFrame,
    synth_0: pd.DataFrame,
    synth_1: pd.DataFrame,
    target_column: str,
) -> tuple[int, int, pd.DataFrame, pd.DataFrame]:
    n0 = int((real_train[target_column] == 0).sum())
    n1 = int((real_train[target_column] == 1).sum())
    maj_label = 0 if n0 >= n1 else 1
    min_label = 1 - maj_label
    synth_maj, synth_min = (synth_0, synth_1) if maj_label == 0 else (synth_1, synth_0)
    return maj_label, min_label, synth_maj, synth_min


def _load_base_metric_rows(
    root: Path,
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
) -> list[dict[str, Any]]:
    combination_path = _base_metrics_path(root, dataset, model, seed, output_ir)
    if combination_path.exists():
        return pd.read_csv(combination_path).to_dict("records")

    master_path = _base_master_path(root, dataset)
    if not master_path.exists():
        return []

    master_df = pd.read_csv(master_path)
    filtered = master_df[
        (master_df["dataset"] == dataset)
        & (master_df["model"] == model)
        & (master_df["seed"] == seed)
        & np.isclose(master_df["output_ir"].astype(float), output_ir)
    ].copy()
    return filtered.to_dict("records")


def _rows_with_context(
    rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    if "dataset" in rows[0]:
        return rows
    return build_rows(rows, dataset=dataset, model=model, seed=seed, output_ir=output_ir)


def _not_rows_for_metric(metric_name: str, reason: str) -> list[dict[str, Any]]:
    block = METRIC_REGISTRY[metric_name]["block"]
    return [
        {
            "block": block,
            "metric": raw_metric,
            "classifier": None,
            "value": float("nan"),
            "status": "not_computable",
            "reason": reason,
        }
        for raw_metric in EXPECTED_METRICS.get(metric_name, [])
    ]


def _temperature_not_rows(
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    tracked_metrics: list[str],
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tracked_metric in tracked_metrics:
        if tracked_metric == "memrisk_output_score":
            metric_names = [
                "temp_memrisk_range",
                "temp_memrisk_slope",
                "temp_memrisk_max",
                "temp_memrisk_min",
            ]
            classifier = None
        else:
            metric_names = [
                "temp_metric_range",
                "temp_metric_std",
                "temp_metric_cv",
                "temp_metric_slope",
            ]
            classifier = tracked_metric
        for metric_name in metric_names:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "output_ir": output_ir,
                    "block": METRIC_REGISTRY["temperature"]["block"],
                    "metric": metric_name,
                    "classifier": classifier,
                    "value": float("nan"),
                    "status": "not_computable",
                    "reason": reason,
                }
            )
    return rows


def _evaluate_stable_metrics(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    target_column: str,
    metric_names: list[str],
    base_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cooccurrence_config = config["extended_evaluation"]["cooccurrence"]
    tradeoff_config = config["extended_evaluation"]["tradeoff"]

    llm_metric_subset = [
        metric_name
        for metric_name in metric_names
        if metric_name in {"record_similarity", "semantic_similarity", "ngram_overlap", "memorization"}
    ]
    if llm_metric_subset:
        rows.extend(
            evaluate_llm_privacy_metrics(
                real_train=real_train,
                real_test=real_test,
                synthetic=synthetic,
                dataset=dataset,
                model=model,
                seed=seed,
                output_ir=output_ir,
                target_column=target_column,
                metrics=llm_metric_subset,
                settings=config["extended_evaluation"]["llm_privacy"],
            )
        )

    if "cooccurrence" in metric_names:
        rows.extend(
            evaluate_cooccurrence_metrics(
                real_train=real_train,
                synthetic=synthetic,
                dataset=dataset,
                model=model,
                seed=seed,
                output_ir=output_ir,
                target_column=target_column,
                numeric_strategy=cooccurrence_config.get("numeric_strategy", "quantile"),
                n_bins=int(cooccurrence_config.get("n_bins", 4)),
                top_k=int(cooccurrence_config.get("top_k", 10)),
            )
        )

    if "tradeoff" in metric_names:
        rows.extend(
            compute_tradeoff_metrics_from_rows(
                metric_rows=base_rows + _rows_with_context(rows, dataset, model, seed, output_ir),
                utility_weight=float(tradeoff_config.get("utility_weight", 0.5)),
                privacy_weight=float(tradeoff_config.get("privacy_weight", 0.5)),
            )
        )

    return _rows_with_context(rows, dataset, model, seed, output_ir)


def _tracked_metric_names(selected_metrics: list[str]) -> list[str]:
    tracked: list[str] = []
    stable_selected = [metric_name for metric_name in selected_metrics if metric_name != "temperature"]
    for metric_name in stable_selected:
        for tracked_metric in TEMPERATURE_TRACKING.get(metric_name, []):
            if tracked_metric not in tracked:
                tracked.append(tracked_metric)
    if not tracked:
        tracked = ["memrisk_output_score"]
    return tracked


def _evaluate_temperature_metrics(
    root: Path,
    logger: logging.Logger,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    seed: int,
    output_ir: float,
    target_column: str,
    stable_metric_names: list[str],
    temperatures: list[float],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracked_metrics = _tracked_metric_names(stable_metric_names)
    temperature_eval_metric_names = stable_metric_names or ["memorization"]
    if len(temperatures) < 2:
        return [], _temperature_not_rows(
            dataset,
            model,
            seed,
            output_ir,
            tracked_metrics,
            "insufficient_temperatures",
        )

    per_temperature_rows: list[dict[str, Any]] = []
    base_rows = _load_base_metric_rows(root, dataset, model, seed, output_ir)

    for temperature in temperatures:
        temperature_samples_dir = _temperature_samples_dir(root, model, dataset, seed, temperature)
        synth_pair = _load_synthetic_pair(temperature_samples_dir, logger)
        if synth_pair is None:
            continue

        synth_0, synth_1 = synth_pair
        maj_label, min_label, synth_maj, synth_min = _determine_class_layout(
            real_train, synth_0, synth_1, target_column
        )
        synth_final = build_synthetic_dataset(
            synth_maj=synth_maj,
            synth_min=synth_min,
            output_ir=output_ir,
            seed=seed,
            maj_label=maj_label,
            min_label=min_label,
        )

        temperature_rows = _evaluate_stable_metrics(
            real_train=real_train,
            real_test=real_test,
            synthetic=synth_final,
            dataset=dataset,
            model=model,
            seed=seed,
            output_ir=output_ir,
            target_column=target_column,
            metric_names=temperature_eval_metric_names,
            base_rows=base_rows,
            config=config,
        )
        for row in temperature_rows:
            row_with_temperature = dict(row)
            row_with_temperature["temperature"] = float(temperature)
            per_temperature_rows.append(row_with_temperature)

        save_combination_csv(
            temperature_rows,
            _extended_combination_path(
                root,
                dataset,
                model,
                seed,
                output_ir,
                temperature=temperature,
                output_dir=config["extended_evaluation"]["output_dir"],
            ),
        )

    if not per_temperature_rows:
        reason = f"missing_temperature_samples:{temperatures[0]}"
        return [], _temperature_not_rows(dataset, model, seed, output_ir, tracked_metrics, reason)

    sensitivity_df = compute_temperature_sensitivity_from_aggregated(
        aggregated_df=pd.DataFrame(per_temperature_rows),
        metrics_to_track=tracked_metrics,
        temperatures=temperatures,
    )
    return per_temperature_rows, sensitivity_df.to_dict("records")


def _build_combinations(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    output_irs: list[float],
) -> list[tuple[str, str, int, float]]:
    return [
        (dataset, model, seed, output_ir)
        for dataset in datasets
        for model in models
        for seed in seeds
        for output_ir in output_irs
    ]


def _run_base_evaluator(args: argparse.Namespace, logger: logging.Logger) -> None:
    cmd = [
        sys.executable,
        str(ROOT.joinpath("main_evaluator.py")),
        "--config",
        args.config,
        "--datasets",
        *args.datasets,
        "--models",
        *args.models,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--ir",
        *[str(value) for value in args.ir],
    ]
    logger.info("Running base evaluator first: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run(args: argparse.Namespace, config: dict[str, Any], logger: logging.Logger) -> None:
    target_column = config["experiment"]["target_column"]
    datasets = list(args.datasets)
    models = list(args.models)
    if args.llm_only:
        models = [model for model in models if model in LLM_ONLY_MODELS]
    seeds = list(args.seeds)
    output_irs = sorted(float(value) for value in args.ir)
    selected_metrics = list(args.metrics)
    stable_metric_names = [metric_name for metric_name in selected_metrics if metric_name != "temperature"]

    extended_config = config["extended_evaluation"]
    master_path = _resolve_output_path(ROOT, extended_config["master_path"])
    aggregated_path = _resolve_output_path(ROOT, extended_config["aggregated_path"])
    append_enabled = bool(extended_config.get("append", True)) if args.append is None else bool(args.append)
    skip_existing = bool(extended_config.get("skip_existing", True)) and not args.force and not args.overwrite

    combinations = _build_combinations(datasets, models, seeds, output_irs)

    logger.info("=" * 72)
    logger.info("main_extended_evaluator started")
    logger.info("Datasets: %s", datasets)
    logger.info("Models: %s", models)
    logger.info("Seeds: %s", seeds)
    logger.info("Output IRs: %s", output_irs)
    logger.info("Metrics: %s", selected_metrics)
    logger.info("Temperatures: %s", args.temperatures)
    logger.info("Workers: %s", args.workers)
    logger.info("Total combinations: %s", len(combinations))
    logger.info("=" * 72)

    if args.run_base_first:
        _run_base_evaluator(args, logger)

    if args.dry_run:
        for dataset, model, seed, output_ir in combinations:
            logger.info(
                "[DRY RUN] %s | %s | seed=%s | ir=%.2f | metrics=%s",
                dataset,
                model,
                seed,
                output_ir,
                ",".join(selected_metrics),
            )
        return

    if not append_enabled or args.overwrite:
        if master_path.exists():
            master_path.unlink()
        if aggregated_path.exists():
            aggregated_path.unlink()

    errors = 0
    skipped = 0

    for combo_index, (dataset, model, seed, output_ir) in enumerate(combinations, start=1):
        prefix = f"[{dataset}|{model}|seed={seed}|ir={output_ir:.2f}]"
        combination_path = _extended_combination_path(
            ROOT,
            dataset,
            model,
            seed,
            output_ir,
            output_dir=extended_config["output_dir"],
        )

        if skip_existing and combination_path.exists():
            logger.info("%s Skipping existing extended metrics at %s", prefix, combination_path)
            skipped += 1
            continue

        try:
            set_global_seed(seed)
            real_train, real_test = _load_split(_processed_dir(ROOT, dataset, seed), target_column)
            base_rows = _load_base_metric_rows(ROOT, dataset, model, seed, output_ir)

            synth_pair = _load_synthetic_pair(_samples_dir(ROOT, model, dataset, seed), logger)
            if synth_pair is None:
                rows = []
                for metric_name in stable_metric_names:
                    rows.extend(_not_rows_for_metric(metric_name, "missing_synthetic_samples"))
                rows = _rows_with_context(rows, dataset, model, seed, output_ir)
                if "temperature" in selected_metrics:
                    rows.extend(
                        _temperature_not_rows(
                            dataset,
                            model,
                            seed,
                            output_ir,
                            _tracked_metric_names(stable_metric_names),
                            f"missing_temperature_samples:{args.temperatures[0]}" if args.temperatures else "insufficient_temperatures",
                        )
                    )
                save_combination_csv(rows, combination_path)
                append_to_master(rows, master_path)
                logger.warning("%s Missing synthetic samples", prefix)
                continue

            synth_0, synth_1 = synth_pair
            maj_label, min_label, synth_maj, synth_min = _determine_class_layout(
                real_train, synth_0, synth_1, target_column
            )
            synth_final = build_synthetic_dataset(
                synth_maj=synth_maj,
                synth_min=synth_min,
                output_ir=output_ir,
                seed=seed,
                maj_label=maj_label,
                min_label=min_label,
            )

            rows = _evaluate_stable_metrics(
                real_train=real_train,
                real_test=real_test,
                synthetic=synth_final,
                dataset=dataset,
                model=model,
                seed=seed,
                output_ir=output_ir,
                target_column=target_column,
                metric_names=stable_metric_names,
                base_rows=base_rows,
                config=config,
            )

            if "temperature" in selected_metrics:
                _, temperature_rows = _evaluate_temperature_metrics(
                    root=ROOT,
                    logger=logger,
                    real_train=real_train,
                    real_test=real_test,
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    output_ir=output_ir,
                    target_column=target_column,
                    stable_metric_names=stable_metric_names,
                    temperatures=[float(value) for value in args.temperatures],
                    config=config,
                )
                rows.extend(temperature_rows)

            save_combination_csv(rows, combination_path)
            append_to_master(rows, master_path)
            logger.info(
                "%s Saved %s extended metric rows (%s/%s)",
                prefix,
                len(rows),
                combo_index,
                len(combinations),
            )
        except Exception:  # noqa: BLE001
            errors += 1
            logger.error("%s FAILED\n%s", prefix, traceback.format_exc())
            if args.fail_fast:
                raise

    aggregated_df = save_aggregated(master_path, aggregated_path)
    logger.info("Aggregation complete: %s rows -> %s", len(aggregated_df), aggregated_path)
    logger.info(
        "main_extended_evaluator finished (errors=%s, skipped=%s, processed=%s)",
        errors,
        skipped,
        len(combinations) - skipped,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args, config = parse_args(argv)
        logger = _setup_logging(ROOT.joinpath("logs"), verbose=args.verbose, dry_run=args.dry_run)
        run(args, config, logger)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
