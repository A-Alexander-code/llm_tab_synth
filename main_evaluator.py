"""
main_evaluator.py

Orquestador del pipeline de evaluación de datos sintéticos tabulares.

Lee las muestras generadas por main_sampler.py y evalúa:
    - Utilidad:  univariada, multivariada, detectabilidad, ML efficacy (TRTR/TSTR)
    - Privacidad: exact matches, DCR, MIA, attribute inference

Flujo:
    para cada dataset:
        cargar real_train, real_test
        para cada seed:
            para cada model:
                cargar synthetic_class_0.csv + synthetic_class_1.csv
                para cada output_ir:
                    construir dataset sintético final
                    evaluar utilidad + privacidad
                    guardar CSV por combinación
                    acumular en master CSV
        guardar master CSV
        generar CSV agregado

Uso:
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

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.joinpath("config").joinpath("params.json")

sys.path.insert(0, str(ROOT))

from src.utils.seed_utils import set_global_seed
from src.evaluation.builder import build_synthetic_dataset
from src.evaluation.utility import (
    compute_univariate,
    compute_multivariate,
    compute_detectability,
    compute_ml_efficacy,
)
from src.evaluation.privacy import (
    compute_exact_matches,
    compute_dcr,
    compute_dcr_holdout,
    compute_mia,
    compute_attribute_inference,
    compute_nnaa,
    compute_kanonymity,
)
from src.evaluation.aggregator import (
    build_rows,
    save_combination_csv,
    append_to_master,
    save_aggregated,
    save_degenerate_audit,
)
from src.evaluation.degenerate import (
    get_valid_columns_for_statistical_metrics,
    compute_degenerate_diagnostics,
    build_degenerate_audit_rows,
)

# ── Logging ────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir.joinpath(f"evaluator_{ts}.log")

    fmt     = "%(asctime)s [%(levelname)s] %(message)s"
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


# ── Config ─────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Path helpers ───────────────────────────────────────────────────────────

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


def _degenerate_audit_path(
    root: Path, dataset: str, model: str, seed: int, output_ir: float
) -> Path:
    return (
        root.joinpath("results")
        .joinpath("metrics_raw")
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
        .joinpath(f"ir_{output_ir:.2f}")
        .joinpath("degenerate.csv")
    )


def _classifiers_dir(
    root: Path, dataset: str, model: str, seed: int, output_ir: float
) -> Path:
    return (
        root.joinpath("results")
        .joinpath("classifiers")
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
        .joinpath(f"ir_{output_ir:.2f}")
    )


def _degenerate_audit_path(
    root: Path, dataset: str, model: str, seed: int, output_ir: float
) -> Path:
    return (
        root.joinpath("results")
        .joinpath("metrics_raw")
        .joinpath(dataset)
        .joinpath(model)
        .joinpath(f"seed_{seed}")
        .joinpath(f"ir_{output_ir:.2f}")
        .joinpath("degenerate.csv")
    )


# ── Data loading ───────────────────────────────────────────────────────────

def _load_split(processed_dir: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga X_train/X_test/y_train/y_test y reconstruye DataFrames con target."""
    meta_path = processed_dir.joinpath("feature_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_meta.json not found in {processed_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["dataset"]["feature_names"]

    X_train = np.load(str(processed_dir.joinpath("X_train.npy")))
    X_test  = np.load(str(processed_dir.joinpath("X_test.npy")))
    y_train = np.load(str(processed_dir.joinpath("y_train.npy")))
    y_test  = np.load(str(processed_dir.joinpath("y_test.npy")))

    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df  = pd.DataFrame(X_test,  columns=feature_names)
    train_df[target] = y_train.astype(int)
    test_df[target]  = y_test.astype(int)

    return train_df, test_df


def _load_synthetic(
    samples_dir: Path,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Carga synthetic_class_0.csv y synthetic_class_1.csv.
    Retorna None si:
      - alguno de los archivos no existe
      - alguno está vacío (0 bytes o sin columnas)
      - la lectura falla por cualquier motivo
    En todos los casos loguea el problema en lugar de lanzar excepción.
    """
    path_0 = samples_dir.joinpath("synthetic_class_0.csv")
    path_1 = samples_dir.joinpath("synthetic_class_1.csv")

    if not path_0.exists() or not path_1.exists():
        return None

    # Verificar que no estén vacíos antes de intentar leer
    for path in (path_0, path_1):
        if path.stat().st_size == 0:
            if logger:
                logger.warning(
                    f"Empty file (0 bytes): {path}. "
                    f"The model may have failed during sampling. Skipping."
                )
            return None

    try:
        synth_0 = pd.read_csv(path_0)
        synth_1 = pd.read_csv(path_1)
    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to read synthetic CSVs from {samples_dir}: {e}. Skipping."
            )
        return None

    # Verificar que tengan filas y columnas
    for name, df, path in (("class_0", synth_0, path_0), ("class_1", synth_1, path_1)):
        if df.empty or df.shape[1] == 0:
            if logger:
                logger.warning(
                    f"Synthetic file has no usable data ({name}): {path}. "
                    f"Shape: {df.shape}. Skipping."
                )
            return None

    return synth_0, synth_1


# ── Evaluation runner ──────────────────────────────────────────────────────

def _timed_block(
    fn,
    label: str,
    prefix: str,
    logger: logging.Logger,
    *args,
    **kwargs,
) -> list[dict]:
    """
    Ejecuta fn(*args, **kwargs), registra su duración y devuelve el resultado.
    Si falla, loguea el error y devuelve lista vacía.
    """
    t0 = datetime.now()
    try:
        result = fn(*args, **kwargs)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"{prefix}   ✓ {label:<30s} {elapsed:5.1f}s  ({len(result)} métricas)")
        return result
    except Exception:
        elapsed = (datetime.now() - t0).total_seconds()
        logger.warning(
            f"{prefix}   ✗ {label:<30s} {elapsed:5.1f}s  FAILED\n{traceback.format_exc()}"
        )
        return []


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
    classifiers_dir: Path | None = None,
    audit_path: Path | None = None,
) -> list[dict]:
    """Ejecuta todos los bloques de evaluación para una combinación dada."""
    prefix = f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}]"
    all_rows = []

    real_train_feat = real_train.drop(columns=[target], errors="ignore")
    real_test_feat  = real_test.drop(columns=[target],  errors="ignore")
    synth_feat      = synth_final.drop(columns=[target], errors="ignore")

    # ── Diagnóstico de columnas degeneradas (una sola vez, reutilizado) ───
    vcr = get_valid_columns_for_statistical_metrics(real_train_feat, synth_feat)
    if vcr.excluded_cols:
        affected = "multivariate, attribute_inference"
        logger.info(
            f"{prefix} Degenerate columns detected: "
            f"synth_constant={len(vcr.synth_report.constant)}, "
            f"real_constant={len(vcr.real_report.constant)}, "
            f"union_excluded={len(vcr.excluded_cols)} "
            f"({', '.join(vcr.excluded_cols)}) — "
            f"excluded from: {affected}"
        )
        if audit_path is not None:
            audit_rows = build_degenerate_audit_rows(vcr, dataset, model, seed, output_ir)
            save_degenerate_audit(audit_rows, audit_path)

    all_rows += _timed_block(
        compute_degenerate_diagnostics, "degenerate_diagnostics",
        prefix, logger, synth_feat, real_train_feat, vcr,
    )

    # ── Utilidad ──────────────────────────────────────────────────────────
    all_rows += _timed_block(compute_univariate,    "univariate",           prefix, logger, real_train_feat, synth_feat)
    all_rows += _timed_block(compute_multivariate,  "multivariate",         prefix, logger, real_train_feat, synth_feat)
    all_rows += _timed_block(compute_detectability, "detectability",        prefix, logger, real_train_feat, synth_feat, seed=seed)
    all_rows += _timed_block(
        compute_ml_efficacy, "ml_efficacy (TRTR+TSTR)", prefix, logger,
        real_train, real_test, synth_final,
        target=target, seed=seed, classifiers_dir=classifiers_dir,
    )

    # ── Privacidad ────────────────────────────────────────────────────────
    all_rows += _timed_block(compute_exact_matches,  "exact_matches",      prefix, logger, synth_feat, real_train_feat)
    all_rows += _timed_block(compute_dcr,            "dcr",                prefix, logger, synth_feat, real_train_feat)
    all_rows += _timed_block(compute_dcr_holdout,    "dcr_holdout",        prefix, logger, real_test_feat, real_train_feat)
    all_rows += _timed_block(compute_mia,            "mia",                prefix, logger, synth_feat, real_train_feat, real_test_feat, seed=seed)
    all_rows += _timed_block(
        compute_attribute_inference, "attribute_inference", prefix, logger,
        synth_final, real_train, real_test, target=target, seed=seed,
    )
    all_rows += _timed_block(compute_nnaa,       "nnaa",        prefix, logger, synth_feat, real_train_feat)
    all_rows += _timed_block(compute_kanonymity, "k_anonymity", prefix, logger, synth_feat, real_train_feat, real_test_feat)

    return build_rows(all_rows, dataset=dataset, model=model, seed=seed, output_ir=output_ir)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args(config: dict) -> argparse.Namespace:
    exp = config["experiment"]
    parser = argparse.ArgumentParser(description="Evaluador de datos sintéticos tabulares")
    parser.add_argument("--config",   type=str, default=str(CONFIG_PATH))
    parser.add_argument("--datasets", nargs="+", default=exp["datasets"])
    parser.add_argument("--models",   nargs="+", default=exp["models"])
    parser.add_argument("--seeds",    nargs="+", type=int, default=exp["seeds"])
    parser.add_argument("--ir",       nargs="+", type=float, default=exp["imbalance_ratios"])
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument(
        "--save-classifiers",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        metavar="BOOL",
        help=(
            "Guardar modelos TSTR entrenados como .pkl en "
            "results/classifiers/{dataset}/{model}/seed_{seed}/ir_{ir}/. "
            "Solo se persisten los modelos TSTR (no TRTR). "
            "Valores aceptados: true/false, 1/0, yes/no. "
            "Por defecto: false."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace, config: dict, logger: logging.Logger) -> None:
    target     = config["experiment"]["target_column"]
    datasets   = args.datasets
    models     = args.models
    seeds      = args.seeds
    output_irs = sorted(args.ir)

    total_combinations = len(datasets) * len(seeds) * len(models) * len(output_irs)
    run_start = datetime.now()

    logger.info("=" * 60)
    logger.info("main_evaluator started")
    logger.info(f"  Datasets:              {datasets}")
    logger.info(f"  Models:                {models}")
    logger.info(f"  Seeds:                 {seeds}")
    logger.info(f"  Output IR:             {output_irs}")
    logger.info(f"  Save classifiers:      {args.save_classifiers}")
    logger.info(f"  Total combinations:    {total_combinations}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] No evaluation will be run.")
        for ds in datasets:
            for seed in seeds:
                for model in models:
                    for ir in output_irs:
                        logger.info(f"  {ds} | seed={seed} | {model} | ir={ir:.2f}")
        return

    combo_idx  = 0
    n_errors   = 0
    n_skipped  = 0

    for dataset in datasets:
        master_path  = _master_path(ROOT, dataset)
        dataset_start = datetime.now()

        if master_path.exists():
            master_path.unlink()
            logger.info(f"[{dataset}] Cleared existing master CSV.")

        for seed in seeds:
            set_global_seed(seed)
            proc_dir = _processed_dir(ROOT, dataset, seed)

            try:
                real_train, real_test = _load_split(proc_dir, target)
                n0 = int((real_train[target] == 0).sum())
                n1 = int((real_train[target] == 1).sum())
                logger.info(
                    f"[{dataset}|seed={seed}] Split loaded — "
                    f"train={len(real_train)} (class_0={n0}, class_1={n1}), "
                    f"test={len(real_test)}"
                )
            except Exception:
                logger.error(
                    f"[{dataset}|seed={seed}] Failed to load split:\n{traceback.format_exc()}"
                )
                n_errors += 1
                combo_idx += len(models) * len(output_irs)
                continue

            for model in models:
                sdir   = _samples_dir(ROOT, model, dataset, seed)
                result = _load_synthetic(sdir, logger=logger)
                model_start = datetime.now()

                if result is None:
                    logger.warning(
                        f"[{dataset}|seed={seed}|{model}] "
                        f"Synthetic samples not found in {sdir}. Skipping."
                    )
                    n_skipped += len(output_irs)
                    combo_idx += len(output_irs)
                    continue

                synth_0, synth_1 = result
                n0_s = int((real_train[target] == 0).sum())
                n1_s = int((real_train[target] == 1).sum())
                maj_label = 0 if n0_s >= n1_s else 1
                min_label = 1 - maj_label
                synth_maj, synth_min = (synth_0, synth_1) if maj_label == 0 else (synth_1, synth_0)

                logger.info(
                    f"[{dataset}|seed={seed}|{model}] "
                    f"Synthetic loaded — class_0={len(synth_0)}, class_1={len(synth_1)}  "
                    f"(majority=class_{maj_label})"
                )

                ir_errors = 0
                for output_ir in output_irs:
                    combo_idx += 1
                    elapsed_total = (datetime.now() - run_start).total_seconds()
                    pct = 100.0 * combo_idx / total_combinations

                    logger.info(
                        f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                        f"── combo {combo_idx}/{total_combinations} ({pct:.0f}%)  "
                        f"elapsed={elapsed_total:.0f}s ──"
                    )

                    t0_combo = datetime.now()
                    try:
                        synth_final = build_synthetic_dataset(
                            synth_maj=synth_maj,
                            synth_min=synth_min,
                            output_ir=output_ir,
                            seed=seed,
                            maj_label=maj_label,
                            min_label=min_label,
                        )
                        n_maj_f = int((synth_final["class"] == maj_label).sum())
                        n_min_f = int((synth_final["class"] != maj_label).sum())
                        logger.info(
                            f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                            f"   Dataset sintético: {len(synth_final)} filas "
                            f"(maj={n_maj_f}, min={n_min_f})"
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
                            classifiers_dir=(
                                _classifiers_dir(ROOT, dataset, model, seed, output_ir)
                                if args.save_classifiers else None
                            ),
                            audit_path=_degenerate_audit_path(ROOT, dataset, model, seed, output_ir),
                        )

                        comb_path = _metrics_combination_path(ROOT, dataset, model, seed, output_ir)
                        save_combination_csv(rows, comb_path)
                        append_to_master(rows, master_path)

                        # Auditoría de columnas degeneradas
                        deg_path = _degenerate_audit_path(ROOT, dataset, model, seed, output_ir)
                        real_feat_tmp  = real_train.drop(columns=[target], errors="ignore")
                        synth_feat_tmp = synth_final.drop(columns=[target], errors="ignore")
                        vc_audit = get_valid_columns_for_statistical_metrics(real_feat_tmp, synth_feat_tmp)
                        audit_rows = build_degenerate_audit_rows(vc_audit, dataset, model, seed, output_ir)
                        save_degenerate_audit(audit_rows, deg_path)

                        combo_elapsed = (datetime.now() - t0_combo).total_seconds()
                        logger.info(
                            f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                            f"   Done in {combo_elapsed:.1f}s — {len(rows)} metric rows saved."
                        )

                    except Exception:
                        combo_elapsed = (datetime.now() - t0_combo).total_seconds()
                        logger.error(
                            f"[{dataset}|seed={seed}|{model}|ir={output_ir:.2f}] "
                            f"   Combination failed after {combo_elapsed:.1f}s:\n{traceback.format_exc()}"
                        )
                        n_errors  += 1
                        ir_errors += 1

                model_elapsed = (datetime.now() - model_start).total_seconds()
                logger.info(
                    f"[{dataset}|seed={seed}|{model}] "
                    f"Finished — {len(output_irs)} IRs in {model_elapsed:.1f}s"
                    + (f", {ir_errors} errors" if ir_errors else "")
                )

        # Agregación final por dataset
        dataset_elapsed = (datetime.now() - dataset_start).total_seconds()
        try:
            agg_path = _aggregated_path(ROOT, dataset)
            agg_df   = save_aggregated(master_path, agg_path)
            logger.info(
                f"[{dataset}] Aggregation done — {len(agg_df)} rows → {agg_path}  "
                f"(dataset total: {dataset_elapsed:.1f}s)"
            )
        except Exception:
            logger.error(f"[{dataset}] Aggregation failed:\n{traceback.format_exc()}")
            n_errors += 1

    total_elapsed = (datetime.now() - run_start).total_seconds()
    logger.info("=" * 60)
    logger.info(
        f"main_evaluator finished — "
        f"{combo_idx} combinations in {total_elapsed:.1f}s  "
        f"(errors={n_errors}, skipped={n_skipped})"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    config = _load_config(CONFIG_PATH)
    logger = _setup_logging(ROOT.joinpath("logs"))
    args   = parse_args(config)
    run(args, config, logger)