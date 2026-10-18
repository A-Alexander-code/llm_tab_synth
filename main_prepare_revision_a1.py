"""Prepare and verify isolated A1 inputs. Never imports samplers or evaluators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from config.const import (
    DATA_REVISION_A1_DIR, LOGGING_CONFIG, PARAMS_FILE, REVISION_A1_CODE_FILES,
    REVISION_A1_SOURCE_FILES, ROOT_DIR,
)
from src.logging_utils import setup_logging
from src.revision_a1 import digest, encode, load_source, prepare, verify_saved, write_json


def provenance(config: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    code = {str(p.relative_to(ROOT_DIR)): digest(p) for p in REVISION_A1_CODE_FILES}
    identity = {"config": config, "source_sha256": sources, "code_sha256": code,
                "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                            "PyYAML": importlib.metadata.version("PyYAML")}}
    return {**identity, "fingerprint": hashlib.sha256(encode(identity)).hexdigest()}


def run(verify_only: bool) -> Path:
    config = json.loads(PARAMS_FILE.read_text())["revision_a1"]
    if set(config["datasets"]) != set(REVISION_A1_SOURCE_FILES):
        raise ValueError("Only the four authorized source datasets are permitted")
    if config["group_split"] != "classwise_subset_sum_v1" or config["mode_tie_break"] != "smallest_numeric_value":
        raise ValueError("Unsupported preparation policy")
    if config["count_medians"] != "retain_fractional_values_as_float64":
        raise ValueError("Unsupported count median policy")
    if not 0 < config["test_fraction"] < 1:
        raise ValueError("Invalid test fraction")
    before = {name: digest(path) for name, path in REVISION_A1_SOURCE_FILES.items()}
    info = provenance(config, before)
    output = DATA_REVISION_A1_DIR.joinpath(info["fingerprint"])
    if verify_only:
        if not output.is_dir():
            raise FileNotFoundError(f"No outputs for the current inputs/code: {output}")
        manifest = json.loads(output.joinpath("run_manifest.json").read_text())
        if manifest["identity"] != info or not manifest["passed"]:
            raise ValueError("Run provenance mismatch or previous failure")
        for name, expected_hash in manifest["run_artifact_sha256"].items():
            if digest(output.joinpath(name)) != expected_hash:
                raise ValueError(f"Run-level artifact changed: {name}")
        logger = logging.getLogger("revision_a1")
    else:
        # No overwrite/reuse, even when a previous run was interrupted.
        output.mkdir(parents=True, exist_ok=False)
        logger = setup_logging(LOGGING_CONFIG, output.joinpath("preparation.log"), "sampler")
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        for dataset, spec in config["datasets"].items():
            try:
                data = load_source(REVISION_A1_SOURCE_FILES[dataset], spec)
                if data.source_hash != before[dataset]:
                    raise ValueError("Source changed during read")
            except Exception as exc:
                failures.append({"dataset": dataset, "error": str(exc)})
                logger.exception("Dataset rejected: %s", dataset)
                continue
            for seed in config["seeds"]:
                try:
                    payloads, summary = prepare(data, spec, seed, config["test_fraction"])
                    summary["dataset"] = dataset
                    summary["source_hash_unchanged"] = digest(REVISION_A1_SOURCE_FILES[dataset]) == before[dataset]
                    if not summary["source_hash_unchanged"]:
                        raise ValueError("Source changed during preparation")
                    directory = output.joinpath(dataset, f"seed_{seed}")
                    if not verify_only:
                        directory.mkdir(parents=True, exist_ok=False)
                        for name, payload in payloads.items():
                            with directory.joinpath(name).open("xb") as stream:
                                stream.write(payload)
                        write_json(directory.joinpath("provenance.json"), info)
                        payloads["provenance.json"] = encode(info)
                        write_json(directory.joinpath("artifact_hashes.json"), {
                            name: digest(directory.joinpath(name)) for name in payloads})
                    else:
                        payloads["provenance.json"] = encode(info)
                    verify_saved(directory, payloads, data)
                    summaries.append(summary)
                    logger.info("Verified %s seed=%d train=%d test=%d columns=%d",
                                dataset, seed, summary["train"]["rows"], summary["test"]["rows"], summary["n_features"])
                except Exception as exc:
                    failures.append({"dataset": dataset, "seed": str(seed), "error": str(exc)})
                    logger.exception("Preparation rejected: %s seed=%d", dataset, seed)
    finally:
        after = {name: digest(path) for name, path in REVISION_A1_SOURCE_FILES.items()}
        if before != after:
            failures.append({"dataset": "source_integrity", "error": "Source hashes changed"})
        if not verify_only:
            write_json(output.joinpath("summary.json"), summaries)
            write_json(output.joinpath("failures.json"), failures)
            with output.joinpath("summary.csv").open("x", newline="") as stream:
                fields = ["dataset", "seed", "train_rows", "test_rows", "train_groups", "test_groups",
                          "train_class_0", "train_class_1", "test_class_0", "test_class_1", "columns",
                          "train_missing_before", "test_missing_before", "train_missing_kept", "test_missing_kept",
                          "postprocess_shared_patterns", "postprocess_test_matches"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for s in summaries:
                    writer.writerow(dict(zip(fields, [s["dataset"], s["seed"], s["train"]["rows"], s["test"]["rows"],
                        s["train"]["groups"], s["test"]["groups"], s["train"]["class_counts"].get(0, 0),
                        s["train"]["class_counts"].get(1, 0), s["test"]["class_counts"].get(0, 0),
                        s["test"]["class_counts"].get(1, 0), s["n_features"],
                        s["train"]["missing_before_selection"], s["test"]["missing_before_selection"],
                        s["train"]["missing_before_imputation_kept"], s["test"]["missing_before_imputation_kept"],
                        s["postprocessing_overlap"]["unique_predictor_patterns"],
                        s["postprocessing_overlap"]["test_rows_matching_train"]])))
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
            write_json(output.joinpath("run_manifest.json"), {
                "identity": info, "git_head": git.stdout.strip(), "code_identity": "working-file SHA256, including uncommitted code",
                "dependencies": {"python": sys.version, "platform": platform.platform(),
                    "PyYAML": importlib.metadata.version("PyYAML"), "numeric_engine": "Python standard library: decimal/statistics/random/struct"},
                "sources_before": before, "sources_after": after, "source_hashes_unchanged": before == after,
                "passed": not failures, "successful_combinations": len(summaries), "failures": failures,
                "run_artifact_sha256": {n: digest(output.joinpath(n)) for n in ["summary.json", "summary.csv", "failures.json"]},
                "status": "Data prepared only; A1 experimental rerun remains pending"})
    if failures:
        raise RuntimeError(f"{len(failures)} failures; inspect failures.json for preparation runs")
    logger.info("All %d combinations verified; sources intact. Outputs: %s", len(summaries), output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read and reproduce existing outputs without writing anything")
    args = parser.parse_args()
    if args.verify_only:
        logging.basicConfig(level=logging.INFO)
    run(args.verify_only)
