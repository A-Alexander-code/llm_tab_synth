from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from main_extended_evaluator import resolve_metrics


ROOT = Path(__file__).resolve().parent.parent


def _make_temp_config(tmp_dir: str) -> Path:
    tmp_path = Path(tmp_dir)
    source_path = ROOT.joinpath("config").joinpath("params.json")
    config = json.loads(source_path.read_text(encoding="utf-8"))
    config["experiment"]["datasets"] = ["tiny_ds"]
    config["experiment"]["models"] = ["great", "ctgan", "tabula"]
    config["experiment"]["seeds"] = [42]
    config["experiment"]["imbalance_ratios"] = [1.0]
    config["extended_evaluation"]["output_dir"] = str(tmp_path.joinpath("metrics_extended_raw"))
    config["extended_evaluation"]["master_path"] = str(tmp_path.joinpath("metrics_extended_master.csv"))
    config["extended_evaluation"]["aggregated_path"] = str(tmp_path.joinpath("metrics_extended_aggregated.csv"))

    config_path = tmp_path.joinpath("params.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class MainExtendedEvaluatorCliTests(unittest.TestCase):
    def test_resolve_metrics_all_alias(self) -> None:
        self.assertEqual(
            resolve_metrics(["all"]),
            [
                "record_similarity",
                "semantic_similarity",
                "ngram_overlap",
                "memorization",
                "cooccurrence",
                "tradeoff",
            ],
        )

    def test_resolve_metrics_all_with_temperature(self) -> None:
        self.assertEqual(
            resolve_metrics(["all"], include_temperature=True),
            [
                "record_similarity",
                "semantic_similarity",
                "ngram_overlap",
                "memorization",
                "cooccurrence",
                "tradeoff",
                "temperature",
            ],
        )

    def test_temperature_without_include_flag_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _make_temp_config(tmp_dir)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT.joinpath("main_extended_evaluator.py")),
                    "--config",
                    str(config_path),
                    "--metrics",
                    "temperature",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("require --include-temperature", result.stderr)

    def test_dry_run_creates_no_outputs_and_llm_only_filters_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _make_temp_config(tmp_dir)
            output_dir = Path(tmp_dir).joinpath("metrics_extended_raw")
            master_path = Path(tmp_dir).joinpath("metrics_extended_master.csv")
            aggregated_path = Path(tmp_dir).joinpath("metrics_extended_aggregated.csv")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT.joinpath("main_extended_evaluator.py")),
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--llm-only",
                    "--metrics",
                    "all",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse(output_dir.exists())
            self.assertFalse(master_path.exists())
            self.assertFalse(aggregated_path.exists())
            self.assertIn("great", result.stdout)
            self.assertIn("tabula", result.stdout)
            self.assertNotIn("ctgan", result.stdout)


if __name__ == "__main__":
    unittest.main()
