from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.temperature import compute_temperature_sensitivity_from_aggregated


def _temp_rows(metric: str, values: list[float]) -> pd.DataFrame:
    temperatures = [0.2, 0.7, 1.0, 1.3][: len(values)]
    return pd.DataFrame(
        {
            "dataset": ["ds"] * len(values),
            "model": ["great"] * len(values),
            "seed": [42] * len(values),
            "output_ir": [1.0] * len(values),
            "metric": [metric] * len(values),
            "classifier": [None] * len(values),
            "value": values,
            "status": ["ok"] * len(values),
            "temperature": temperatures,
        }
    )


class TemperatureTests(unittest.TestCase):
    def test_metrics_are_computed_for_multiple_temperatures(self) -> None:
        df = pd.concat(
            [
                _temp_rows("record_sim_high_ratio_095", [0.1, 0.2, 0.3, 0.4]),
                _temp_rows("memrisk_output_score", [0.6, 0.5, 0.4, 0.3]),
            ],
            ignore_index=True,
        )

        result = compute_temperature_sensitivity_from_aggregated(
            df,
            metrics_to_track=["record_sim_high_ratio_095", "memrisk_output_score"],
            temperatures=[0.2, 0.7, 1.0, 1.3],
        )

        self.assertFalse(result.empty)
        self.assertIn("temp_metric_range", set(result["metric"]))
        self.assertIn("temp_metric_std", set(result["metric"]))
        self.assertIn("temp_metric_cv", set(result["metric"]))
        self.assertIn("temp_metric_slope", set(result["metric"]))
        self.assertIn("temp_memrisk_range", set(result["metric"]))
        self.assertIn("temp_memrisk_slope", set(result["metric"]))
        self.assertIn("temp_memrisk_max", set(result["metric"]))
        self.assertIn("temp_memrisk_min", set(result["metric"]))

    def test_one_available_temperature_is_not_computable(self) -> None:
        df = _temp_rows("record_sim_high_ratio_095", [0.2])

        result = compute_temperature_sensitivity_from_aggregated(
            df,
            metrics_to_track=["record_sim_high_ratio_095"],
            temperatures=[0.2],
        )

        self.assertEqual(set(result["status"]), {"not_computable"})
        self.assertEqual(set(result["reason"]), {"insufficient_temperatures"})

    def test_all_nan_values_are_not_computable(self) -> None:
        df = _temp_rows(
            "record_sim_high_ratio_095",
            [float("nan"), float("nan"), float("nan"), float("nan")],
        )

        result = compute_temperature_sensitivity_from_aggregated(
            df,
            metrics_to_track=["record_sim_high_ratio_095"],
            temperatures=[0.2, 0.7, 1.0, 1.3],
        )

        self.assertEqual(set(result["status"]), {"not_computable"})
        self.assertEqual(set(result["reason"]), {"all_nan_values"})


if __name__ == "__main__":
    unittest.main()
