from __future__ import annotations

import math
import unittest

import pandas as pd

from src.evaluation.cooccurrence import compute_cooccurrence_metrics, evaluate_cooccurrence_metrics


def _cooc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [40, 40, 60, 60],
            "bmi": [20, 35, 20, 35],
            "smoker": ["no", "yes", "no", "yes"],
            "class": [0, 0, 1, 1],
        }
    )


class CooccurrenceTests(unittest.TestCase):
    def test_identical_data_has_zero_distance(self) -> None:
        real_train = _cooc_df()
        synthetic = real_train.copy()

        metrics = compute_cooccurrence_metrics(real_train, synthetic)

        self.assertEqual(metrics["cooc_l1_mean"], 0.0)
        self.assertEqual(metrics["cooc_jsd_mean"], 0.0)

    def test_detects_distribution_shift(self) -> None:
        real_train = _cooc_df()
        synthetic = pd.DataFrame(
            {
                "age": [40, 40, 40, 40],
                "bmi": [35, 35, 35, 35],
                "smoker": ["yes", "yes", "yes", "yes"],
                "class": [0, 0, 1, 1],
            }
        )

        metrics = compute_cooccurrence_metrics(real_train, synthetic)

        self.assertGreater(metrics["cooc_l1_mean"], 0.0)

    def test_too_few_valid_columns_is_not_computable(self) -> None:
        real_train = pd.DataFrame({"age": [40, 50, 60], "class": [0, 1, 0]})
        synthetic = pd.DataFrame({"age": [40, 45, 50], "class": [0, 1, 0]})

        rows = evaluate_cooccurrence_metrics(
            real_train=real_train,
            synthetic=synthetic,
            dataset="ds",
            model="ctgan",
            seed=13,
            output_ir=1.0,
        )

        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "not_computable" for row in rows))
        self.assertEqual({row["reason"] for row in rows}, {"insufficient_valid_features"})
        self.assertTrue(all(math.isnan(float(row["value"])) for row in rows))


if __name__ == "__main__":
    unittest.main()
