from __future__ import annotations

import math
import unittest

from src.evaluation.tradeoff import compute_tradeoff_metrics_from_rows


def _row(metric: str, value: float, classifier: str | None = None) -> dict:
    return {
        "dataset": "ds",
        "model": "great",
        "seed": 42,
        "output_ir": 1.0,
        "block": "block",
        "metric": metric,
        "classifier": classifier,
        "value": value,
        "status": "ok",
        "reason": None,
    }


class TradeoffTests(unittest.TestCase):
    def test_computes_weighted_scores(self) -> None:
        rows = [
            _row("trtr_aucroc", 0.8, "xgboost"),
            _row("tstr_aucroc", 0.6, "xgboost"),
            _row("trtr_auprc", 0.5, "xgboost"),
            _row("tstr_auprc", 0.4, "xgboost"),
            _row("trtr_f1", 0.5, "xgboost"),
            _row("tstr_f1", 0.45, "xgboost"),
            _row("exact_matches_ratio", 0.1),
            _row("dcr_p5", 0.4),
            _row("dcr_holdout_p5", 0.5),
            _row("record_sim_high_ratio_095", 0.2),
            _row("semantic_sim_high_ratio_095", 0.1),
            _row("memrisk_output_score", 0.3),
            _row("ngram2_rare_reuse_ratio", 0.1),
            _row("ngram3_rare_reuse_ratio", 0.2),
        ]

        result_rows = compute_tradeoff_metrics_from_rows(
            rows,
            utility_weight=0.6,
            privacy_weight=0.4,
        )
        result_map = {row["metric"]: row for row in result_rows}

        expected_utility = (0.75 + 0.8 + 0.9) / 3.0
        expected_privacy = (0.9 + 0.8 + 0.8 + 0.9 + 0.7 + 0.9 + 0.8) / 7.0
        expected_balance = 0.6 * expected_utility + 0.4 * expected_privacy

        self.assertAlmostEqual(result_map["tradeoff_utility_score"]["value"], expected_utility)
        self.assertAlmostEqual(result_map["tradeoff_privacy_score"]["value"], expected_privacy)
        self.assertAlmostEqual(result_map["tradeoff_balance_score"]["value"], expected_balance)

    def test_missing_base_metrics_returns_not_computable(self) -> None:
        result_rows = compute_tradeoff_metrics_from_rows([_row("memrisk_output_score", 0.2)])

        self.assertTrue(result_rows)
        self.assertTrue(all(row["status"] == "not_computable" for row in result_rows))
        self.assertEqual({row["reason"] for row in result_rows}, {"missing_base_metrics"})
        self.assertTrue(all(math.isnan(float(row["value"])) for row in result_rows))

    def test_ignores_nan_classifier_values_from_csv_roundtrip(self) -> None:
        rows = [
            _row("trtr_aucroc", 0.8, "xgboost"),
            _row("tstr_aucroc", 0.6, "xgboost"),
            _row("trtr_auprc", 0.5, "xgboost"),
            _row("tstr_auprc", 0.4, "xgboost"),
            _row("trtr_f1", 0.5, "xgboost"),
            _row("tstr_f1", 0.45, "xgboost"),
            _row("exact_matches_ratio", 0.1, float("nan")),
            _row("dcr_p5", 0.4, float("nan")),
            _row("dcr_holdout_p5", 0.5, float("nan")),
        ]

        result_rows = compute_tradeoff_metrics_from_rows(rows)
        result_map = {row["metric"]: row for row in result_rows}

        self.assertEqual(result_map["tradeoff_utility_score"]["status"], "ok")
        self.assertAlmostEqual(result_map["tradeoff_utility_score"]["value"], (0.75 + 0.8 + 0.9) / 3.0)
        self.assertAlmostEqual(result_map["tradeoff_privacy_score"]["value"], (0.9 + 0.8) / 2.0)


if __name__ == "__main__":
    unittest.main()
