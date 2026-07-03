from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation.llm_privacy import (
    compute_ngram_overlap_metrics,
    compute_record_similarity_metrics,
    compute_semantic_similarity_metrics,
    evaluate_llm_privacy_metrics,
)


def _real_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [50, 60, 70],
            "bmi": [25.0, 30.0, 35.0],
            "smoker": ["no", "yes", "no"],
            "class": [0, 1, 0],
        }
    )


class LlmPrivacyTests(unittest.TestCase):
    def test_identical_data_is_maximally_similar(self) -> None:
        real_train = _real_df()
        synthetic = real_train.copy()

        record_metrics = compute_record_similarity_metrics(real_train, synthetic)
        ngram_metrics = compute_ngram_overlap_metrics(real_train, synthetic)
        semantic_metrics = compute_semantic_similarity_metrics(real_train, synthetic)

        self.assertEqual(record_metrics["record_sim_max_mean"], 1.0)
        self.assertEqual(record_metrics["record_sim_exactish_ratio"], 1.0)
        self.assertEqual(ngram_metrics["ngram2_overlap_mean"], 1.0)
        self.assertEqual(ngram_metrics["ngram2_synthetic_vocab_seen_ratio"], 1.0)
        self.assertGreater(semantic_metrics["semantic_sim_max_mean"], 0.99)
        self.assertEqual(semantic_metrics["semantic_sim_high_ratio_095"], 1.0)

    def test_different_data_reduces_overlap(self) -> None:
        real_train = _real_df()
        synthetic = pd.DataFrame(
            {
                "age": [10, 15, 20],
                "bmi": [45.0, 46.0, 47.0],
                "smoker": ["unknown", "unknown", "unknown"],
                "class": [1, 1, 1],
            }
        )

        record_metrics = compute_record_similarity_metrics(real_train, synthetic)
        ngram_metrics = compute_ngram_overlap_metrics(real_train, synthetic)

        self.assertLess(record_metrics["record_sim_max_mean"], 0.5)
        self.assertLess(ngram_metrics["ngram2_rare_reuse_ratio"], 0.5)

    def test_no_common_columns_returns_not_computable(self) -> None:
        real_train = pd.DataFrame({"age": [1, 2], "class": [0, 1]})
        synthetic = pd.DataFrame({"weight": [1, 2], "class": [0, 1]})

        rows = evaluate_llm_privacy_metrics(
            real_train=real_train,
            real_test=None,
            synthetic=synthetic,
            dataset="ds",
            model="great",
            seed=42,
            output_ir=1.0,
            metrics=["record_similarity"],
        )

        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "not_computable" for row in rows))
        self.assertEqual({row["reason"] for row in rows}, {"no_common_columns"})
        self.assertTrue(all(math.isnan(float(row["value"])) for row in rows))

    def test_empty_input_returns_not_computable(self) -> None:
        real_train = pd.DataFrame(columns=["age", "class"])
        synthetic = pd.DataFrame(columns=["age", "class"])

        rows = evaluate_llm_privacy_metrics(
            real_train=real_train,
            real_test=None,
            synthetic=synthetic,
            dataset="ds",
            model="great",
            seed=42,
            output_ir=1.0,
            metrics=["semantic_similarity"],
        )

        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "not_computable" for row in rows))
        self.assertEqual({row["reason"] for row in rows}, {"empty_input"})

    def test_semantic_similarity_accepts_sparse_max_outputs(self) -> None:
        real_train = _real_df()
        synthetic = real_train.copy()

        class FakeSparseMaxResult:
            def toarray(self) -> list[list[float]]:
                return [[0.9], [1.0], [0.95]]

        class FakeSimilarityMatrix:
            def max(self, axis: int) -> FakeSparseMaxResult:
                self_axis = axis
                if self_axis != 1:
                    raise AssertionError("Expected row-wise max over axis=1")
                return FakeSparseMaxResult()

        class FakeTfidfMatrix:
            @property
            def T(self) -> "FakeTfidfMatrix":
                return self

            def __matmul__(self, other: object) -> FakeSimilarityMatrix:
                del other
                return FakeSimilarityMatrix()

        class FakeVectorizer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

            def fit_transform(self, values: list[str]) -> FakeTfidfMatrix:
                del values
                return FakeTfidfMatrix()

            def transform(self, values: list[str]) -> FakeTfidfMatrix:
                del values
                return FakeTfidfMatrix()

        with patch("src.evaluation.llm_privacy.TfidfVectorizer", FakeVectorizer):
            semantic_metrics = compute_semantic_similarity_metrics(real_train, synthetic)

        self.assertAlmostEqual(semantic_metrics["semantic_sim_max_mean"], 0.95)
        self.assertEqual(semantic_metrics["semantic_sim_high_ratio_095"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
