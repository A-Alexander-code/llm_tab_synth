"""Adversarial tests for A1 preparation; no model libraries or historical data."""

from __future__ import annotations

import ast
import json
import struct
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from src.revision_a1 import (
    Preprocessor, SourceData, canonical, checks, load_source, npy_bytes, prepare,
    split_groups, verify_saved, write_json, digest,
)


class RevisionA1Tests(unittest.TestCase):
    def data(self) -> SourceData:
        return SourceData(("x",), tuple((Decimal(i),) for i in range(8)),
                          (0, 0, 0, 0, 1, 1, 1, 1), tuple(f"r{i}" for i in range(8)),
                          ("a", "a", "b", "c", "d", "d", "e", "f"), "hash", {})

    def spec(self) -> dict:
        return {"types": {"x": "continuous"}, "drop_missing_threshold": None}

    def test_groups_preserve_every_row_and_class(self) -> None:
        data = self.data()
        train, test = split_groups(data, 42, 0.25)
        self.assertEqual(len(test), 2)
        self.assertEqual(sorted(train + test), list(range(8)))
        self.assertFalse({data.group_ids[i] for i in train} & {data.group_ids[i] for i in test})
        self.assertEqual({data.y[i] for i in test}, {0, 1})
        self.assertEqual((train, test), split_groups(data, 42, 0.25))

    def test_impossible_class_group_partition_stops(self) -> None:
        data = SourceData(("x",), ((Decimal(0),), (Decimal(1),)), (0, 1), ("a", "b"), ("a", "b"), "hash", {})
        with self.assertRaisesRegex(ValueError, "cannot appear"):
            split_groups(data, 42, 0.2)

    def test_mode_tie_and_fractional_count_median(self) -> None:
        rows = [(Decimal(2), Decimal(1)), (Decimal(1), Decimal(2)), (None, None)]
        prep = Preprocessor.fit(rows, ("binary", "count"), {"binary": "categorical", "count": "count"}, None)
        self.assertEqual(prep.fill_values, {"binary": "1", "count": "1.5"})
        self.assertEqual(prep.transform([(None, None)]), [[1.0, 1.5]])
        with self.assertRaises(TypeError):
            prep.transform([(None, None)], [1])

    def test_selection_uses_only_train_and_threshold_includes_equality(self) -> None:
        rows = [(None, Decimal(1))] * 3 + [(Decimal(2), Decimal(3))] * 2
        prep = Preprocessor.fit(rows, ("drop", "keep"), {"drop": "continuous", "keep": "continuous"}, 0.6)
        self.assertEqual(prep.kept, ("keep",))
        self.assertEqual(prep.transform([(Decimal(999), None)]), [[1.0]])
        with self.assertRaisesRegex(ValueError, "Entirely missing"):
            Preprocessor.fit([(None,)], ("x",), {"x": "continuous"}, None)

    def test_nonfinite_detected_and_leaky_statistics_rejected(self) -> None:
        data = self.data()
        train, test = split_groups(data, 42, 0.25)
        prep = Preprocessor.fit([data.X[i] for i in train], data.names, self.spec()["types"], None)
        good = checks(data, self.spec(), 42, 0.25, train, test, prep,
                      prep.transform([data.X[i] for i in train]), prep.transform([data.X[i] for i in test]))
        self.assertTrue(good["passed"])
        bad = Preprocessor(prep.names, prep.kept, prep.missing_counts, prep.missing_fractions,
                           {"x": "999"}, prep.strategies, prep.train_categories)
        result = checks(data, self.spec(), 42, 0.25, train, test, bad,
                        [[float("nan")]] * len(train), bad.transform([data.X[i] for i in test]))
        self.assertIn("independent_train_statistics", result["failures"])
        self.assertIn("aligned_finite_matrices", result["failures"])

    def test_conflicting_targets_and_unexpected_missing_stop(self) -> None:
        spec = {"column_mapping": {"x": "x", "target": "class"}, "source_target": "target",
                "types": {"x": "continuous"}, "missing_tokens": [], "zero_as_missing": [], "categorical_domains": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).joinpath("source.csv")
            path.write_text("x,target\n1,0\n1.0,1\n")
            with self.assertRaisesRegex(ValueError, "Contradictory"):
                load_source(path, spec)
            path.write_text("x,target\n,0\n")
            with self.assertRaisesRegex(ValueError, "Unexpected blank"):
                load_source(path, spec)
            path.write_text("1,0\n2,1\n")
            with self.assertRaisesRegex(ValueError, "Header mismatch"):
                load_source(path, spec)

    def test_npy_standard_header_and_payload(self) -> None:
        payload = npy_bytes([[1.5, -2.0]], True)
        self.assertEqual(payload[:8], b"\x93NUMPY\x01\x00")
        length = int.from_bytes(payload[8:10], "little")
        header = ast.literal_eval(payload[10:10 + length].decode())
        self.assertEqual(header, {"descr": "<f8", "fortran_order": False, "shape": (1, 2)})
        self.assertEqual((10 + length) % 16, 0)
        self.assertEqual(struct.unpack("<dd", payload[10 + length:]), (1.5, -2.0))

    def test_saved_object_replay_and_tampering(self) -> None:
        data = self.data()
        payloads, _ = prepare(data, self.spec(), 42, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in payloads.items():
                root.joinpath(name).write_bytes(payload)
            write_json(root.joinpath("artifact_hashes.json"), {n: digest(root.joinpath(n)) for n in payloads})
            verify_saved(root, payloads, data)
            root.joinpath("preprocessing.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_saved(root, payloads, data)

    def test_group_canonicalization_is_exact(self) -> None:
        self.assertEqual(canonical(Decimal("1.000")), canonical(Decimal("1")))
        self.assertNotEqual(canonical(Decimal("1.00000000000000000000000000001")), canonical(Decimal("1")))

    def test_lazy_public_exports_remain_available(self) -> None:
        import src
        from src.logging_utils import setup_logging
        self.assertIs(src.setup_logging, setup_logging)
        with self.assertRaises(AttributeError):
            getattr(src, "nonexistent_utility")

    def test_postprocessing_collisions_are_reported_without_changing_split(self) -> None:
        data = self.data()
        train, test = split_groups(data, 42, 0.25)
        prep = Preprocessor.fit([data.X[i] for i in train], data.names, self.spec()["types"], None)
        # Deliberately supply equal transformed patterns across the fixed split.
        result = checks(data, self.spec(), 42, 0.25, train, test, prep,
                        [[0.0]] * len(train), [[0.0]] * len(test))
        self.assertEqual(result["postprocessing_overlap"]["unique_predictor_patterns"], 1)
        self.assertEqual(result["postprocessing_overlap"]["test_rows_matching_train"], len(test))
        self.assertEqual((train, test), split_groups(data, 42, 0.25))


if __name__ == "__main__":
    unittest.main()
