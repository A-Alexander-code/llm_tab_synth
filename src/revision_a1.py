"""Isolated A1 preparation: source-only groups, train-only statistics, no models.

Only the standard library is required. Decimal preserves observed numeric values
when grouping; float64 is used only for the exported matrices, without scaling.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import statistics
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

Cell = Decimal | None
Row = tuple[Cell, ...]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(encode(value))


def canonical(value: Cell) -> str | None:
    # Decimal equality treats 1, 1.0 and 1e0 identically, without rounding.
    if value is None:
        return None
    text = format(value, "f")
    return "0" if value == 0 else text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class SourceData:
    names: tuple[str, ...]
    X: tuple[Row, ...]
    y: tuple[int, ...]
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_hash: str
    audit: dict[str, Any]


def load_source(path: Path, spec: dict[str, Any]) -> SourceData:
    """Read only the explicit source path; reject schema or missing surprises."""
    source_hash = digest(path)
    mapping = spec["column_mapping"]
    names = tuple(v for v in mapping.values() if v != "class")
    if set(names) != set(spec["types"]) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Invalid explicit schema")
    X: list[Row] = []
    y: list[int] = []
    row_ids: list[str] = []
    group_ids: list[str] = []
    missing = Counter()
    tokens = Counter()
    zeros = Counter()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, delimiter=",")
        if next(reader) != list(mapping):
            raise ValueError(f"Header mismatch: {path.name}; no data row was discarded")
        for position, fields in enumerate(reader, 1):
            if len(fields) != len(mapping):
                raise ValueError(f"Malformed source row {position}")
            record = dict(zip(mapping, fields))
            target = Decimal(record[spec["source_target"]])
            if target not in (0, 1):
                raise ValueError(f"Unexpected target at source row {position}")
            values: list[Cell] = []
            for original, name in mapping.items():
                if name == "class":
                    continue
                token = record[original]
                if token in spec["missing_tokens"]:
                    value = None
                    tokens[token] += 1
                else:
                    if not token.strip():
                        raise ValueError(f"Unexpected blank: {name}, row {position}")
                    value = Decimal(token)
                    if not value.is_finite():
                        raise ValueError(f"Nonfinite source: {name}, row {position}")
                    if value == 0:
                        zeros[name] += 1
                        if name in spec["zero_as_missing"]:
                            value = None
                kind = spec["types"][name]
                if value is not None:
                    if kind == "binary" and value not in (0, 1):
                        raise ValueError(f"Invalid binary: {name}, row {position}")
                    if kind == "count" and (value < 0 or value != value.to_integral_value()):
                        raise ValueError(f"Invalid count: {name}, row {position}")
                    if name in spec["categorical_domains"] and value not in spec["categorical_domains"][name]:
                        raise ValueError(f"Unexpected category: {name}, row {position}")
                else:
                    missing[name] += 1
                values.append(value)
            row = tuple(values)
            group = hashlib.sha256(encode([names, [canonical(v) for v in row]])).hexdigest()
            X.append(row)
            y.append(int(target))
            row_ids.append(f"{path.name}:{source_hash}:row_{position}")
            group_ids.append(group)
    labels: dict[str, set[int]] = defaultdict(set)
    for group, target in zip(group_ids, y):
        labels[group].add(target)
    conflicts = [g for g, targets in labels.items() if len(targets) != 1]
    if conflicts:
        raise ValueError(f"Contradictory targets in {len(conflicts)} predictor groups; preparation stopped")
    return SourceData(names, tuple(X), tuple(y), tuple(row_ids), tuple(group_ids), source_hash, {
        "source_rows": len(X), "source_groups": len(labels),
        "repeated_rows_extra": len(X) - len(labels), "conflicting_groups": 0,
        "class_counts": dict(Counter(y)), "missing_by_column": dict(missing),
        "missing_tokens": dict(tokens), "zero_counts_before_missing": dict(zeros),
        "source_row_position": "one-based data record, excluding header",
        "subject_id": "No explicit subject identifier in the authorized schema",
    })


def split_groups(data: SourceData, seed: int, fraction: float) -> tuple[list[int], list[int]]:
    """Find nearest reachable test row count independently per class.

    Shuffle sorted group hashes using Random(seed), then subset-sum by group size.
    Each reachable sum retains its first predecessor; ties prefer fewer test rows.
    Target counts use half-up rounding. Both partitions must retain each class.
    No features beyond group identity, fitted statistics or outcomes are scored.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(data.group_ids):
        groups[group].append(i)
    rng = random.Random(seed)
    test_groups: set[str] = set()
    for label in (0, 1):
        candidates = sorted(g for g, ids in groups.items() if data.y[ids[0]] == label)
        rng.shuffle(candidates)
        n = sum(len(groups[g]) for g in candidates)
        wanted = math.floor(n * fraction + 0.5)
        reachable = 1
        predecessors: dict[int, tuple[int, str]] = {}
        for group in candidates:
            size = len(groups[group])
            shifted = reachable << size
            new = shifted & ~reachable
            while new:
                bit = new & -new
                total = bit.bit_length() - 1
                predecessors[total] = (total - size, group)
                new -= bit
            reachable |= shifted
        possible = [i for i in range(1, n) if (reachable >> i) & 1]
        if not possible:
            raise ValueError(f"Class {label} cannot appear on both sides of a group split")
        total = min(possible, key=lambda i: (abs(i - wanted), i))
        while total:
            total, group = predecessors[total]
            test_groups.add(group)
    train = [i for i, g in enumerate(data.group_ids) if g not in test_groups]
    test = [i for i, g in enumerate(data.group_ids) if g in test_groups]
    return train, test


@dataclass(frozen=True)
class Preprocessor:
    """JSON-serializable fitted object; fit and transform have no target input."""

    names: tuple[str, ...]
    kept: tuple[str, ...]
    missing_counts: dict[str, int]
    missing_fractions: dict[str, float]
    fill_values: dict[str, str]
    strategies: dict[str, str]
    train_categories: dict[str, list[str]]

    @classmethod
    def fit(cls, X_train: Sequence[Row], names: tuple[str, ...],
            types: dict[str, str], threshold: float | None) -> Preprocessor:
        if not X_train:
            raise ValueError("Empty train")
        kept: list[str] = []
        counts: dict[str, int] = {}
        fractions: dict[str, float] = {}
        fills: dict[str, str] = {}
        strategies: dict[str, str] = {}
        categories: dict[str, list[str]] = {}
        for j, name in enumerate(names):
            observed = [r[j] for r in X_train if r[j] is not None]
            counts[name] = len(X_train) - len(observed)
            fractions[name] = counts[name] / len(X_train)
            if threshold is not None and fractions[name] >= threshold:
                continue
            if not observed:
                raise ValueError(f"Entirely missing train column outside drop rule: {name}")
            kept.append(name)
            if types[name] in ("binary", "categorical"):
                frequency = Counter(observed)
                value = min(v for v, n in frequency.items() if n == max(frequency.values()))
                strategies[name] = "mode_smallest_numeric_tie"
                categories[name] = [canonical(v) for v in sorted(frequency)]
            else:
                value = statistics.median(observed)
                strategies[name] = "median_fraction_preserved"
            fills[name] = str(value)
        return cls(names, tuple(kept), counts, fractions, fills, strategies, categories)

    def transform(self, X: Sequence[Row]) -> list[list[float]]:
        indices = [self.names.index(c) for c in self.kept]
        return [[float(row[j] if row[j] is not None else Decimal(self.fill_values[c]))
                 for c, j in zip(self.kept, indices)] for row in X]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Preprocessor:
        return cls(**{**value, "names": tuple(value["names"]), "kept": tuple(value["kept"])})


def npy_bytes(values: Sequence[Any], matrix: bool) -> bytes:
    """NumPy NPY v1.0, little-endian float64 matrices / int32 labels, C order."""
    shape = (len(values), len(values[0])) if matrix else (len(values),)
    header = repr({"descr": "<f8" if matrix else "<i4", "fortran_order": False, "shape": shape})
    padding = (16 - (10 + len(header) + 1) % 16) % 16
    encoded = (header + " " * padding + "\n").encode("ascii")
    flat = [v for row in values for v in row] if matrix else list(values)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + struct.pack(
        "<" + ("d" if matrix else "i") * len(flat), *flat)


def read_npy(path: Path) -> list[Any]:
    """Read back the restricted NPY format without importing NumPy."""
    with path.open("rb") as stream:
        if stream.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError("Unexpected NPY version")
        size = struct.unpack("<H", stream.read(2))[0]
        header = ast.literal_eval(stream.read(size).decode("ascii"))
        shape = header["shape"]
        kind = {"<f8": "d", "<i4": "i"}[header["descr"]]
        flat = [v[0] for v in struct.iter_unpack("<" + kind, stream.read())]
        if math.prod(shape) != len(flat) or header["fortran_order"]:
            raise ValueError("Invalid NPY payload")
        return [flat[i:i + shape[1]] for i in range(0, len(flat), shape[1])] if len(shape) == 2 else flat


def checks(data: SourceData, spec: dict[str, Any], seed: int, fraction: float,
           train: list[int], test: list[int], prep: Preprocessor,
           X_train: list[list[float]], X_test: list[list[float]]) -> dict[str, Any]:
    """Independent statistics and fixed-split intervention checks, without fits of models."""
    passed: dict[str, bool] = {}
    passed["disjoint_source_rows"] = not set(train) & set(test)
    passed["all_source_rows_preserved"] = sorted(train + test) == list(range(len(data.X)))
    passed["disjoint_groups"] = not {data.group_ids[i] for i in train} & {data.group_ids[i] for i in test}
    passed["both_classes"] = all({data.y[i] for i in ids} == {0, 1} for ids in (train, test))
    train_raw = [data.X[i] for i in train]
    test_raw = [data.X[i] for i in test]
    # Independent implementation: sorted middle values and sorted frequency pairs.
    independent = True
    for j, name in enumerate(data.names):
        values = sorted(r[j] for r in train_raw if r[j] is not None)
        n_missing = len(train) - len(values)
        independent &= prep.missing_counts[name] == n_missing
        independent &= prep.missing_fractions[name] == n_missing / len(train)
        drop = spec["drop_missing_threshold"] is not None and n_missing / len(train) >= spec["drop_missing_threshold"]
        independent &= (name not in prep.kept) == drop
        if drop:
            continue
        if spec["types"][name] in ("binary", "categorical"):
            expected = sorted(Counter(values).items(), key=lambda p: (-p[1], p[0]))[0][0]
        else:
            mid = len(values) // 2
            expected = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
        independent &= Decimal(prep.fill_values[name]) == expected
    passed["independent_train_statistics"] = independent
    # A label-bearing evaluation wrapper is exercised with three different targets;
    # transform itself cannot accept targets, so none can choose a class imputor.
    def evaluation_view(labels: list[int]) -> tuple[list[list[float]], list[int]]:
        return prep.transform(test_raw), labels
    labels = [data.y[i] for i in test]
    shuffled = labels.copy()
    random.Random(seed).shuffle(shuffled)
    passed["test_labels_flipped_invariance"] = evaluation_view([1 - v for v in labels])[0] == X_test
    passed["test_labels_shuffled_invariance"] = evaluation_view(shuffled)[0] == X_test
    # Intervention is on the full predictor collection with row membership fixed.
    intervened = list(data.X)
    for i in test:
        intervened[i] = tuple(None if j % 2 else Decimal(100000 + j) for j in range(len(data.names)))
    other = Preprocessor.fit([intervened[i] for i in train], data.names,
                             spec["types"], spec["drop_missing_threshold"])
    passed["test_predictors_do_not_change_fit"] = other.to_dict() == prep.to_dict()
    passed["test_predictors_do_not_change_train"] = other.transform([intervened[i] for i in train]) == X_train
    passed["aligned_finite_matrices"] = bool(prep.kept) and all(
        len(row) == len(prep.kept) and all(math.isfinite(v) for v in row)
        for row in X_train + X_test)
    train2, test2 = split_groups(data, seed, fraction)
    prep2 = Preprocessor.fit([data.X[i] for i in train2], data.names,
                             spec["types"], spec["drop_missing_threshold"])
    passed["repeat_partition"] = (train2, test2) == (train, test)
    passed["repeat_fit_and_values"] = prep2.to_dict() == prep.to_dict() and (
        prep2.transform([data.X[i] for i in train2]) == X_train
        and prep2.transform([data.X[i] for i in test2]) == X_test)
    train_keys = Counter(tuple(r) for r in X_train)
    test_keys = Counter(tuple(r) for r in X_test)
    overlap = train_keys.keys() & test_keys.keys()
    return {"checks": passed, "passed": all(passed.values()), "failures": [k for k, v in passed.items() if not v],
            "postprocessing_overlap": {"unique_predictor_patterns": len(overlap),
                "test_rows_matching_train": sum(test_keys[k] for k in overlap),
                "train_rows_matching_test": sum(train_keys[k] for k in overlap)}}


def prepare(data: SourceData, spec: dict[str, Any], seed: int,
            fraction: float) -> tuple[dict[str, bytes], dict[str, Any]]:
    train, test = split_groups(data, seed, fraction)
    prep = Preprocessor.fit([data.X[i] for i in train], data.names,
                            spec["types"], spec["drop_missing_threshold"])
    X_train = prep.transform([data.X[i] for i in train])
    X_test = prep.transform([data.X[i] for i in test])
    verification = checks(data, spec, seed, fraction, train, test, prep, X_train, X_test)
    if not verification["passed"]:
        raise ValueError(f"Verification failures: {verification['failures']}")
    partitions = {side: [{"matrix_row": j, "source_position": i + 1,
                         "row_id": data.row_ids[i], "group_id": data.group_ids[i]}
                        for j, i in enumerate(ids)] for side, ids in (("train", train), ("test", test))}
    summary: dict[str, Any] = {"seed": seed, "source_sha256": data.source_hash,
        "feature_names": list(prep.kept), "dropped_columns": [c for c in data.names if c not in prep.kept],
        "split_method": "classwise_subset_sum_v1; nearest reachable class row quotas; approximate stratification",
        "requested_test_fraction": fraction, "effective_test_fraction": len(test) / len(data.X),
        "n_source": len(data.X), "n_source_groups": len(set(data.group_ids)),
        "n_features": len(prep.kept), "postprocessing_overlap": verification["postprocessing_overlap"]}
    artifacts: dict[str, bytes] = {"preprocessing.json": encode(prep.to_dict()),
        "partition.json": encode(partitions), "verification.json": encode(verification),
        "schema.json": encode(spec), "source_audit.json": encode(data.audit)}
    for side, ids, matrix in (("train", train, X_train), ("test", test, X_test)):
        summary[side] = {"rows": len(ids), "groups": len({data.group_ids[i] for i in ids}),
            "class_counts": dict(Counter(data.y[i] for i in ids)),
            "class_proportions": {k: sum(data.y[i] == k for i in ids) / len(ids) for k in (0, 1)},
            "missing_before_selection": sum(v is None for i in ids for v in data.X[i]),
            "missing_before_imputation_kept": sum(data.X[i][data.names.index(c)] is None for i in ids for c in prep.kept),
            "missing_after": 0}
        artifacts[f"X_{side}.npy"] = npy_bytes(matrix, True)
        artifacts[f"y_{side}.npy"] = npy_bytes([data.y[i] for i in ids], False)
    summary["fractional_count_imputers"] = [c for c in prep.kept if spec["types"][c] == "count"
        and Decimal(prep.fill_values[c]) != Decimal(prep.fill_values[c]).to_integral_value()]
    artifacts["feature_meta.json"] = encode(summary)
    return artifacts, summary


def verify_saved(directory: Path, expected: dict[str, bytes], data: SourceData) -> None:
    manifest = json.loads(directory.joinpath("artifact_hashes.json").read_text())
    if set(manifest) != set(expected):
        raise ValueError("Artifact inventory mismatch")
    for name, payload in expected.items():
        path = directory.joinpath(name)
        if path.read_bytes() != payload or digest(path) != manifest[name]:
            raise ValueError(f"Artifact/reproduction mismatch: {name}")
    prep = Preprocessor.from_dict(json.loads(directory.joinpath("preprocessing.json").read_text()))
    partitions = json.loads(directory.joinpath("partition.json").read_text())
    for side in ("train", "test"):
        matrix = read_npy(directory.joinpath(f"X_{side}.npy"))
        labels = read_npy(directory.joinpath(f"y_{side}.npy"))
        if len(matrix) != len(labels) or any(len(r) != len(prep.kept) for r in matrix):
            raise ValueError("Serialized matrix shape mismatch")
        ids = [r["source_position"] - 1 for r in partitions[side]]
        if prep.transform([data.X[i] for i in ids]) != matrix or [data.y[i] for i in ids] != labels:
            raise ValueError("Reloaded preprocessing object/matrices mismatch")
