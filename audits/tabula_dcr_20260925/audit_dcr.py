"""Read-only DCR reconstruction; writes exclusively into a new output directory.

Run with --root REPOSITORY --out NEW_DIRECTORY. No fitting of generators occurs.
The only fit is the original training-only StandardScaler for distance evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

SEEDS = (13, 23, 42, 69, 88)
DATASETS = ("heart_framingham", "diabetes_prima_indian")
CAT_ATOL = 1e-6
SMALL_VARIANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": float(values.min()), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)), "maximum": float(values.max()),
        "mean": float(values.mean()), "zero_count": int((values == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from src.evaluation.builder import build_synthetic_dataset
    from src.evaluation.privacy import compute_dcr, compute_dcr_holdout

    inputs: dict[str, dict[str, Any]] = {}

    def track(path: Path) -> Path:
        inputs[str(path.relative_to(root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    summaries, predictors, extremes, contributions, populations = [], [], [], [], []
    spec = json.loads(track(root.joinpath("config", "datasets_description.json")).read_text())
    sources = ["main_evaluator.py", "src/evaluation/privacy.py", "src/evaluation/builder.py",
               "src/postprocessing.py", "src/plots/privacy.py", "plotting_results_ver3.ipynb",
               "src/evaluation/aggregator.py", "src/data_splitter.py"]
    for source in sources:
        track(root.joinpath(source))
    for collection, collection_root in [("current", root.joinpath("results")),
                                        ("archived", root.joinpath("results", "old_deprecated"))]:
        for dataset in DATASETS:
            stored = pd.read_csv(track(collection_root.joinpath("metrics_raw", f"{dataset}_master.csv")))
            stored = stored[(stored.model == "tabula") & (stored.output_ir == 1.0)]
            for seed in SEEDS:
                key = {"collection": collection, "dataset": dataset, "seed": seed}
                split_dir = root.joinpath("data", "processed", dataset, f"seed_{seed}")
                meta = json.loads(track(split_dir.joinpath("feature_meta.json")).read_text())
                names = meta["dataset"]["feature_names"]
                arrays = {n: np.load(track(split_dir.joinpath(f"{n}.npy")))
                          for n in ["X_train", "X_test", "y_train", "y_test"]}
                train = pd.DataFrame(arrays["X_train"], columns=names)
                test = pd.DataFrame(arrays["X_test"], columns=names)
                sample_dir = collection_root.joinpath("samples", "tabula", dataset, f"seed_{seed}")
                samples, identities = [], []
                for label in (0, 1):
                    path = track(sample_dir.joinpath(f"synthetic_class_{label}.csv"))
                    df = pd.read_csv(path)
                    samples.append(df)
                    identities.append(pd.DataFrame({"source_path": str(path.relative_to(root)),
                                                    "source_row_index": np.arange(len(df)),
                                                    "source_csv_line": np.arange(len(df)) + 2}))
                for filename in ["params_snapshot.json", "timing.json"]:
                    if sample_dir.joinpath(filename).exists():
                        track(sample_dir.joinpath(filename))
                combination = collection_root.joinpath("metrics_raw", dataset, "tabula", f"seed_{seed}", "ir_1.00", "metrics.csv")
                saved_combination = pd.read_csv(track(combination)) if combination.exists() else None
                maj = 0 if (arrays["y_train"] == 0).sum() >= (arrays["y_train"] == 1).sum() else 1
                synth_full = build_synthetic_dataset(samples[maj], samples[1-maj], 1.0, seed, maj, 1-maj)
                ids = build_synthetic_dataset(identities[maj], identities[1-maj], 1.0, seed, maj, 1-maj)
                synth = synth_full.drop(columns=["class"], errors="ignore")
                common = [c for c in train.select_dtypes(include=[np.number]).columns if c in synth.columns]
                assert common == names and set(synth.columns) == set(names)
                scaler = StandardScaler()
                r = scaler.fit_transform(train[common].fillna(0))
                s = scaler.transform(synth[common].fillna(0))
                t = scaler.transform(test[common].fillna(0))
                assert np.isfinite(r).all() and np.isfinite(s).all() and np.isfinite(t).all()
                tree = BallTree(r, metric="euclidean")
                sd, sn = tree.query(s, k=1)
                td, tn = tree.query(t, k=1)
                sd, sn, td, tn = sd[:, 0], sn[:, 0], td[:, 0], tn[:, 0]
                # Call actual repository functions as an additional faithful reconstruction check.
                original_num = {x["metric"]: x["value"] for x in compute_dcr(synth, train)}["dcr_mean"]
                original_den = {x["metric"]: x["value"] for x in compute_dcr_holdout(test, train)}["dcr_holdout_mean"]
                assert original_num == sd.mean() and original_den == td.mean()
                num, den = float(sd.mean()), float(td.mean())
                metrics = stored[stored.seed == seed].set_index("metric").value
                saved_num, saved_den = float(metrics["dcr_mean"]), float(metrics["dcr_holdout_mean"])
                ratio, saved_ratio = num / den, saved_num / saved_den
                top_n = math.ceil(0.01 * len(sd))
                order = np.argsort(-sd, kind="stable")
                kept_mean = float(sd[order[top_n:]].mean())
                outside_rows = np.zeros(len(synth), dtype=bool)
                for j, column in enumerate(common):
                    a, b = train[column].to_numpy(), synth[column].to_numpy()
                    low, high = float(np.nanmin(a)), float(np.nanmax(a))
                    below, above = b < low, b > high
                    outside_rows |= below | above
                    allowed = [0., 1.] if column in spec[dataset].get("binary", []) else ([1., 2., 3., 4.] if column == "education" else None)
                    domain: dict[str, Any] = {"allowed_codes": json.dumps(allowed), "domain_exact_n": None,
                                               "domain_precision_only_n": None, "domain_substantial_n": None,
                                               "domain_exact_pct": None, "domain_substantial_pct": None}
                    if allowed:
                        gap = np.min(np.abs(b[:, None] - np.array(allowed)[None, :]), axis=1)
                        domain.update(domain_exact_n=int((gap > 0).sum()),
                                      domain_precision_only_n=int(((gap > 0) & (gap <= CAT_ATOL)).sum()),
                                      domain_substantial_n=int((gap > CAT_ATOL).sum()),
                                      domain_exact_pct=float(100*(gap > 0).mean()),
                                      domain_substantial_pct=float(100*(gap > CAT_ATOL).mean()))
                    # Only a documented definitional physical constraint: BMI must be positive.
                    physical_rule = "BMI > 0 (positive mass / squared positive height)" if column.lower() == "bmi" else "not_assessed"
                    predictors.append({**key, "predictor": column, "train_min": low, "train_max": high,
                        "synth_min": float(np.nanmin(b)), "synth_max": float(np.nanmax(b)),
                        "below_train_n": int(below.sum()), "above_train_n": int(above.sum()),
                        "below_train_pct": float(100*below.mean()), "above_train_pct": float(100*above.mean()),
                        "outside_train_pct": float(100*(below | above).mean()),
                        "train_nan_n": int(pd.isna(a).sum()), "synth_nan_n": int(pd.isna(b).sum()),
                        "train_inf_n": int(np.isinf(a).sum()), "synth_inf_n": int(np.isinf(b).sum()),
                        "train_variance": float(scaler.var_[j]), "train_scale": float(scaler.scale_[j]),
                        "train_mean": float(scaler.mean_[j]), "train_constant": bool(scaler.var_[j] == 0),
                        "train_small_variance": bool(0 < scaler.var_[j] < SMALL_VARIANCE),
                        "synth_constant": bool(np.nanmin(b) == np.nanmax(b)),
                        "physical_rule": physical_rule, "physical_violation_n": int((b <= 0).sum()) if column.lower() == "bmi" else None,
                        **domain})
                report = {**key, "n_train": len(train), "n_test": len(test), "n_synth": len(synth),
                    "n_predictors": len(common), "columns_in_order": json.dumps(common),
                    "class0_rows": len(samples[0]), "class1_rows": len(samples[1]),
                    "numerator": num, "denominator": den, "ratio": ratio,
                    "saved_numerator": saved_num, "saved_denominator": saved_den, "saved_ratio": saved_ratio,
                    "outside_train_rows_n": int(outside_rows.sum()), "outside_train_rows_pct": float(100*outside_rows.mean()),
                    "top1_n": top_n, "top1_distance_sum_share": float(sd[order[:top_n]].sum()/sd.sum()),
                    "trimmed_mean": kept_mean, "trimmed_ratio": kept_mean/den,
                    "constant_train_columns": json.dumps([common[j] for j in range(len(common)) if scaler.var_[j] == 0]),
                    "small_variance_train_columns": json.dumps([common[j] for j in range(len(common)) if 0 < scaler.var_[j] < SMALL_VARIANCE]),
                    "source_headers_match": all(list(x.drop(columns=['class'], errors='ignore').columns) == names for x in samples),
                    "master_combination_match": bool(saved_combination is not None and all(float(saved_combination.loc[saved_combination.metric == m, 'value'].iloc[0]) == float(metrics[m]) for m in ['dcr_mean','dcr_holdout_mean']))}
                for name, computed, saved in [("numerator",num,saved_num),("denominator",den,saved_den),("ratio",ratio,saved_ratio)]:
                    report[name+"_abs_diff"] = abs(computed-saved)
                    report[name+"_rel_diff"] = abs(computed-saved)/abs(saved) if saved else None
                report.update({"synth_"+k:v for k,v in distribution(sd).items()})
                report.update({"test_"+k:v for k,v in distribution(td).items()})
                summaries.append(report)
                for population, distances, nn in [("synthetic",sd,sn),("test",td,tn)]:
                    for i, (distance, neighbor) in enumerate(zip(distances, nn)):
                        identity = ids.iloc[i].drop(labels=['class']).to_dict() if population == 'synthetic' else {"source_path": str(split_dir.joinpath('X_test.npy').relative_to(root)), "source_row_index": i, "source_csv_line": None}
                        populations.append({**key,"population":population,"evaluation_row_index":i,
                                            "distance":float(distance),"train_neighbor_index":int(neighbor),**identity})
                for rank, i in enumerate(order[:top_n], 1):
                    neighbor = int(sn[i])
                    diff = s[i].astype(float) - r[neighbor].astype(float)
                    squared = diff**2
                    direct = cdist(s[i:i+1].astype(float), r.astype(float), metric='euclidean')[0]
                    # Robust tie-aware comparison in extended precision; not used for primary metrics.
                    longdist = np.sqrt(np.sum((s[i].astype(np.longdouble)-r.astype(np.longdouble))**2,axis=1))
                    best_long = int(longdist.argmin())
                    identity = ids.iloc[i].drop(labels=['class']).to_dict()
                    extremes.append({**key,"rank":rank,"evaluation_row_index":int(i),**identity,
                        "distance":float(sd[i]),"train_neighbor_index":neighbor,
                        "neighbor_train_class":int(arrays['y_train'][neighbor]),
                        "cdist_min":float(direct.min()),"cdist_abs_diff":abs(float(direct.min())-float(sd[i])),
                        "cdist_rel_diff":abs(float(direct.min())-float(sd[i]))/float(sd[i]) if sd[i] else 0.,
                        "cdist_equal_min_neighbors":int((direct == direct.min()).sum()),
                        "longdouble_neighbor_index":best_long,
                        "longdouble_neighbor_gap":float(longdist[neighbor]-longdist[best_long]),
                        "leading_predictor":common[int(squared.argmax())],
                        "leading_squared_share":float(squared.max()/squared.sum())})
                    for j,c in enumerate(common):
                        contributions.append({**key,"rank":rank,**identity,"train_neighbor_index":neighbor,
                            "predictor":c,"synth_original":float(synth.iloc[i][c]),
                            "train_neighbor_original":float(train.iloc[neighbor][c]),
                            "scaler_mean":float(scaler.mean_[j]),"scaler_scale":float(scaler.scale_[j]),
                            "synth_standardized":float(s[i,j]),"train_standardized":float(r[neighbor,j]),
                            "squared_contribution":float(squared[j]),"squared_share":float(squared[j]/squared.sum())})
    tables = {"summary":summaries,"predictors":predictors,"extreme_records":extremes,
              "extreme_contributions":contributions,"distances":populations}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out.joinpath(name+'.csv'),index=False)
    aggregate = []
    for (collection,dataset),g in pd.DataFrame(summaries).groupby(['collection','dataset']):
        aggregate.append({"collection":collection,"dataset":dataset,"mean_seed_ratios":g.ratio.mean(),
                          "ratio_means":g.numerator.mean()/g.denominator.mean(),
                          "saved_mean_seed_ratios":g.saved_ratio.mean(),
                          "saved_ratio_means":g.saved_numerator.mean()/g.saved_denominator.mean(),
                          "diagnostic_mean_trimmed_seed_ratios":g.trimmed_ratio.mean()})
    pd.DataFrame(aggregate).to_csv(out.joinpath('aggregation.csv'),index=False)
    manifest = {"git_head":subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                "python":sys.version,"platform":platform.platform(),
                "packages":{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
                "script_sha256":sha256(Path(__file__)),"inputs":inputs,
                "output_sha256":{p.name:sha256(p) for p in out.glob('*.csv')},
                "categorical_absolute_tolerance":CAT_ATOL,"small_variance_absolute_threshold":SMALL_VARIANCE,
                "trim_rule":"drop ceil(0.01*n) largest synthetic distances; fixed original scaler and denominator",
                "percentile_method":"numpy default linear", "source_indices":"zero-based arrays/CSV data rows; CSV line includes header",
                "historical_binding":"Current evaluator and shared stored splits used to reconstruct each separate sample collection; no original environment lock or evaluation manifest available."}
    out.joinpath('manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Verify originals have not changed during the audit.
    assert all(sha256(root.joinpath(p)) == info['sha256'] for p,info in inputs.items())


if __name__ == '__main__':
    main()
