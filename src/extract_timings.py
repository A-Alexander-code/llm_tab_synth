import json
import pandas as pd
from pathlib import Path


def _to_number(value, default=0.0):
    """Convierte a float de forma segura."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def consolidate_timings_complete(
    base_path: str | Path = Path("results").joinpath("samples"),
    output_path: str | Path = Path("results").joinpath("consolidated_timings.csv"),
):
    config = {
        "datasets": [
            "T1D_child",
            "heart_framingham",
            "cervical_cancer",
            "diabetic_retinopathy",
            "diabetes_prima_indian",
            "kidney_disease",
            "df_static_101_tabular"
        ],
        "models": [
            "great", "tabula", "realtabformer", "ctgan", "tvae",
            "gaussian_copula", "copulagan", "tabpfgen", "forestdiffusion", "mostlyai",
        ],
        "seeds": [13, 23, 42, 69, 88],
    }

    required_keys = [
        "training_class_0_s",
        "sampling_class_0_s",
        "training_class_1_s",
        "sampling_class_1_s",
    ]

    results = []
    missing_files = []
    incomplete_files = []
    error_files = []

    base_dir = Path(base_path)
    output_path = Path(output_path)

    total_expected = (
        len(config["models"]) *
        len(config["datasets"]) *
        len(config["seeds"])
    )

    print(
        f"Starting consolidation for "
        f"{len(config['models'])} models, "
        f"{len(config['datasets'])} datasets, "
        f"{len(config['seeds'])} seeds "
        f"({total_expected} expected records)..."
    )

    for model in config["models"]:
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                file_path = (
                    base_dir
                    .joinpath(model)
                    .joinpath(dataset)
                    .joinpath(f"seed_{seed}")
                    .joinpath("timing.json")
                )

                row = {
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "file_path": str(file_path),
                    "status": "ok",
                    "missing_keys": "",
                    "training_class_0_s": 0.0,
                    "sampling_class_0_s": 0.0,
                    "training_class_1_s": 0.0,
                    "sampling_class_1_s": 0.0,
                }

                if not file_path.exists():
                    row["status"] = "missing_file"
                    missing_files.append({
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "file_path": str(file_path),
                    })
                    row["total_training_s"] = 0.0
                    row["total_sampling_s"] = 0.0
                    results.append(row)
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    missing_keys = [k for k in required_keys if k not in data]
                    if missing_keys:
                        row["status"] = "incomplete"
                        row["missing_keys"] = ",".join(missing_keys)
                        incomplete_files.append({
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "file_path": str(file_path),
                            "missing_keys": missing_keys,
                        })

                    for key in required_keys:
                        row[key] = _to_number(data.get(key, 0.0), default=0.0)

                    row["total_training_s"] = (
                        row["training_class_0_s"] + row["training_class_1_s"]
                    )
                    row["total_sampling_s"] = (
                        row["sampling_class_0_s"] + row["sampling_class_1_s"]
                    )

                    results.append(row)

                except Exception as e:
                    row["status"] = "error"
                    row["missing_keys"] = str(e)
                    row["total_training_s"] = 0.0
                    row["total_sampling_s"] = 0.0
                    error_files.append({
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "file_path": str(file_path),
                        "error": str(e),
                    })
                    results.append(row)

    df = pd.DataFrame(results)

    cols = [
        "model", "dataset", "seed",
        "status", "missing_keys", "file_path",
        "total_training_s", "total_sampling_s",
        "training_class_0_s", "training_class_1_s",
        "sampling_class_0_s", "sampling_class_1_s",
    ]
    df = df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("-" * 40)
    print("Process completed.")
    print(f"Expected records: {total_expected}")
    print(f"Rows written: {len(df)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Incomplete files: {len(incomplete_files)}")
    print(f"Files with errors: {len(error_files)}")
    print(f"Saved file: {output_path}")

    return df


if __name__ == "__main__":
    consolidate_timings_complete()