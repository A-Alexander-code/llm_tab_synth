import json
import pandas as pd
from pathlib import Path

def consolidate_timings_strict(base_path: str | Path = Path("results").joinpath("samples")):
    config = {
        "datasets": [
            "T1D_child",
            "heart_framingham",
            "cervical_cancer",
            "diabetic_retinopathy",
            "diabetes_prima_indian",
            "kidney_disease"
        ],
        "models": [
            "great", "tabula", "realtabformer", "ctgan", "tvae",
            "gaussian_copula", "copulagan", "tabpfgen", "forestdiffusion", "mostlyai"
        ],
        "seeds": [13, 23, 42, 69, 88]
    }

    required_keys = [
        "training_class_0_s",
        "sampling_class_0_s",
        "training_class_1_s",
        "sampling_class_1_s"
    ]

    results = []
    missing_files = []
    invalid_records = []
    error_files = []

    base_dir = Path(base_path)

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

                record_id = {
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "file_path": str(file_path)
                }

                if not file_path.exists():
                    missing_files.append(record_id)
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    missing_keys = [k for k in required_keys if k not in data]

                    if missing_keys:
                        invalid_records.append({
                            **record_id,
                            "missing_keys": missing_keys
                        })
                        continue

                    data["model"] = model
                    data["dataset"] = dataset
                    data["seed"] = seed

                    data["total_training_s"] = (
                        data["training_class_0_s"] + data["training_class_1_s"]
                    )
                    data["total_sampling_s"] = (
                        data["sampling_class_0_s"] + data["sampling_class_1_s"]
                    )

                    results.append(data)

                except Exception as e:
                    error_files.append({
                        **record_id,
                        "error": str(e)
                    })

    df = pd.DataFrame(results)

    if not df.empty:
        cols = [
            "model", "dataset", "seed",
            "total_training_s", "total_sampling_s",
            "training_class_0_s", "training_class_1_s",
            "sampling_class_0_s", "sampling_class_1_s"
        ]
        df = df[cols]

        output_path = Path("results").joinpath("consolidated_timings.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("-" * 40)
        print("Process completed.")
        print(f"Expected records: {total_expected}")
        print(f"Valid rows found: {len(df)}")
        print(f"Missing files: {len(missing_files)}")
        print(f"Invalid records: {len(invalid_records)}")
        print(f"Files with read/parse errors: {len(error_files)}")
        print(f"Saved file: {output_path}")

        if missing_files:
            print("\nMissing file records:")
            for item in missing_files:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> {item['file_path']}"
                )

        if invalid_records:
            print("\nInvalid records (existing file but missing required keys):")
            for item in invalid_records:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> missing_keys={item['missing_keys']} "
                    f"-> {item['file_path']}"
                )

        if error_files:
            print("\nFiles with errors:")
            for item in error_files:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> error={item['error']} "
                    f"-> {item['file_path']}"
                )

    else:
        print("No valid timing.json files were consolidated.")

        if missing_files:
            print("\nMissing file records:")
            for item in missing_files:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> {item['file_path']}"
                )

        if invalid_records:
            print("\nInvalid records:")
            for item in invalid_records:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> missing_keys={item['missing_keys']} "
                    f"-> {item['file_path']}"
                )

        if error_files:
            print("\nFiles with errors:")
            for item in error_files:
                print(
                    f"  - model={item['model']}, "
                    f"dataset={item['dataset']}, "
                    f"seed={item['seed']} "
                    f"-> error={item['error']} "
                    f"-> {item['file_path']}"
                )

    return df

if __name__ == "__main__":
    consolidate_timings_strict()