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
        
    results = []
    missing_count = 0
    base_dir = Path(base_path)
    
    print(f"Starting consolidation for {len(config['models'])} models and {len(config['datasets'])} datasets...")

    for model in config['models']:
        for dataset in config['datasets']:
            for seed in config['seeds']:
                file_path = (
                    base_dir
                    .joinpath(model)
                    .joinpath(dataset)
                    .joinpath(f"seed_{seed}")
                    .joinpath("timing.json")
                )
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                            data['model'] = model
                            data['dataset'] = dataset
                            data['seed'] = seed
                            
                            data['total_training_s'] = data.get('training_class_0_s', 0) + data.get('training_class_1_s', 0)
                            data['total_sampling_s'] = data.get('sampling_class_0_s', 0) + data.get('sampling_class_1_s', 0)
                            
                            results.append(data)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                else:
                    missing_count += 1

    df = pd.DataFrame(results)
    
    if not df.empty:
        cols = ['model', 'dataset', 'seed', 'total_training_s', 'total_sampling_s', 
                'training_class_0_s', 'training_class_1_s', 'sampling_class_0_s', 'sampling_class_1_s']
        df = df[cols]
        
        output_path = Path("results").joinpath("consolidated_timings.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("-" * 30)
        print("Process completed.")
        print(f"Rows found: {len(df)}")
        print(f"Missing rows: {missing_count}")
        print(f"Saved file: {output_path}")
    else:
        print("No timing.json files were found. Check the base path.")

    return df

if __name__ == "__main__":
    consolidate_timings_strict()
