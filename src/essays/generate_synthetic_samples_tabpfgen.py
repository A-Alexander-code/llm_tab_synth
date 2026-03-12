import os
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tabpfgen import TabPFGen


os.environ["HF_TOKEN"] = "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" 


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"
OUTPUT_BASE = Path("synth_samples")
TARGET_COLUMN = "Outcome"
N_SAMPLES_PER_CLASS = 200 

TABPFGEN_PARAMS = {
    "n_sgld_steps": 500,
    "sgld_step_size": 0.01,
    "sgld_noise_scale": 0.01,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.read_csv(INPUT_FILE)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)
    unique_classes = np.unique(y)
    
    collected_samples = pd.DataFrame(columns=df.columns)
    generator = TabPFGen(**TABPFGEN_PARAMS)

    iteration = 1
    while True:
        counts = collected_samples[TARGET_COLUMN].value_counts()
        missing_per_class = {}
        
        for cls in unique_classes:
            current_count = counts.loc[cls] if cls in counts.index else 0
            missing_per_class[int(cls)] = max(0, N_SAMPLES_PER_CLASS - int(current_count))
        
        total_missing = sum(missing_per_class.values())
        if total_missing <= 0:
            break

        logger.info("Iteration %d | Missing: %s", iteration, missing_per_class)
        
        batch_size = max(total_missing + 10, 50) 
        X_batch, y_batch = generator.generate_classification(
            X, y, n_samples=batch_size, balance_classes=True
        )

        batch_df = pd.DataFrame(X_batch, columns=feature_cols)
        batch_df[TARGET_COLUMN] = y_batch
        
        collected_samples = pd.concat([collected_samples, batch_df], ignore_index=True)
        iteration += 1

    for label in unique_classes:
        class_df = collected_samples[collected_samples[TARGET_COLUMN] == label].head(N_SAMPLES_PER_CLASS)
        out_dir = OUTPUT_BASE / run_id / f"class_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        class_df.to_csv(out_dir / f"synthetic_class_{label}.csv", index=False)
        logger.info("Saved %d samples for class %s", len(class_df), label)

if __name__ == "__main__":
    main()