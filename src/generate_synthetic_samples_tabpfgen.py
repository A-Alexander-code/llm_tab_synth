from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfgen import TabPFGen


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"
OUTPUT_BASE = Path("synth_samples")

TARGET_COLUMN = "Outcome"
N_SAMPLES_PER_CLASS = 200 

TABPFGEN_PARAMS: dict = {
    "n_sgld_steps": 500,
    "sgld_step_size": 0.01,
    "sgld_noise_scale": 0.01,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)
    return df

def build_output_dir(label: str, run_id: str) -> Path:
    safe_label = str(label).replace(" ", "_").lower()
    out_dir = OUTPUT_BASE / run_id / f"class_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    df = load_data(INPUT_FILE)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)
    
    unique_classes = np.unique(y)
    
    collected_samples = pd.DataFrame(columns=df.columns)
    
    generator = TabPFGen(**TABPFGEN_PARAMS)

    iteration = 1
    while True:
        counts = collected_samples[TARGET_COLUMN].value_counts()
        missing_per_class = {
            cls: max(0, N_SAMPLES_PER_CLASS - int(counts.get(cls, 0))) 
            for cls in unique_classes
        }
        
        total_missing = sum(missing_per_class.values())
        
        if total_missing <= 0:
            logger.info("Target reached for all classes.")
            break

        logger.info("Iteration %d | Missing samples: %s", iteration, missing_per_class)
        
        batch_size = max(total_missing, 20) 
        X_batch, y_batch = generator.generate_classification(
            X, y, n_samples=batch_size, balance_classes=True
        )

        batch_df = pd.DataFrame(X_batch, columns=feature_cols)
        batch_df[TARGET_COLUMN] = y_batch
        
        collected_samples = pd.concat([collected_samples, batch_df], ignore_index=True)
        iteration += 1

    for label in unique_classes:
        class_df = collected_samples[collected_samples[TARGET_COLUMN] == label]
        class_df = class_df.head(N_SAMPLES_PER_CLASS)
        
        out_dir = build_output_dir(str(label), run_id)
        out_path = out_dir / f"synthetic_class_{label}.csv"
        
        class_df.to_csv(out_path, index=False)
        logger.info("Finalized class '%s': %d samples saved to %s", label, len(class_df), out_path)

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)

if __name__ == "__main__":
    main()