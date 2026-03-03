from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabebm.TabEBM import TabEBM

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"

OUTPUT_BASE = Path("synth_samples")

TARGET_COLUMN = "Outcome"
N_SAMPLES = 200

TABEBM_PARAMS: dict = {
    "max_data_size": 10000,
    "sgld_steps": 200,
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
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    df = load_data(INPUT_FILE)

    counts = df[TARGET_COLUMN].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    logger.info("Initializing TabEBM...")
    model = TabEBM(max_data_size=TABEBM_PARAMS["max_data_size"])

    logger.info("Generating %d synthetic samples per class...", N_SAMPLES)
    augmented_data = model.generate(
        X,
        y,
        num_samples=N_SAMPLES,
        sgld_steps=TABEBM_PARAMS["sgld_steps"],
    )

    for class_id, X_synth in augmented_data.items():
        label = int(class_id.replace("class_", "")) if isinstance(class_id, str) else int(class_id)

        synthetic_df = pd.DataFrame(X_synth, columns=feature_cols)
        synthetic_df[TARGET_COLUMN] = label

        out_dir = OUTPUT_BASE / run_id / f"class_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synthetic_class_{label}.csv"
        synthetic_df.to_csv(out_path, index=False)
        logger.info("Saved %d samples for class %s to: %s", len(synthetic_df), label, out_path)

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()