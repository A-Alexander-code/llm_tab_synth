from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ForestDiffusion import ForestDiffusionModel

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"

OUTPUT_BASE = Path("synth_samples")

TARGET_COLUMN = "Outcome"
N_SAMPLES_PER_CLASS = 200

FORESTDIFFUSION_PARAMS: dict = {
    "n_t": 50,
    "duplicate_K": 100,
    "n_jobs": -1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    df = pd.read_csv(INPUT_FILE)
    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)

    counts = df[TARGET_COLUMN].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    unique_classes = sorted(df[TARGET_COLUMN].unique().tolist())

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    logger.info("Training ForestDiffusion on full dataset...")
    model = ForestDiffusionModel(
        X,
        label_y=y,
        n_t=FORESTDIFFUSION_PARAMS["n_t"],
        duplicate_K=FORESTDIFFUSION_PARAMS["duplicate_K"],
        n_jobs=FORESTDIFFUSION_PARAMS["n_jobs"],
    )

    collected = pd.DataFrame(columns=df.columns)
    iteration = 1

    while True:
        class_counts = collected[TARGET_COLUMN].value_counts() if len(collected) > 0 else pd.Series(dtype=int)
        missing = {cls: max(0, N_SAMPLES_PER_CLASS - int(class_counts.get(cls, 0))) for cls in unique_classes}
        total_missing = sum(missing.values())

        if total_missing <= 0:
            break

        logger.info("Iteration %d | Missing: %s", iteration, missing)
        batch_size = max(total_missing + 20, 50)

        X_synth = model.generate(batch_size=batch_size)

        if isinstance(X_synth, np.ndarray) and X_synth.ndim == 2:
            n_features = len(feature_cols)
            if X_synth.shape[1] == n_features + 1:
                X_part = X_synth[:, :n_features]
                y_part = X_synth[:, n_features].astype(int)
            else:
                X_part = X_synth
                y_part = np.full(len(X_synth), unique_classes[0], dtype=int)
        else:
            X_part = X_synth
            y_part = np.full(len(X_synth), unique_classes[0], dtype=int)

        batch_df = pd.DataFrame(X_part, columns=feature_cols)
        batch_df[TARGET_COLUMN] = y_part
        collected = pd.concat([collected, batch_df], ignore_index=True)
        iteration += 1

    for label in unique_classes:
        class_df = collected[collected[TARGET_COLUMN] == label].head(N_SAMPLES_PER_CLASS)
        out_dir = OUTPUT_BASE / run_id / f"class_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synthetic_class_{label}.csv"
        class_df.to_csv(out_path, index=False)
        logger.info("Saved %d samples for class %s to: %s", len(class_df), label, out_path)

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()