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
N_SAMPLES = 200

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


def split_by_class(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df[TARGET_COLUMN].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    df_majority = df[df[TARGET_COLUMN] == majority_label].reset_index(drop=True)
    df_minority = df[df[TARGET_COLUMN] == minority_label].reset_index(drop=True)

    logger.info(
        "Majority class: '%s' (%d samples) | Minority class: '%s' (%d samples)",
        majority_label, len(df_majority),
        minority_label, len(df_minority),
    )
    return df_majority, df_minority


def build_output_dir(label: str, run_id: str) -> Path:
    safe_label = str(label).replace(" ", "_").lower()
    out_dir = OUTPUT_BASE / run_id / f"class_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def train_and_sample(
    df: pd.DataFrame,
    label: str,
    out_dir: Path,
    n_samples: int,
    params: dict,
) -> None:
    logger.info("Training ForestDiffusion for class '%s' (%d training rows)...", label, len(df))

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    model = ForestDiffusionModel(
        X,
        label_y=y,
        n_t=params["n_t"],
        duplicate_K=params["duplicate_K"],
        n_jobs=params["n_jobs"],
    )

    logger.info("Generating %d synthetic samples for class '%s'...", n_samples, label)
    X_synth, y_synth = model.generate(batch_size=n_samples)

    synthetic_df = pd.DataFrame(X_synth, columns=feature_cols)
    synthetic_df[TARGET_COLUMN] = y_synth

    synthetic_df = synthetic_df[synthetic_df[TARGET_COLUMN] == label].head(n_samples)

    out_path = out_dir / f"synthetic_class_{str(label).replace(' ', '_').lower()}.csv"
    synthetic_df.to_csv(out_path, index=False)
    logger.info("Saved %d samples to: %s", len(synthetic_df), out_path)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    df = load_data(INPUT_FILE)
    df_majority, df_minority = split_by_class(df)

    majority_label = df_majority[TARGET_COLUMN].iloc[0]
    minority_label = df_minority[TARGET_COLUMN].iloc[0]

    for subset_df, label in [
        (df_majority, majority_label),
        (df_minority, minority_label),
    ]:
        out_dir = build_output_dir(str(label), run_id)
        train_and_sample(
            df=subset_df,
            label=label,
            out_dir=out_dir,
            n_samples=N_SAMPLES,
            params=FORESTDIFFUSION_PARAMS,
        )

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()