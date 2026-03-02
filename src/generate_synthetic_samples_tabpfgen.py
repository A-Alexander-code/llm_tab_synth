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
N_SAMPLES = 200

TABPFGEN_PARAMS: dict = {
    "n_sgld_steps": 500,
    "step_size": 0.01,
    "noise_scale": 0.01,
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


def generate_for_class(
    df: pd.DataFrame,
    label: int,
    out_dir: Path,
    n_samples: int,
    params: dict,
) -> None:
    logger.info("Generating %d synthetic samples for class '%s'...", n_samples, label)

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    all_labels = np.unique(y)

    generator = TabPFGen(
        n_sgld_steps=params["n_sgld_steps"],
        step_size=params["step_size"],
        noise_scale=params["noise_scale"],
    )

    target_per_class = {int(lbl): (n_samples if int(lbl) == int(label) else 0) for lbl in all_labels}

    X_synth, y_synth, _, _ = generator.balance_dataset(
        X, y, target_per_class={int(label): n_samples}
    )

    mask = y_synth == int(label)
    X_class = X_synth[mask]
    y_class = y_synth[mask]

    synthetic_df = pd.DataFrame(X_class, columns=feature_cols)
    synthetic_df[TARGET_COLUMN] = y_class

    out_path = out_dir / f"synthetic_class_{str(label).replace(' ', '_').lower()}.csv"
    synthetic_df.to_csv(out_path, index=False)
    logger.info("Saved %d samples to: %s", len(synthetic_df), out_path)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    df = load_data(INPUT_FILE)
    df_majority, df_minority = split_by_class(df)

    majority_label = int(df_majority[TARGET_COLUMN].iloc[0])
    minority_label = int(df_minority[TARGET_COLUMN].iloc[0])

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    generator = TabPFGen(
        n_sgld_steps=TABPFGEN_PARAMS["n_sgld_steps"],
        step_size=TABPFGEN_PARAMS["step_size"],
        noise_scale=TABPFGEN_PARAMS["noise_scale"],
    )

    for label in [majority_label, minority_label]:
        logger.info("Generating %d samples for class '%s'...", N_SAMPLES, label)

        X_synth, y_synth, _, _ = generator.balance_dataset(
            X, y, target_per_class={label: N_SAMPLES}
        )

        mask = y_synth == label
        X_class = X_synth[mask]
        y_class = y_synth[mask]

        out_dir = build_output_dir(str(label), run_id)
        synthetic_df = pd.DataFrame(X_class, columns=feature_cols)
        synthetic_df[TARGET_COLUMN] = y_class

        out_path = out_dir / f"synthetic_class_{label}.csv"
        synthetic_df.to_csv(out_path, index=False)
        logger.info("Saved %d samples to: %s", len(synthetic_df), out_path)

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()