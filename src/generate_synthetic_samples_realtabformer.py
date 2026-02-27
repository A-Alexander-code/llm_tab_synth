from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from realtabformer import REaLTabFormer

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"

OUTPUT_BASE = Path("synth_samples")

TARGET_COLUMN = "Outcome"
N_SAMPLES = 200

RTF_PARAMS: dict = {
    "model_type": "tabular",
    "epochs": 100,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "early_stopping_patience": 5,
    "random_state": 42,
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
    rtf_params: dict,
) -> None:
    logger.info("Training REaLTabFormer for class '%s' (%d training rows)...", label, len(df))

    model = REaLTabFormer(
        model_type=rtf_params["model_type"],
        epochs=rtf_params["epochs"],
        batch_size=rtf_params["batch_size"],
        gradient_accumulation_steps=rtf_params["gradient_accumulation_steps"],
        early_stopping_patience=rtf_params["early_stopping_patience"],
        random_state=rtf_params["random_state"],
    )
    model.fit(df)

    logger.info("Generating %d synthetic samples for class '%s'...", n_samples, label)
    synthetic_df = model.sample(n_samples=n_samples)

    out_path = out_dir / f"synthetic_class_{str(label).replace(' ', '_').lower()}.csv"
    synthetic_df.to_csv(out_path, index=False)
    logger.info("Samples saved to: %s", out_path)


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
        out_dir = build_output_dir(label, run_id)
        train_and_sample(
            df=subset_df,
            label=label,
            out_dir=out_dir,
            n_samples=N_SAMPLES,
            rtf_params=RTF_PARAMS,
        )

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()