from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

TAPTAP_DIR = Path("src/models/taptap")
if str(TAPTAP_DIR) not in sys.path:
    sys.path.insert(0, str(TAPTAP_DIR))

from taptap import Taptap

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"
OUTPUT_BASE = Path("synth_samples")
TARGET_COLUMN = "Outcome"
N_SAMPLES = 200

TAPTAP_PARAMS: dict = {
    "llm": "ztphs980/taptap-distilgpt2",
    "experiment_dir": "./experiment_taptap/",
    "steps": 1000,
    "batch_size": 8,
    "numerical_modeling": "split",
    "gradient_accumulation_steps": 2,
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
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
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
        majority_label, len(df_majority), minority_label, len(df_minority),
    )
    return df_majority, df_minority


def build_output_dir(label: str, run_id: str) -> Path:
    safe_label = str(label).replace(" ", "_").lower()
    out_dir = OUTPUT_BASE / run_id / f"class_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def train_and_sample(df, label, out_dir, n_samples, params):
    logger.info("Training TapTap for class '%s' (%d rows)...", label, len(df))

    model = Taptap(
        llm=params["llm"],
        experiment_dir=params["experiment_dir"],
        steps=params["steps"],
        batch_size=params["batch_size"],
        numerical_modeling=params["numerical_modeling"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
    )
    model.fit(df, target_col=TARGET_COLUMN, task="binclass")

    logger.info("Generating %d samples for class '%s'...", n_samples, label)
    synthetic_df = model.sample(
        n_samples=n_samples,
        data=df,
        task="binclass",
        max_length=1024,
    )

    synthetic_df[TARGET_COLUMN] = label

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
        out_dir = build_output_dir(str(label), run_id)
        train_and_sample(subset_df, label, out_dir, N_SAMPLES, TAPTAP_PARAMS)

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()