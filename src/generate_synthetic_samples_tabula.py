from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes-retin.csv"

OUTPUT_BASE = Path("synth_samples")

TABULA_REPO_URL = "https://github.com/zhao-zilong/Tabula.git"
TABULA_DIR = Path("src/models/tabula")

TARGET_COLUMN = "Outcome"
N_SAMPLES = 200

TABULA_PARAMS: dict = {
    "llm": "gpt2",
    "batch_size": 8,
    "epochs": 50,
    "max_length": 200,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_tabula(repo_url: str, target_dir: Path) -> None:
    if not target_dir.exists():
        logger.info("TabuLa not found. Cloning into %s...", target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
        )
        logger.info("TabuLa cloned successfully.")
    else:
        logger.info("TabuLa already present at %s.", target_dir)

    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))


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
    tabula_params: dict,
) -> None:
    from tabula.tabula import Tabula

    df_clean = df.copy()
    df_clean.columns = [col.replace(" ", "_") for col in df_clean.columns]

    logger.info("Training TabuLa for class '%s' (%d training rows)...", label, len(df_clean))

    model = Tabula(
        llm=tabula_params["llm"],
        batch_size=tabula_params["batch_size"],
        epochs=tabula_params["epochs"],
    )
    model.fit(df_clean)

    logger.info("Generating %d synthetic samples for class '%s'...", n_samples, label)
    synthetic_df = model.sample(
        n_samples=n_samples,
        max_length=tabula_params["max_length"],
    )

    out_path = out_dir / f"synthetic_class_{str(label).replace(' ', '_').lower()}.csv"
    synthetic_df.to_csv(out_path, index=False)
    logger.info("Samples saved to: %s", out_path)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run started | run_id: %s", run_id)

    ensure_tabula(TABULA_REPO_URL, TABULA_DIR)

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
            tabula_params=TABULA_PARAMS,
        )

    logger.info("Process complete. Results saved to: %s/%s", OUTPUT_BASE, run_id)


if __name__ == "__main__":
    main()