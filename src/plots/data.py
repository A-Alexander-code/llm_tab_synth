import logging
from pathlib import Path

import pandas as pd


def load_aggregated_metrics(
    datasets: tuple[str, ...],
    data_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames = []
    missing_files = []

    logger.info("Loading aggregated metrics")

    for dataset in datasets:
        path = data_dir.joinpath(f"{dataset}_aggregated.csv")

        if not path.exists():
            missing_files.append(path)
            continue

        frame = pd.read_csv(path)
        frames.append(frame)
        logger.debug("Loaded %s rows from %s", len(frame), path)

    for path in missing_files:
        logger.warning("Missing metrics file: %s", path)

    if not frames:
        raise FileNotFoundError(
            f"No aggregated metric files were found in: {data_dir}"
        )

    data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %s aggregated metric rows", len(data))
    return data


def require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    context: str,
) -> None:
    missing = [column for column in columns if column not in data.columns]

    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing columns for {context}: {names}.")
