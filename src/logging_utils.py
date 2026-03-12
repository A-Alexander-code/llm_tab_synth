"""
Centralized logging system configuration.
Injects the log file path at runtime from main_*.py scripts.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(
    config_path: Path,
    log_file: Path,
    logger_name: str = "sampler",
) -> logging.Logger:
    """
    Configure logging system from a YAML configuration file.
    Overrides the log file path at runtime for each execution.

    Args:
        config_path: Path to logging.yaml configuration file
        log_file: Path where execution logs will be written
        logger_name: Logger name to return (sampler/evaluator/plotter)

    Returns:
        Configured Logger ready to use.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Logging config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    error_file = log_file.parent.joinpath("errors.log")

    config["handlers"]["file_main"]["filename"]   = str(log_file)
    config["handlers"]["file_errors"]["filename"] = str(error_file)

    logging.config.dictConfig(config)
    return logging.getLogger(logger_name)