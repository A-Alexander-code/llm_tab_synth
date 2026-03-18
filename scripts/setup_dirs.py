"""
Create the complete project directory structure.
Run once after cloning the repository.

Usage:
    python scripts/setup_dirs.py
    python scripts/setup_dirs.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.const import (
    CONFIG_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_SAMPLES_DIR,
    SCRIPTS_DIR,
    SRC_DIR,
    TRAIN_MODEL_INFO_DIR,
)

DIRECTORIES = [
    CONFIG_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TRAIN_MODEL_INFO_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_SAMPLES_DIR,
    SCRIPTS_DIR,
    SRC_DIR,
]

GITKEEP_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TRAIN_MODEL_INFO_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_SAMPLES_DIR,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up the project directory structure."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each directory as it is created.",
    )
    return parser.parse_args()


def create_directories(verbose: bool = False) -> None:
    created = 0
    skipped = 0

    for directory in DIRECTORIES:
        if directory.exists():
            if verbose:
                print(f"  [skip]    {directory.relative_to(ROOT_DIR)}")
            skipped += 1
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  [created] {directory.relative_to(ROOT_DIR)}")
            created += 1

    print(f"\nDirectories created: {created} | Already existed: {skipped}")


def create_gitkeep_files(verbose: bool = False) -> None:
    """
    Create .gitkeep files in empty directories to track them with git.
    """
    for directory in GITKEEP_DIRS:
        gitkeep = directory.joinpath(".gitkeep")
        if not gitkeep.exists():
            gitkeep.touch()
            if verbose:
                print(f"  [.gitkeep] {gitkeep.relative_to(ROOT_DIR)}")


def create_gitignore(verbose: bool = False) -> None:
    """
    Create a basic .gitignore in data and models directories
    to prevent versioning large files.
    """
    ignore_rules = {
        DATA_RAW_DIR: "*.csv\n*.parquet\n*.xlsx\n*.json\n!.gitkeep\n",
        DATA_PROCESSED_DIR: "*.csv\n*.parquet\n!.gitkeep\n",
        MODELS_DIR: "*\n!.gitkeep\n",
        TRAIN_MODEL_INFO_DIR: "*\n!.gitkeep\n",
        LOGS_DIR: "*.log\n!.gitkeep\n",
    }
    for directory, rules in ignore_rules.items():
        gitignore = directory.joinpath(".gitignore")
        if not gitignore.exists():
            gitignore.write_text(rules, encoding="utf-8")
            if verbose:
                print(f"  [.gitignore] {gitignore.relative_to(ROOT_DIR)}")


def main() -> None:
    args = parse_args()
    print(f"Setting up project structure at: {ROOT_DIR}\n")

    print("Creating directories...")
    create_directories(verbose=args.verbose)

    print("\nCreating .gitkeep files...")
    create_gitkeep_files(verbose=args.verbose)

    print("\nCreating .gitignore files...")
    create_gitignore(verbose=args.verbose)

    print("\nSetup complete.")


if __name__ == "__main__":
    main()
