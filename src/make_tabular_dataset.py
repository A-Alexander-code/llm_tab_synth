import argparse
import json
import os
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"MR", "Started", "Ended"}
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DEFAULT_INPUT = PROJECT_ROOT.joinpath("data", "raw", "df_static_101.csv")
DEFAULT_OUTPUT = PROJECT_ROOT.joinpath("data", "raw", "df_static_101_tabular.csv")
DEFAULT_DESCRIPTION = PROJECT_ROOT.joinpath("config", "datasets_description.json")
DATASET_KEY = "static_101"
TARGET_COLUMN = "class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tabular version of df_static_101.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, type=Path)
    parser.add_argument("--update-description", action="store_true")
    parser.add_argument("--mappings-output", default=None, type=Path)
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def drop_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()


def create_delta_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce")
    df["Ended"] = pd.to_datetime(df["Ended"], errors="coerce")
    df["delta_time"] = (df["Ended"] - df["Started"]).dt.total_seconds() / 86400
    df = df[df["delta_time"] > 0].copy()
    return df.drop(columns=["Started", "Ended"])


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"MR": TARGET_COLUMN})
    unique_values = sorted(df[TARGET_COLUMN].dropna().unique())

    if len(unique_values) != 2:
        raise ValueError(f"Target column must have exactly two classes. Found: {unique_values}")

    mapping = {value: index for index, value in enumerate(unique_values)}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(mapping).astype(int)

    return df


def encode_non_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    mappings = {}

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = sorted(df[column].dropna().astype(str).unique())
        mapping = {value: index for index, value in enumerate(values)}
        df[column] = df[column].astype(str).map(mapping)
        mappings[column] = mapping

    return df, mappings


def move_target_to_end(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in df.columns if column != TARGET_COLUMN]
    return df[columns + [TARGET_COLUMN]]


def build_dataset_description(df: pd.DataFrame, mappings: dict) -> dict:
    numeric_int = []
    numeric_float = []
    binary = []
    categorical = []

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue

        if column in mappings:
            n_values = df[column].dropna().nunique()

            if n_values == 2:
                binary.append(column)
            else:
                categorical.append(column)

        elif pd.api.types.is_integer_dtype(df[column]):
            numeric_int.append(column)

        elif pd.api.types.is_float_dtype(df[column]):
            numeric_float.append(column)

    description = {}

    if numeric_int:
        description["numeric_int"] = numeric_int

    if numeric_float:
        description["numeric_float"] = numeric_float

    if binary:
        description["binary"] = binary

    if categorical:
        description["categorical"] = categorical

    description["target"] = TARGET_COLUMN

    return description


def update_dataset_description(path: Path, dataset_description: dict) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            descriptions = json.load(file)
    else:
        descriptions = {}

    descriptions[DATASET_KEY] = dataset_description

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(descriptions, file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_mappings(path: Path, mappings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(mappings, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    validate_columns(df)

    df = drop_index_columns(df)
    df = create_delta_time(df)
    df = encode_target(df)
    df, mappings = encode_non_numeric_columns(df)
    df = move_target_to_end(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    mappings_output = args.mappings_output or args.output.with_name(
        f"{args.output.stem}_categorical_mappings.json"
    )
    save_mappings(mappings_output, mappings)

    if args.update_description:
        dataset_description = build_dataset_description(df, mappings)
        update_dataset_description(args.description, dataset_description)

    print(f"Saved dataset: {args.output}")
    print(f"Saved mappings: {mappings_output}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()