"""Build final synthetic datasets for a requested output imbalance ratio."""

from __future__ import annotations
import math
import pandas as pd


def build_synthetic_dataset(synth_maj: pd.DataFrame, 
                            synth_min: pd.DataFrame, 
                            output_ir: float, 
                            seed: int,) -> pd.DataFrame:
    """Build one labeled synthetic dataset for the requested output IR."""
    if not (0 < output_ir <= 1.0):
        raise ValueError(f"output_ir must be in (0, 1]. Got {output_ir}")

    n_maj = len(synth_maj)
    n_min = math.floor(output_ir * n_maj)
    n_min = max(1, min(n_min, len(synth_min)))

    synth_min_shuffled = synth_min.sample(frac=1, random_state=seed).reset_index(drop=True)
    synth_min_subset = synth_min_shuffled.iloc[:n_min].copy()

    synth_maj_labeled = synth_maj.copy()
    synth_maj_labeled["class"] = 0
    synth_min_subset["class"] = 1

    combined = pd.concat([synth_maj_labeled, synth_min_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


def get_class_column(df: pd.DataFrame, target: str = "class") -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target."""
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
