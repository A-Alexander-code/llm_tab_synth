"""
src/evaluation/builder.py

Construcción del dataset sintético final con un output_ir dado.

Definición:
    IR = n_min / n_maj
    n_maj = n  (fijo, todas las muestras de clase mayoritaria)
    n_min = floor(IR * n)

Propiedad de subconjuntos anidados:
    La clase minoritaria se baraja una sola vez con la seed del experimento.
    Para cada IR se toman las primeras k filas de esa permutación.
    Esto garantiza que el conjunto para IR=0.5 ⊆ IR=0.6 ⊆ ... ⊆ IR=1.0

IMPORTANTE — etiquetas reales:
    build_synthetic_dataset recibe maj_label y min_label para asignar
    la etiqueta correcta a cada clase. No se asume que la mayoritaria
    sea siempre la clase 0 (esto varía según el dataset).
"""

from __future__ import annotations

import math

import pandas as pd


def build_synthetic_dataset(
    synth_maj: pd.DataFrame,
    synth_min: pd.DataFrame,
    output_ir: float,
    seed: int,
    maj_label: int = 0,
    min_label: int = 1,
) -> pd.DataFrame:
    """
    Construye el dataset sintético final para un output_ir dado.

    Args:
        synth_maj:  DataFrame de muestras sintéticas de la clase mayoritaria.
        synth_min:  DataFrame de muestras sintéticas de la clase minoritaria.
        output_ir:  Ratio deseado IR = n_min / n_maj (0 < output_ir <= 1.0).
        seed:       Seed para barajar la clase minoritaria.
        maj_label:  Etiqueta real de la clase mayoritaria (0 o 1).
        min_label:  Etiqueta real de la clase minoritaria (0 o 1).

    Returns:
        DataFrame concatenado y barajado con la proporción output_ir solicitada.
        Incluye la columna 'class' con las etiquetas reales del dataset.
        El índice se resetea.
    """
    if not (0 < output_ir <= 1.0):
        raise ValueError(f"output_ir must be in (0, 1]. Got {output_ir}")

    n_maj = len(synth_maj)
    n_min = math.floor(output_ir * n_maj)
    n_min = max(1, min(n_min, len(synth_min)))

    # Barajar la minoritaria con seed fija para garantizar subconjuntos anidados
    synth_min_shuffled = synth_min.sample(frac=1, random_state=seed).reset_index(drop=True)
    synth_min_subset   = synth_min_shuffled.iloc[:n_min].copy()

    # Asignar etiquetas reales del dataset
    synth_maj_labeled        = synth_maj.copy()
    synth_maj_labeled["class"] = maj_label
    synth_min_subset["class"]  = min_label

    combined = pd.concat([synth_maj_labeled, synth_min_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


def get_class_column(df: pd.DataFrame, target: str = "class") -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target. Devuelve (X, y)."""
    y = df[target]
    X = df.drop(columns=[target])
    return X, y