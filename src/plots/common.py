import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.plots.styles import MODEL_COLORS, MODEL_FAMILIES


def draw_family_bands(ax, model_order: tuple[str, ...]) -> None:
    for family in MODEL_FAMILIES:
        indices = [
            model_order.index(model)
            for model in family["models"]
            if model in model_order
        ]

        if not indices:
            continue

        start = min(indices) - 0.5
        end = max(indices) + 0.5

        ax.axvspan(
            start,
            end,
            facecolor=family["color"],
            alpha=0.55,
            zorder=0,
        )
        ax.axvline(
            end,
            color="white",
            linewidth=2.0,
            zorder=1,
        )


def style_boxplot(boxplot, model_order: tuple[str, ...]) -> None:
    for box, model in zip(boxplot["boxes"], model_order):
        box.set(
            facecolor=MODEL_COLORS[model],
            edgecolor="black",
            alpha=0.90,
            linewidth=0.85,
            zorder=3,
        )

    for key in ("whiskers", "caps"):
        for item in boxplot[key]:
            item.set(
                color="black",
                linewidth=0.8,
                zorder=3,
            )

    for median in boxplot["medians"]:
        median.set(
            color="black",
            linewidth=1.15,
            zorder=4,
        )


def finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def save_figure(
    figure: Figure,
    output_dir: Path,
    basename: str,
    png_dpi: int,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir.joinpath(f"{basename}.png")
    pdf_path = output_dir.joinpath(f"{basename}.pdf")

    try:
        figure.savefig(
            png_path,
            dpi=png_dpi,
            bbox_inches="tight",
        )
        logger.info("Saved PNG: %s", png_path)

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        logger.info("Saved PDF: %s", pdf_path)
    finally:
        plt.close(figure)

    return png_path, pdf_path
