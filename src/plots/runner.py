import logging

from config.const import (
    METRICS_AGGREGATED_DIR,
    PRIVACY_FIGURES_DIR,
    TRADEOFF_FIGURES_DIR,
    UTILITY_FIGURES_DIR,
)
from src.plots.common import save_figure
from src.plots.data import load_aggregated_metrics
from src.plots.privacy import (
    build_privacy_table,
    create_privacy_figure,
    privacy_output_basename,
)
from src.plots.settings import PlotSettings
from src.plots.styles import validate_model_styles
from src.plots.tradeoff import (
    build_tradeoff_table,
    create_tradeoff_figure,
    tradeoff_output_basename,
)
from src.plots.utility import (
    build_utility_table,
    create_utility_figure,
    utility_output_basename,
)


def run_plot_command(
    command: str,
    settings: PlotSettings,
    datasets: tuple[str, ...],
    classifier: str,
    output_ir: float,
    include_title: bool,
    logger: logging.Logger,
) -> None:
    validate_model_styles(settings.model_order)
    raw_data = load_aggregated_metrics(
        datasets,
        METRICS_AGGREGATED_DIR,
        logger,
    )

    if command in ("utility", "all"):
        logger.info("Generating utility plot")
        utility_data = build_utility_table(
            raw_data,
            datasets,
            settings.model_order,
            classifier,
        )
        utility_figure = create_utility_figure(
            utility_data,
            datasets,
            settings.model_order,
            classifier,
            include_title,
        )
        save_figure(
            utility_figure,
            UTILITY_FIGURES_DIR,
            utility_output_basename(classifier),
            settings.png_dpi,
            logger,
        )

    if command in ("privacy", "all"):
        logger.info("Generating privacy plot")
        privacy_data = build_privacy_table(
            raw_data,
            datasets,
            settings.model_order,
        )
        privacy_figure = create_privacy_figure(
            privacy_data,
            datasets,
            settings.model_order,
            include_title,
        )
        save_figure(
            privacy_figure,
            PRIVACY_FIGURES_DIR,
            privacy_output_basename(),
            settings.png_dpi,
            logger,
        )

    if command in ("tradeoff", "all"):
        logger.info("Generating tradeoff plot")
        tradeoff_data = build_tradeoff_table(
            raw_data,
            datasets,
            settings.model_order,
            classifier,
            output_ir,
        )
        tradeoff_figure = create_tradeoff_figure(
            tradeoff_data,
            datasets,
            settings.model_order,
            classifier,
            output_ir,
            include_title,
        )
        save_figure(
            tradeoff_figure,
            TRADEOFF_FIGURES_DIR,
            tradeoff_output_basename(classifier, output_ir),
            settings.png_dpi,
            logger,
        )
