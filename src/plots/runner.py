"""Dispatch the publication figures reproduced from the version 4 notebook."""
import logging

import matplotlib.pyplot as plt

from config.const import ROOT_DIR
from src.plots import utility, privacy, tradeoff, supplementary_tradeoff, detectability, timing
from src.plots.settings import PlotSettings


def run_plot_command(
    command: str,
    settings: PlotSettings,
    datasets: tuple[str, ...],
    classifier: str,
    include_title: bool,
    logger: logging.Logger,
) -> None:
    classifiers = (
        settings.supported_classifiers if classifier == "all" else (classifier,)
    )
    modules = {
        "utility": (utility,),
        "privacy": (privacy,),
        "tradeoff": tuple(
            module for enabled, module in (
                ("xgboost" in classifiers, tradeoff),
                (any(c != "xgboost" for c in classifiers), supplementary_tradeoff),
            ) if enabled
        ),
        "detectability": (detectability,),
        "timing": (timing,),
    }
    selected = tuple(modules) if command == "all" else (command,)
    for category in selected:
        for module in modules[category]:
            logger.info("Generating %s", module.__name__)
            # Notebook style settings are scoped to each figure family.
            figures_before = set(plt.get_fignums())
            try:
                with plt.rc_context():
                    module.run(
                        datasets, classifiers, ROOT_DIR,
                        settings.png_dpi, include_title,
                    )
            finally:
                for number in set(plt.get_fignums()) - figures_before:
                    plt.close(number)
