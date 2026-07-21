from dataclasses import dataclass
import json
from pathlib import Path

from config.const import PARAMS_FILE


@dataclass(frozen=True)
class PlotSettings:
    available_datasets: tuple[str, ...]
    default_datasets: tuple[str, ...]
    available_models: tuple[str, ...]
    model_order: tuple[str, ...]
    supported_classifiers: tuple[str, ...]
    default_classifier: str
    default_output_ir: float
    default_title: bool
    png_dpi: int


def load_plot_settings(params_file: Path = PARAMS_FILE) -> PlotSettings:
    with params_file.open("r", encoding="utf-8") as file:
        params = json.load(file)

    experiment = params.get("experiment")
    plots = params.get("plots")

    if not isinstance(experiment, dict):
        raise ValueError("Missing 'experiment' section in params.json.")

    if not isinstance(plots, dict):
        raise ValueError("Missing 'plots' section in params.json.")

    settings = PlotSettings(
        available_datasets=tuple(experiment.get("datasets", [])),
        default_datasets=tuple(plots.get("datasets", [])),
        available_models=tuple(experiment.get("models", [])),
        model_order=tuple(plots.get("model_order", [])),
        supported_classifiers=tuple(plots.get("supported_classifiers", [])),
        default_classifier=str(plots.get("default_classifier", "xgboost")),
        default_output_ir=float(plots.get("default_output_ir", 1.0)),
        default_title=bool(plots.get("default_title", False)),
        png_dpi=int(plots.get("png_dpi", 300)),
    )

    validate_plot_settings(settings)
    return settings


def validate_plot_settings(settings: PlotSettings) -> None:
    if not settings.available_datasets:
        raise ValueError("No datasets are configured in experiment.datasets.")

    if not settings.default_datasets:
        raise ValueError("No default datasets are configured in plots.datasets.")

    unsupported_datasets = [
        dataset
        for dataset in settings.default_datasets
        if dataset not in settings.available_datasets
    ]

    if unsupported_datasets:
        names = ", ".join(unsupported_datasets)
        raise ValueError(f"Unsupported default datasets: {names}.")

    if not settings.model_order:
        raise ValueError("No models are configured in plots.model_order.")

    unsupported_models = [
        model
        for model in settings.model_order
        if model not in settings.available_models
    ]

    if unsupported_models:
        names = ", ".join(unsupported_models)
        raise ValueError(f"Unsupported plot models: {names}.")

    if settings.default_classifier not in settings.supported_classifiers:
        raise ValueError(
            "plots.default_classifier must be included in "
            "plots.supported_classifiers."
        )

    if settings.default_output_ir <= 0:
        raise ValueError("plots.default_output_ir must be greater than zero.")

    if settings.png_dpi <= 0:
        raise ValueError("plots.png_dpi must be greater than zero.")
