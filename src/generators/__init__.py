"""Registry of available synthetic data generators and factory function."""

from src.generators.base import BaseGenerator
from src.generators.great import GreatGenerator
from src.generators.tabula import TabulaGenerator
from src.generators.realtabformer import RealtabformerGenerator
from src.generators.ctgan import CtganGenerator
from src.generators.tvae import TvaeGenerator
from src.generators.gaussian_copula import GaussianCopulaGenerator
from src.generators.copulagan import CopulaganGenerator
from src.generators.tabpfgen import TabpfgenGenerator
from src.generators.forestdiffusion import ForestDiffusionGenerator
from src.generators.mostlyai import MostlyAIGenerator

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    "great": GreatGenerator,
    "tabula": TabulaGenerator,
    "realtabformer": RealtabformerGenerator,
    "ctgan": CtganGenerator,
    "tvae": TvaeGenerator,
    "gaussian_copula": GaussianCopulaGenerator,
    "copulagan": CopulaganGenerator,
    "tabpfgen": TabpfgenGenerator,
    "forestdiffusion": ForestDiffusionGenerator,
    "mostlyai": MostlyAIGenerator,
}


def get_generator(model_name: str, params: dict) -> BaseGenerator:
    """Instantiate and return the generator for the given model name."""
    if model_name not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(GENERATOR_REGISTRY.keys())}"
        )
    return GENERATOR_REGISTRY[model_name](params)


__all__ = ["GENERATOR_REGISTRY", "get_generator", "BaseGenerator"]