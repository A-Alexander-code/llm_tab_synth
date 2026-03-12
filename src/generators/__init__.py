"""Registry of available synthetic data generators and factory function."""

from src.generators.base import BaseGenerator

_GENERATOR_CLASSES = {
    "great": ("src.generators.great", "GreatGenerator"),
    "tabula": ("src.generators.tabula", "TabulaGenerator"),
    "realtabformer": ("src.generators.realtabformer", "RealtabformerGenerator"),
    "ctgan": ("src.generators.ctgan", "CtganGenerator"),
    "tvae": ("src.generators.tvae", "TvaeGenerator"),
    "gaussian_copula": ("src.generators.gaussian_copula", "GaussianCopulaGenerator"),
    "copulagan": ("src.generators.copulagan", "CopulaganGenerator"),
    "tabpfgen": ("src.generators.tabpfgen", "TabpfgenGenerator"),
    "forestdiffusion": ("src.generators.forestdiffusion", "ForestDiffusionGenerator"),
    "mostlyai": ("src.generators.mostlyai", "MostlyAIGenerator"),
}

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {}


def _load_generator_class(model_name: str) -> type[BaseGenerator]:
    """Lazy-load generator class to avoid import errors for missing dependencies."""
    if model_name in GENERATOR_REGISTRY:
        return GENERATOR_REGISTRY[model_name]
    
    if model_name not in _GENERATOR_CLASSES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(_GENERATOR_CLASSES.keys())}"
        )
    
    module_path, class_name = _GENERATOR_CLASSES[model_name]
    try:
        module = __import__(module_path, fromlist=[class_name])
        generator_class = getattr(module, class_name)
        GENERATOR_REGISTRY[model_name] = generator_class
        return generator_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load generator '{model_name}'. "
            f"Make sure the required dependencies are installed. "
            f"Error: {e}"
        )


def get_generator(model_name: str, params: dict) -> BaseGenerator:
    """Instantiate and return the generator for the given model name."""
    generator_class = _load_generator_class(model_name)
    return generator_class(params)


__all__ = ["GENERATOR_REGISTRY", "get_generator", "BaseGenerator"]