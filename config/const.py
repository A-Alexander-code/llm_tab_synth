"""Project constants and directory paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR.joinpath("config")
DATA_DIR = ROOT_DIR.joinpath("data")
RESULTS_DIR = ROOT_DIR.joinpath("results")
MODELS_DIR = ROOT_DIR.joinpath("models")
TRAIN_MODEL_INFO_DIR = ROOT_DIR.joinpath("train_model_info")
LOGS_DIR = ROOT_DIR.joinpath("logs")
SCRIPTS_DIR = ROOT_DIR.joinpath("scripts")
SRC_DIR = ROOT_DIR.joinpath("src")

DATA_RAW_DIR = DATA_DIR.joinpath("raw")
DATA_PROCESSED_DIR = DATA_DIR.joinpath("processed")
DATA_ORIGINAL_SOURCE_DIR = DATA_DIR.joinpath("original_source")
DATA_REVISION_A1_DIR = DATA_DIR.joinpath("processed_revision_a1")
REVISION_A1_SOURCE_FILES = {
    "heart_framingham": DATA_ORIGINAL_SOURCE_DIR.joinpath("framingham.csv"),
    "diabetes_prima_indian": DATA_ORIGINAL_SOURCE_DIR.joinpath("pima_indians_diabetes.csv"),
    "diabetic_retinopathy": DATA_ORIGINAL_SOURCE_DIR.joinpath("Retinopathy_Diabetes.csv"),
    "cervical_cancer": DATA_ORIGINAL_SOURCE_DIR.joinpath("risk_factors_cervical_cancer.csv"),
}
REVISION_A1_CODE_FILES = (
    ROOT_DIR.joinpath("main_prepare_revision_a1.py"),
    SRC_DIR.joinpath("revision_a1.py"),
    SRC_DIR.joinpath("__init__.py"),
    SRC_DIR.joinpath("logging_utils.py"),
    ROOT_DIR.joinpath("tests", "test_revision_a1.py"),
    CONFIG_DIR.joinpath("const.py"),
)

RESULTS_METRICS_DIR = RESULTS_DIR.joinpath("metrics")
RESULTS_FIGURES_DIR = RESULTS_DIR.joinpath("figures")
RESULTS_SAMPLES_DIR = RESULTS_DIR.joinpath("samples")

PARAMS_FILE = CONFIG_DIR.joinpath("params.json")
LOGGING_CONFIG = CONFIG_DIR.joinpath("logging.yaml")

METRICS_AGGREGATED_DIR = RESULTS_DIR.joinpath("metrics_aggregated")

FIGURES_DIR = ROOT_DIR.joinpath("figures")
EVALUATION_FIGURES_DIR = FIGURES_DIR.joinpath("evaluation_profiles")

UTILITY_FIGURES_DIR = EVALUATION_FIGURES_DIR.joinpath("utility")
PRIVACY_FIGURES_DIR = EVALUATION_FIGURES_DIR.joinpath("privacy")
TRADEOFF_FIGURES_DIR = EVALUATION_FIGURES_DIR.joinpath("tradeoff")

SYNTHETIC_FILENAME = "synthetic_class_{label}.csv"
TIMING_FILENAME = "timing.json"
MODEL_HASH_FILENAME = "model_hash.json"
PARAMS_SNAPSHOT_NAME = "params_snapshot.json"
FEATURE_META_FILENAME = "feature_meta.json"

X_TRAIN_FILENAME = "X_train.npy"
X_TEST_FILENAME = "X_test.npy"
Y_TRAIN_FILENAME = "y_train.npy"
Y_TEST_FILENAME = "y_test.npy"

SUPPORTED_MODELS = [
    "great",
    "tabula",
    "realtabformer",
    "ctgan",
    "tvae",
    "gaussian_copula",
    "copulagan",
    "tabpfgen",
    "forestdiffusion",
    "mostlyai",
]

DEFAULT_TARGET_COLUMN = "class"


def get_processed_dir(dataset: str, seed: int) -> Path:
    """Path to processed split: data/processed/{dataset}/seed_{seed}/"""
    return DATA_PROCESSED_DIR.joinpath(dataset).joinpath(f"seed_{seed}")


def get_model_dir(model: str, dataset: str, seed: int) -> Path:
    """Path to trained model: models/{model}/{dataset}/seed_{seed}/"""
    return MODELS_DIR.joinpath(model).joinpath(dataset).joinpath(f"seed_{seed}")


def get_model_class_dir(model: str, dataset: str, seed: int, label: int) -> Path:
    """Path to class-specific model: models/{model}/{dataset}/seed_{seed}/class_{label}/"""
    return get_model_dir(model, dataset, seed).joinpath(f"class_{label}")


def get_train_model_info_dir(model: str, dataset: str, seed: int) -> Path:
    """Path to training auxiliaries: train_model_info/{model}/{dataset}/seed_{seed}/"""
    return TRAIN_MODEL_INFO_DIR.joinpath(model).joinpath(dataset).joinpath(
        f"seed_{seed}"
    )


def get_train_model_info_class_dir(
    model: str,
    dataset: str,
    seed: int,
    label: int,
) -> Path:
    """Path to class-specific training auxiliaries."""
    return get_train_model_info_dir(model, dataset, seed).joinpath(f"class_{label}")


def get_samples_dir(model: str, dataset: str, seed: int) -> Path:
    """Path to synthetic samples: results/samples/{model}/{dataset}/seed_{seed}/"""
    return RESULTS_SAMPLES_DIR.joinpath(model).joinpath(dataset).joinpath(
        f"seed_{seed}"
    )


def get_metrics_dir(model: str, dataset: str, seed: int, ir: float) -> Path:
    """Path to evaluation metrics: results/metrics/{model}/{dataset}/seed_{seed}/ir_{ir:.2f}/"""
    return RESULTS_METRICS_DIR.joinpath(model).joinpath(dataset).joinpath(
        f"seed_{seed}"
    ).joinpath(f"ir_{ir:.2f}")


def get_figures_dir(model: str, dataset: str) -> Path:
    """Path to figures: results/figures/{model}/{dataset}/"""
    return RESULTS_FIGURES_DIR.joinpath(model).joinpath(dataset)
