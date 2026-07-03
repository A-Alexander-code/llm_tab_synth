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

LLM_ONLY_MODELS = [
    "great",
    "tabula",
    "realtabformer",
]

METRIC_REGISTRY = {
    "record_similarity": {
        "block": "privacy_record_similarity",
        "requires_base_metrics": False,
        "requires_temperature": False,
    },
    "semantic_similarity": {
        "block": "privacy_semantic_similarity",
        "requires_base_metrics": False,
        "requires_temperature": False,
    },
    "ngram_overlap": {
        "block": "privacy_ngram_overlap",
        "requires_base_metrics": False,
        "requires_temperature": False,
    },
    "memorization": {
        "block": "privacy_memorization_proxy",
        "requires_base_metrics": False,
        "requires_temperature": False,
    },
    "cooccurrence": {
        "block": "utility_cooccurrence",
        "requires_base_metrics": False,
        "requires_temperature": False,
    },
    "tradeoff": {
        "block": "privacy_utility_tradeoff",
        "requires_base_metrics": True,
        "requires_temperature": False,
    },
    "temperature": {
        "block": "privacy_temperature_sensitivity",
        "requires_base_metrics": True,
        "requires_temperature": True,
    },
}

METRIC_ALIASES = {
    "llm_privacy": [
        "record_similarity",
        "semantic_similarity",
        "ngram_overlap",
        "memorization",
    ],
    "all": [
        "record_similarity",
        "semantic_similarity",
        "ngram_overlap",
        "memorization",
        "cooccurrence",
        "tradeoff",
    ],
}

EXPECTED_METRICS = {
    "record_similarity": [
        "record_sim_max_mean",
        "record_sim_max_median",
        "record_sim_max_p95",
        "record_sim_max_p99",
        "record_sim_high_ratio_090",
        "record_sim_high_ratio_095",
        "record_sim_high_ratio_099",
        "record_sim_exactish_ratio",
    ],
    "semantic_similarity": [
        "semantic_sim_max_mean",
        "semantic_sim_max_median",
        "semantic_sim_max_p95",
        "semantic_sim_max_p99",
        "semantic_sim_high_ratio_090",
        "semantic_sim_high_ratio_095",
        "semantic_sim_high_ratio_099",
    ],
    "ngram_overlap": [
        "ngram2_overlap_mean",
        "ngram2_overlap_median",
        "ngram2_overlap_p95",
        "ngram2_train_vocab_reuse_ratio",
        "ngram2_synthetic_vocab_seen_ratio",
        "ngram2_rare_reuse_ratio",
        "ngram2_rare_reuse_per_record_mean",
        "ngram3_overlap_mean",
        "ngram3_overlap_median",
        "ngram3_overlap_p95",
        "ngram3_train_vocab_reuse_ratio",
        "ngram3_synthetic_vocab_seen_ratio",
        "ngram3_rare_reuse_ratio",
        "ngram3_rare_reuse_per_record_mean",
    ],
    "memorization": [
        "memrisk_output_score",
        "memrisk_components_available",
    ],
    "cooccurrence": [
        "cooc_l1_mean",
        "cooc_jsd_mean",
        "cooc_l2_frobenius_normed",
        "cooc_max_abs_diff",
        "cooc_top10_pair_overlap",
        "cooc_spearman_pairwise_strength",
    ],
    "tradeoff": [
        "tradeoff_utility_score",
        "tradeoff_privacy_score",
        "tradeoff_balance_score",
        "tradeoff_min_score",
    ],
}

TEMPERATURE_TRACKING = {
    "record_similarity": ["record_sim_high_ratio_095"],
    "semantic_similarity": ["semantic_sim_high_ratio_095"],
    "ngram_overlap": ["ngram2_rare_reuse_ratio", "ngram3_rare_reuse_ratio"],
    "memorization": ["memrisk_output_score"],
    "cooccurrence": ["cooc_jsd_mean"],
    "tradeoff": ["tradeoff_balance_score"],
}


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
