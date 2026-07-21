MODEL_LABELS = {
    "great": "GReaT",
    "tabula": "TabuLa",
    "realtabformer": "REaLTabFormer",
    "ctgan": "CTGAN",
    "copulagan": "CopulaGAN",
    "tvae": "TVAE",
    "gaussian_copula": "GaussCop.",
    "forestdiffusion": "ForestDiff.",
    "tabpfgen": "TabPFGen",
    "mostlyai": "MOSTLY AI",
}

MODEL_COLORS = {
    "great": "#0057FF",
    "tabula": "#00A6FF",
    "realtabformer": "#7A00FF",
    "ctgan": "#FF2E2E",
    "copulagan": "#FF8C00",
    "tvae": "#FFD500",
    "gaussian_copula": "#00B050",
    "forestdiffusion": "#00C2A8",
    "tabpfgen": "#FF00A8",
    "mostlyai": "#8B5A00",
}

MODEL_MARKERS = {
    "great": "o",
    "tabula": "s",
    "realtabformer": "^",
    "ctgan": "D",
    "copulagan": "P",
    "tvae": "X",
    "gaussian_copula": "v",
    "forestdiffusion": "<",
    "tabpfgen": ">",
    "mostlyai": "*",
}

MODEL_FAMILIES = (
    {
        "label": "LLM-based generators",
        "models": ("great", "tabula", "realtabformer"),
        "color": "#DDEBFF",
    },
    {
        "label": "Adversarial, variational and copula generators",
        "models": ("ctgan", "copulagan", "tvae", "gaussian_copula"),
        "color": "#FFF1CC",
    },
    {
        "label": "Tree-, diffusion- and neural-engine-based generators",
        "models": ("forestdiffusion", "tabpfgen", "mostlyai"),
        "color": "#E3F8F3",
    },
)

DATASET_LABELS = {
    "heart_framingham": "Heart Framingham",
    "cervical_cancer": "Cervical cancer",
    "diabetic_retinopathy": "Diabetic retinopathy",
    "diabetes_prima_indian": "Diabetes Prima Indian",
    "kidney_disease": "Kidney disease",
}

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def classifier_label(classifier: str) -> str:
    return CLASSIFIER_LABELS.get(
        classifier,
        classifier.replace("_", " ").title(),
    )


def validate_model_styles(model_order: tuple[str, ...]) -> None:
    missing_labels = [model for model in model_order if model not in MODEL_LABELS]
    missing_colors = [model for model in model_order if model not in MODEL_COLORS]
    missing_markers = [model for model in model_order if model not in MODEL_MARKERS]

    if missing_labels:
        raise ValueError(f"Missing model labels: {', '.join(missing_labels)}.")

    if missing_colors:
        raise ValueError(f"Missing model colors: {', '.join(missing_colors)}.")

    if missing_markers:
        raise ValueError(f"Missing model markers: {', '.join(missing_markers)}.")

    family_models = {
        model
        for family in MODEL_FAMILIES
        for model in family["models"]
    }
    missing_families = [
        model for model in model_order if model not in family_models
    ]

    if missing_families:
        raise ValueError(
            f"Missing model families: {', '.join(missing_families)}."
        )
