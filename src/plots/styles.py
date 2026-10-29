"""Styles and default panel order from plotting_results_ver4.ipynb."""

DATASETS = [
    "heart_framingham",
    "cervical_cancer",
    "diabetic_retinopathy",
    "diabetes_prima_indian",
]

DATASET_LABELS = {
    "heart_framingham": "Heart Framingham",
    "cervical_cancer": "Cervical Cancer",
    "diabetic_retinopathy": "Diabetic\nRetinopathy",
    "diabetes_prima_indian": "Pima Indians\nDiabetes",
}

MODEL_ORDER = [
    "great",
    "tabula",
    "realtabformer",
    "ctgan",
    "copulagan",
    "tvae",
    "gaussian_copula",
    "mostlyai",
]

MODEL_LABELS = {
    "great": "GReaT",
    "tabula": "TabuLa",
    "realtabformer": "REaLTabFormer",
    "ctgan": "CTGAN",
    "copulagan": "CopulaGAN",
    "tvae": "TVAE",
    "gaussian_copula": "Gaussian Copula",
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
    "mostlyai": "#8B5A00",
}

MODEL_FAMILIES = [
    {
        "label": "LLM-based generators",
        "models": ["great", "tabula", "realtabformer"],
        "color": "#DDEBFF",
    },
    {
        "label": "Comparison generators",
        "models": ["ctgan", "copulagan", "tvae", "gaussian_copula", "mostlyai"],
        "color": "#FFF1CC",
    },
]
