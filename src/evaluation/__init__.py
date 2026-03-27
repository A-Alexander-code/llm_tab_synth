from src.evaluation.builder import build_synthetic_dataset, get_class_column
from src.evaluation.utility import (
    compute_univariate, compute_multivariate,
    compute_detectability, compute_ml_efficacy,
)
from src.evaluation.privacy import (
    compute_exact_matches, compute_dcr, compute_dcr_holdout,
    compute_mia, compute_attribute_inference,
    compute_nnaa, compute_kanonymity,
)
from src.evaluation.degenerate import (
    detect_degenerate_columns,
    get_valid_columns_for_statistical_metrics,
    compute_degenerate_diagnostics,
    build_degenerate_audit_rows,
)
from src.evaluation.aggregator import (
    build_rows, save_combination_csv, append_to_master,
    save_aggregated, save_degenerate_audit, make_not_computable_row,
)

__all__ = [
    "build_synthetic_dataset", "get_class_column",
    "compute_univariate", "compute_multivariate",
    "compute_detectability", "compute_ml_efficacy",
    "compute_exact_matches", "compute_dcr", "compute_dcr_holdout",
    "compute_mia", "compute_attribute_inference",
    "compute_nnaa", "compute_kanonymity",
    "detect_degenerate_columns", "get_valid_columns_for_statistical_metrics",
    "compute_degenerate_diagnostics", "build_degenerate_audit_rows",
    "build_rows", "save_combination_csv", "append_to_master",
    "save_aggregated", "save_degenerate_audit", "make_not_computable_row",
]