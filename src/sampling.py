from __future__ import annotations

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split as sk_train_test_split


def split_by_class(
    df: pd.DataFrame,
    target_col: str,
) -> dict[int, pd.DataFrame]:
    """
    Split DataFrame into subsets by unique values of target_col.

    Returns:
        Dictionary {label: DataFrame} ordered by class size (largest first).
    """
    counts = df[target_col].value_counts()
    return {
        int(label): df[df[target_col] == label].reset_index(drop=True)
        for label in counts.index
    }


def apply_imbalance_ratio(
    class_dfs: dict[int, pd.DataFrame],
    ir: float,
    target_col: str,
    seed: int,
) -> dict[int, pd.DataFrame]:
    """
    Apply imbalance ratio (IR) to dataset using Random Undersampling.
    IR = n_minority / n_majority. IR=1.0 → balanced classes.

    The majority class is randomly undersampled to achieve the desired IR.
    The minority class remains unchanged.

    Args:
        class_dfs: Dictionary {label: DataFrame} from split_by_class
        ir: Desired minority/majority ratio (0 < ir <= 1.0)
        target_col: Target column name
        seed: Random seed for reproducibility

    Returns:
        Dictionary {label: DataFrame} with IR applied.
    """
    if not (0 < ir <= 1.0):
        raise ValueError(f"IR must be in (0, 1]. Got: {ir}")

    combined_df = pd.concat(
        [df.assign(**{target_col: lbl}) for lbl, df in class_dfs.items()],
        ignore_index=True
    )

    X = combined_df.drop(columns=[target_col])
    y = combined_df[target_col]

    sizes = y.value_counts()
    min_size = sizes.min()
    maj_size = sizes.max()

    target_maj_size = int(round(min_size / ir))
    target_maj_size = min(target_maj_size, maj_size)

    sampling_strategy = {
        label: target_maj_size if sizes[label] == maj_size else sizes[label]
        for label in sizes.index
    }

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    result_df = X_resampled.copy()
    result_df[target_col] = y_resampled

    result = {}
    for label in class_dfs.keys():
        result[label] = result_df[result_df[target_col] == label].drop(
            columns=[target_col]
        ).reset_index(drop=True)

    return result


def train_test_split_by_class(
    class_dfs: dict[int, pd.DataFrame],
    test_size: float,
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """
    Perform stratified train/test split per class.

    Args:
        class_dfs: Dictionary {label: DataFrame}
        test_size: Test fraction (e.g., 0.2)
        seed: Random seed for reproducibility

    Returns:
        Tuple (train_dfs, test_dfs) where each is a dict {label: DataFrame}.
    """
    train_dfs: dict[int, pd.DataFrame] = {}
    test_dfs:  dict[int, pd.DataFrame] = {}

    for label, df in class_dfs.items():
        if len(df) < 2:
            raise ValueError(
                f"Class {label} has only {len(df)} samples. "
                "Cannot perform train/test split."
            )
        train, test = sk_train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
        )
        train_dfs[label] = train.reset_index(drop=True)
        test_dfs[label]  = test.reset_index(drop=True)

    return train_dfs, test_dfs