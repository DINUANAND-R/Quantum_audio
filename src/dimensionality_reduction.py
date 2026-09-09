"""
dimensionality_reduction.py

Speaker-independent dimensionality reduction for
Quantum-Audio-Emotion.

Pipeline:

    Audio Features
          ↓
    StandardScaler
          ↓
        PCA
          ↓
    Classical / Quantum Models

Important:
- Scaler is fitted ONLY on training data.
- PCA is fitted ONLY on training data.
- Test data is transformed using the fitted scaler/PCA.
- No information from test speakers is used during fitting.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    N_PCA_COMPONENTS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

# PCA values that can be tested experimentally.
# More components preserve more acoustic information,
# but quantum models require the number of components
# to match the number of qubits.
SUPPORTED_PCA_COMPONENTS = [4, 6, 8, 12, 16]


# ============================================================
# HELPERS
# ============================================================

def get_feature_columns(df):
    """
    Return all acoustic feature columns.

    Features are expected to use the naming convention:

        feat_0
        feat_1
        feat_2
        ...

    Returns
    -------
    list[str]
        Feature column names.
    """

    columns = [
        column
        for column in df.columns
        if str(column).startswith("feat_")
    ]

    if not columns:
        raise ValueError(
            "No feature columns found. "
            "Expected columns starting with 'feat_'."
        )

    return columns


def get_labels(df):
    """
    Extract integer emotion labels from a dataframe.

    Supported formats:
        label
        emotion_label
        emotion

    Returns
    -------
    np.ndarray
        Integer labels.
    """

    if "label" in df.columns:
        return df["label"].to_numpy(dtype=np.int64)

    if "emotion_label" in df.columns:
        return df["emotion_label"].to_numpy(dtype=np.int64)

    if "emotion" in df.columns:
        from src.config import EMOTION_LABEL_MAP

        labels = df["emotion"].map(EMOTION_LABEL_MAP)

        if labels.isna().any():
            unknown = df.loc[
                labels.isna(),
                "emotion"
            ].unique()

            raise ValueError(
                f"Unknown emotion labels found: {unknown}"
            )

        return labels.to_numpy(dtype=np.int64)

    raise KeyError(
        "Could not find emotion labels. "
        "Expected 'label', 'emotion_label', or 'emotion'."
    )


def _clean_features(X):
    """
    Safely clean feature matrix.

    Replaces:
        NaN
        +Inf
        -Inf

    with finite values.
    """

    X = np.asarray(X, dtype=np.float64)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X


def _validate_n_components(
    n_components,
    n_samples,
    n_features,
):
    """
    Validate PCA component count.
    """

    try:
        n_components = int(n_components)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid n_components: {n_components}"
        ) from exc

    if n_components <= 0:
        raise ValueError(
            "n_components must be greater than 0."
        )

    max_components = min(
        n_samples,
        n_features,
    )

    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} is too large. "
            f"Maximum allowed value is {max_components} "
            f"for {n_samples} samples and {n_features} features."
        )

    return n_components


def _validate_train_test_features(
    train_df,
    test_df,
):
    """
    Ensure train and test use exactly the same
    feature columns in the same order.
    """

    train_features = get_feature_columns(train_df)
    test_features = get_feature_columns(test_df)

    if train_features != test_features:
        missing_from_test = [
            feature
            for feature in train_features
            if feature not in test_features
        ]

        extra_in_test = [
            feature
            for feature in test_features
            if feature not in train_features
        ]

        raise ValueError(
            "Train/test feature mismatch.\n"
            f"Missing from test: {missing_from_test}\n"
            f"Extra in test: {extra_in_test}"
        )

    return train_features


# ============================================================
# FIT SCALER + PCA
# ============================================================

def fit_scaler_pca(
    train_df,
    n_components=N_PCA_COMPONENTS,
):
    """
    Fit StandardScaler and PCA using TRAINING DATA ONLY.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataframe.

    n_components : int
        Number of PCA components.

    Returns
    -------
    scaler : StandardScaler
    pca : PCA
    """

    features = get_feature_columns(train_df)

    X_train = train_df[
        features
    ].to_numpy(
        dtype=np.float64
    )

    X_train = _clean_features(
        X_train
    )

    n_components = _validate_n_components(
        n_components=n_components,
        n_samples=X_train.shape[0],
        n_features=X_train.shape[1],
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    pca.fit(
        X_scaled
    )

    logger.info(
        "PCA fitted: %d components",
        n_components,
    )

    logger.info(
        "Explained variance: %.4f",
        np.sum(
            pca.explained_variance_ratio_
        ),
    )

    return scaler, pca


# ============================================================
# TRANSFORM
# ============================================================

def transform(
    df,
    scaler,
    pca,
):
    """
    Transform data using an already-fitted scaler and PCA.

    IMPORTANT:
    This function NEVER fits anything.

    That prevents test-data leakage.
    """

    features = get_feature_columns(df)

    X = df[
        features
    ].to_numpy(
        dtype=np.float64
    )

    X = _clean_features(
        X
    )

    X_scaled = scaler.transform(
        X
    )

    X_pca = pca.transform(
        X_scaled
    )

    X_pca = np.asarray(
        X_pca,
        dtype=np.float64,
    )

    X_pca = np.nan_to_num(
        X_pca,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_pca


# ============================================================
# SAVE
# ============================================================

def save_scaler_pca(
    scaler,
    pca,
    n_components,
):
    """
    Save fitted scaler and PCA models.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_components = int(
        n_components
    )

    scaler_path = (
        MODELS_DIR
        / f"scaler_pca_{n_components}.pkl"
    )

    pca_path = (
        MODELS_DIR
        / f"pca_{n_components}.pkl"
    )

    config_path = (
        MODELS_DIR
        / f"pca_config_{n_components}.json"
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    with open(
        scaler_path,
        "wb",
    ) as file:
        pickle.dump(
            scaler,
            file,
        )

    # --------------------------------------------------------
    # Save PCA
    # --------------------------------------------------------

    with open(
        pca_path,
        "wb",
    ) as file:
        pickle.dump(
            pca,
            file,
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "n_components": int(
            pca.n_components_
        ),
        "n_features": int(
            pca.n_features_in_
        ),
        "explained_variance_ratio": [
            float(value)
            for value in pca.explained_variance_ratio_
        ],
        "total_explained_variance": float(
            np.sum(
                pca.explained_variance_ratio_
            )
        ),
        "random_state": RANDOM_STATE,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    logger.info(
        "Scaler saved: %s",
        scaler_path,
    )

    logger.info(
        "PCA saved: %s",
        pca_path,
    )

    return (
        scaler_path,
        pca_path,
    )


# ============================================================
# LOAD
# ============================================================

def load_scaler_pca(
    n_components=N_PCA_COMPONENTS,
):
    """
    Load previously fitted scaler and PCA.
    """

    n_components = int(
        n_components
    )

    scaler_path = (
        MODELS_DIR
        / f"scaler_pca_{n_components}.pkl"
    )

    pca_path = (
        MODELS_DIR
        / f"pca_{n_components}.pkl"
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA not found: {pca_path}"
        )

    with open(
        scaler_path,
        "rb",
    ) as file:
        scaler = pickle.load(
            file
        )

    with open(
        pca_path,
        "rb",
    ) as file:
        pca = pickle.load(
            file
        )

    return scaler, pca


# ============================================================
# EXPLAINED VARIANCE PLOT
# ============================================================

def _plot_explained_variance(
    pca,
    n_components,
):
    """
    Plot cumulative explained variance.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    variance = np.asarray(
        pca.explained_variance_ratio_,
        dtype=np.float64,
    )

    cumulative = np.cumsum(
        variance
    )

    plt.figure(
        figsize=(8, 5)
    )

    components = np.arange(
        1,
        len(variance) + 1,
    )

    plt.plot(
        components,
        cumulative,
        marker="o",
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Cumulative Explained Variance"
    )

    plt.title(
        f"PCA Explained Variance "
        f"({n_components} components)"
    )

    plt.ylim(
        0.0,
        min(
            1.0,
            max(
                1.0,
                float(cumulative[-1]) + 0.05,
            ),
        ),
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"pca_variance_{n_components}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "PCA variance plot saved: %s",
        output_path,
    )

    return output_path


# ============================================================
# COMPONENT SUMMARY
# ============================================================

def _print_pca_summary(
    pca,
):
    """
    Print useful PCA statistics.
    """

    variance = (
        pca.explained_variance_ratio_
    )

    cumulative = np.cumsum(
        variance
    )

    print()
    print("=" * 60)
    print("PCA SUMMARY")
    print("=" * 60)

    print(
        f"Components              : "
        f"{pca.n_components_}"
    )

    print(
        f"Original features       : "
        f"{pca.n_features_in_}"
    )

    print(
        f"Total explained variance: "
        f"{np.sum(variance):.4f}"
    )

    print(
        f"Explained variance (%)  : "
        f"{np.sum(variance) * 100:.2f}%"
    )

    print(
        f"First component         : "
        f"{variance[0]:.4f}"
    )

    print(
        f"Last component          : "
        f"{variance[-1]:.4f}"
    )

    print()

    for index, value in enumerate(
        cumulative,
        start=1,
    ):
        print(
            f"PC{index:<2} cumulative variance: "
            f"{value:.4f}"
        )

    print("=" * 60)
    print()


# ============================================================
# COMPLETE PCA STAGE
# ============================================================

def run_pca_stage(
    train_df,
    test_df,
    n_components=N_PCA_COMPONENTS,
):
    """
    Complete PCA pipeline.

    Steps:

        1. Validate train/test features
        2. Fit scaler on train
        3. Fit PCA on train
        4. Transform train
        5. Transform test
        6. Extract labels
        7. Save scaler/PCA
        8. Save explained variance plot

    Returns
    -------
    X_train
    y_train
    X_test
    y_test
    scaler
    pca
    """

    # --------------------------------------------------------
    # Validate feature consistency
    # --------------------------------------------------------

    _validate_train_test_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    scaler, pca = fit_scaler_pca(
        train_df,
        n_components=n_components,
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_train = transform(
        train_df,
        scaler,
        pca,
    )

    X_test = transform(
        test_df,
        scaler,
        pca,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = get_labels(
        train_df
    )

    y_test = get_labels(
        test_df
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training feature/label count mismatch."
        )

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "Test feature/label count mismatch."
        )

    if X_train.shape[1] != int(n_components):
        raise ValueError(
            "Unexpected PCA training dimension."
        )

    if X_test.shape[1] != int(n_components):
        raise ValueError(
            "Unexpected PCA test dimension."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_scaler_pca(
        scaler,
        pca,
        n_components,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    _plot_explained_variance(
        pca,
        n_components,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    _print_pca_summary(
        pca
    )

    print(
        f"[PCA] Components : "
        f"{pca.n_components_}"
    )

    print(
        f"[PCA] Variance   : "
        f"{np.sum(pca.explained_variance_ratio_):.4f}"
    )

    print(
        f"[PCA] X_train    : "
        f"{X_train.shape}"
    )

    print(
        f"[PCA] X_test     : "
        f"{X_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        pca,
    )
