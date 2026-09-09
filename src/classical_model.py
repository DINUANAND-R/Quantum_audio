"""
classical_model.py

Classical SVM models for speaker-independent
speech emotion recognition.

Supported experiments
---------------------

1. Original acoustic features
   1025 features -> StandardScaler -> SVM

2. PCA-4
   1025 features -> StandardScaler -> PCA(4) -> SVM

3. PCA-6
   1025 features -> StandardScaler -> PCA(6) -> SVM

4. PCA-8
   1025 features -> StandardScaler -> PCA(8) -> SVM

Important
---------

The dataset is speaker-independent.

Training speakers:
    Actors 1-20

Testing speakers:
    Actors 21-24

The test speakers are NEVER used during
hyperparameter tuning.

Cross-validation scaling is performed inside
the sklearn Pipeline to prevent data leakage.

Existing main.py compatibility
-------------------------------

The following continues to work:

    python main.py --stage classical

Existing public functions are preserved:

    train_classical_svm()
    tune_svm()
    evaluate_classical_svm()
    save_classical_svm()
    load_classical_svm()
    run_classical_stage()

Additional PCA functions are provided:

    train_pca_svm()
    tune_pca_svm()
    evaluate_pca_svm()
    run_pca_svm_experiment()
    run_all_pca_experiments()
"""


from __future__ import annotations


# ============================================================
# IMPORTS
# ============================================================

import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


# ============================================================
# CONFIG
# ============================================================

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_C = SVM_C

DEFAULT_KERNEL = SVM_KERNEL

DEFAULT_GAMMA = SVM_GAMMA

DEFAULT_PCA_COMPONENTS = [4, 6, 8]


# ============================================================
# MODEL PATHS
# ============================================================

SVM_MODEL_PATH = (
    MODELS_DIR / "classical_svm.pkl"
)

SVM_SCALER_PATH = (
    MODELS_DIR / "classical_svm_scaler.pkl"
)

SVM_CONFIG_PATH = (
    MODELS_DIR / "classical_svm_config.json"
)


# ============================================================
# HELPER:
# GET FEATURE COLUMNS
# ============================================================

def _get_feature_cols(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return all acoustic feature columns.

    Feature columns are expected to start
    with 'feat_'.
    """

    feature_columns = [
        str(column)
        for column in df.columns
        if str(column).startswith("feat_")
    ]

    if not feature_columns:

        raise ValueError(
            "No acoustic feature columns found. "
            "Expected columns beginning with 'feat_'."
        )

    return feature_columns


# ============================================================
# HELPER:
# VALIDATE TRAIN / TEST FEATURES
# ============================================================

def _validate_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[str]:
    """
    Make sure train and test have identical
    feature columns.
    """

    train_features = _get_feature_cols(
        train_df
    )

    test_features = _get_feature_cols(
        test_df
    )

    if train_features != test_features:

        missing_from_test = [
            column
            for column in train_features
            if column not in test_features
        ]

        extra_in_test = [
            column
            for column in test_features
            if column not in train_features
        ]

        raise ValueError(
            "Train/test feature mismatch.\n"
            f"Missing from test: {missing_from_test}\n"
            f"Extra in test: {extra_in_test}"
        )

    return train_features


# ============================================================
# HELPER:
# EXTRACT LABELS
# ============================================================

def _extract_labels(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Extract integer emotion labels.

    Supported columns:

        label
        emotion_label
        emotion
    """

    # --------------------------------------------------------
    # label
    # --------------------------------------------------------

    if "label" in df.columns:

        return df[
            "label"
        ].to_numpy(
            dtype=np.int64
        )

    # --------------------------------------------------------
    # emotion_label
    # --------------------------------------------------------

    if "emotion_label" in df.columns:

        return df[
            "emotion_label"
        ].to_numpy(
            dtype=np.int64
        )

    # --------------------------------------------------------
    # emotion
    # --------------------------------------------------------

    if "emotion" in df.columns:

        from src.config import (
            EMOTION_LABEL_MAP,
        )

        labels = df[
            "emotion"
        ].map(
            EMOTION_LABEL_MAP
        )

        if labels.isna().any():

            unknown = (
                df.loc[
                    labels.isna(),
                    "emotion",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Unknown emotion labels: "
                f"{unknown}"
            )

        return labels.to_numpy(
            dtype=np.int64
        )

    raise KeyError(
        "Could not find emotion labels. "
        "Expected 'label', "
        "'emotion_label', or 'emotion'."
    )


# ============================================================
# HELPER:
# CLEAN FEATURES
# ============================================================

def _clean_features(
    X: np.ndarray,
) -> np.ndarray:
    """
    Replace NaN and infinite values.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X


# ============================================================
# HELPER:
# LOAD X / Y
# ============================================================

def _prepare_xy(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dataframe into feature matrix
    and label vector.
    """

    feature_columns = _get_feature_cols(
        df
    )

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    y = _extract_labels(
        df
    )

    X = _clean_features(
        X
    )

    if len(X) != len(y):

        raise ValueError(
            "Feature and label count mismatch."
        )

    if len(X) == 0:

        raise ValueError(
            "Dataset is empty."
        )

    return X, y


# ============================================================
# HELPER:
# BUILD SVM
# ============================================================

def _build_svm(
    C: float,
    kernel: str,
    gamma: str | float,
    random_state: int = RANDOM_STATE,
) -> SVC:
    """
    Construct SVC.
    """

    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        decision_function_shape="ovr",
        random_state=random_state,
    )


# ============================================================
# HELPER:
# PARAMETER GRID
# ============================================================

def _get_svm_param_grid() -> list[dict]:
    """
    Hyperparameter grid for SVM.

    RBF:
        C
        gamma

    Linear:
        C
    """

    return [
        {
            "svm__C": [
                0.1,
                1.0,
                10.0,
                50.0,
                100.0,
            ],
            "svm__kernel": [
                "rbf",
            ],
            "svm__gamma": [
                "scale",
                0.001,
                0.01,
                0.05,
                0.1,
            ],
        },
        {
            "svm__C": [
                0.1,
                1.0,
                10.0,
                50.0,
                100.0,
            ],
            "svm__kernel": [
                "linear",
            ],
            "svm__gamma": [
                "scale",
            ],
        },
    ]


# ============================================================
# ORIGINAL SVM TRAINING
# ============================================================

def train_classical_svm(
    train_df: pd.DataFrame,
    C: float = DEFAULT_C,
    kernel: str = DEFAULT_KERNEL,
    gamma: str | float = DEFAULT_GAMMA,
    random_state: int = RANDOM_STATE,
) -> tuple[SVC, StandardScaler]:
    """
    Train SVM using all original acoustic features.

    Pipeline:

        1025 features
             ↓
        StandardScaler
             ↓
        RBF SVM
    """

    X_train, y_train = _prepare_xy(
        train_df
    )

    print()

    print(
        "[classical_svm] "
        f"Training on {len(X_train)} samples"
    )

    print(
        "[classical_svm] "
        f"Features: {X_train.shape[1]}"
    )

    print(
        "[classical_svm] "
        f"C={C}, "
        f"kernel={kernel}, "
        f"gamma={gamma}"
    )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm = _build_svm(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
    )

    start = time.time()

    svm.fit(
        X_scaled,
        y_train,
    )

    elapsed = (
        time.time() - start
    )

    print(
        "[classical_svm] "
        f"Training complete in "
        f"{elapsed:.2f}s"
    )

    return svm, scaler


# ============================================================
# ORIGINAL SVM TUNING
# ============================================================

def tune_svm(
    train_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Tune SVM using original acoustic features.

    Scaling is performed INSIDE the CV pipeline.

    This prevents scaling leakage.
    """

    X, y = _prepare_xy(
        train_df
    )

    print()

    print("=" * 60)

    print(
        "  SVM HYPERPARAMETER SEARCH"
    )

    print("=" * 60)

    print(
        f"  Training samples : {len(X)}"
    )

    print(
        f"  Features         : {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    probability=True,
                    decision_function_shape="ovr",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_get_svm_param_grid(),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    start = time.time()

    search.fit(
        X,
        y,
    )

    elapsed = (
        time.time() - start
    )

    best_params = {
        "C": search.best_params_[
            "svm__C"
        ],
        "kernel": search.best_params_[
            "svm__kernel"
        ],
        "gamma": search.best_params_[
            "svm__gamma"
        ],
    }

    print()

    print(
        "[classical_svm] "
        f"Search completed in "
        f"{elapsed:.2f}s"
    )

    print(
        "[classical_svm] "
        f"Best CV Macro F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        "[classical_svm] "
        f"Best parameters: "
        f"{best_params}"
    )

    print("=" * 60)

    return {
        "best_params": best_params,
        "best_cv_macro_f1": float(
            search.best_score_
        ),
        "search_time_seconds": float(
            elapsed
        ),
    }


# ============================================================
# ORIGINAL SVM EVALUATION
# ============================================================

def evaluate_classical_svm(
    svm: SVC,
    scaler: StandardScaler,
    test_df: pd.DataFrame,
    save: bool = True,
    train_size: int = 0,
    training_config: dict | None = None,
) -> dict:
    """
    Evaluate original-feature SVM on the complete
    held-out test set.
    """

    X_test, y_test = _prepare_xy(
        test_df
    )

    # --------------------------------------------------------
    # Scale using TRAINING scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start = time.time()

    y_pred = svm.predict(
        X_scaled
    )

    y_proba = svm.predict_proba(
        X_scaled
    )

    elapsed = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    labels = np.sort(
        np.unique(
            np.concatenate(
                [
                    y_test,
                    y_pred,
                ]
            )
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "model": "Classical SVM",

        "experiment": "original_1025",

        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "n_train": int(
            train_size
        ),

        "n_test": int(
            len(X_test)
        ),

        "n_features": int(
            X_test.shape[1]
        ),

        "svm_C": float(
            svm.C
        ),

        "svm_kernel": str(
            svm.kernel
        ),

        "svm_gamma": str(
            svm.gamma
        ),

        "inference_time_seconds": float(
            elapsed
        ),

        "classification_report": report_dict,
    }

    if training_config is not None:

        metrics[
            "training_config"
        ] = training_config

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "  CLASSICAL SVM RESULTS"
    )

    print("=" * 60)

    print(
        f"  Accuracy     : "
        f"{accuracy:.4f}"
    )

    print(
        f"  Macro F1     : "
        f"{macro_f1:.4f}"
    )

    print(
        f"  Weighted F1  : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"  Macro Prec   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"  Macro Recall : "
        f"{macro_recall:.4f}"
    )

    print(
        f"  Train samples: "
        f"{train_size}"
    )

    print(
        f"  Test samples : "
        f"{len(X_test)}"
    )

    print(
        f"  Features     : "
        f"{X_test.shape[1]}"
    )

    print()

    print(
        "  Classification Report:"
    )

    print(
        report_text
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        _save_experiment_results(
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=svm.classes_,
            cm=cm,
            experiment_name="classical_svm",
        )

    return metrics


# ============================================================
# PCA + SVM:
# TUNE
# ============================================================

def tune_pca_svm(
    train_df: pd.DataFrame,
    n_components: int,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Tune PCA + SVM.

    IMPORTANT
    ---------

    PCA and StandardScaler are both INSIDE
    the cross-validation pipeline.

    Therefore each CV fold independently fits:

        StandardScaler
              ↓
           PCA
              ↓
            SVM

    This is the cleanest way to compare PCA
    dimensions.
    """

    X, y = _prepare_xy(
        train_df
    )

    if n_components <= 0:

        raise ValueError(
            "n_components must be > 0."
        )

    if n_components >= X.shape[1]:

        raise ValueError(
            f"PCA components ({n_components}) "
            f"must be smaller than the original "
            f"feature count ({X.shape[1]})."
        )

    print()

    print("=" * 60)

    print(
        f"  PCA-{n_components} + SVM SEARCH"
    )

    print("=" * 60)

    print(
        f"  Training samples : {len(X)}"
    )

    print(
        f"  Original features: {X.shape[1]}"
    )

    print(
        f"  PCA components   : {n_components}"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),

            (
                "svm",
                SVC(
                    probability=True,
                    decision_function_shape="ovr",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_get_svm_param_grid(),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    start = time.time()

    search.fit(
        X,
        y,
    )

    elapsed = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    best_params = {
        "C": search.best_params_[
            "svm__C"
        ],
        "kernel": search.best_params_[
            "svm__kernel"
        ],
        "gamma": search.best_params_[
            "svm__gamma"
        ],
    }

    # --------------------------------------------------------
    # Best CV PCA
    # --------------------------------------------------------

    best_pipeline = (
        search.best_estimator_
    )

    fitted_scaler = (
        best_pipeline[
            "scaler"
        ]
    )

    fitted_pca = (
        best_pipeline[
            "pca"
        ]
    )

    explained_variance = float(
        np.sum(
            fitted_pca.explained_variance_ratio_
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        f"[PCA-{n_components}] "
        f"Search completed in "
        f"{elapsed:.2f}s"
    )

    print(
        f"[PCA-{n_components}] "
        f"Best CV Macro F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"[PCA-{n_components}] "
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    print(
        f"[PCA-{n_components}] "
        f"Best parameters: "
        f"{best_params}"
    )

    print("=" * 60)

    return {
        "best_params": best_params,

        "best_cv_macro_f1": float(
            search.best_score_
        ),

        "search_time_seconds": float(
            elapsed
        ),

        "n_components": int(
            n_components
        ),

        "explained_variance": (
            explained_variance
        ),

        "pipeline": best_pipeline,

        "scaler": fitted_scaler,

        "pca": fitted_pca,
    }


# ============================================================
# PCA + SVM:
# TRAIN FINAL MODEL
# ============================================================

def train_pca_svm(
    train_df: pd.DataFrame,
    n_components: int,
    C: float = DEFAULT_C,
    kernel: str = DEFAULT_KERNEL,
    gamma: str | float = DEFAULT_GAMMA,
    random_state: int = RANDOM_STATE,
) -> tuple[SVC, StandardScaler, PCA]:
    """
    Train final PCA + SVM model.

    PCA and scaler are fitted ONLY on training data.

    Returns:

        svm
        scaler
        pca
    """

    X_train, y_train = _prepare_xy(
        train_df
    )

    if n_components <= 0:

        raise ValueError(
            "n_components must be > 0."
        )

    if n_components >= X_train.shape[1]:

        raise ValueError(
            f"PCA components ({n_components}) "
            f"must be smaller than the original "
            f"feature count ({X_train.shape[1]})."
        )

    print()

    print(
        f"[PCA-{n_components}] "
        f"Training final model"
    )

    print(
        f"[PCA-{n_components}] "
        f"Original features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"[PCA-{n_components}] "
        f"Components: "
        f"{n_components}"
    )

    print(
        f"[PCA-{n_components}] "
        f"C={C}, "
        f"kernel={kernel}, "
        f"gamma={gamma}"
    )

    # --------------------------------------------------------
    # StandardScaler
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
        random_state=random_state,
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    explained_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    print(
        f"[PCA-{n_components}] "
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    print(
        f"[PCA-{n_components}] "
        f"Transformed shape: "
        f"{X_pca.shape}"
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm = _build_svm(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
    )

    start = time.time()

    svm.fit(
        X_pca,
        y_train,
    )

    elapsed = (
        time.time() - start
    )

    print(
        f"[PCA-{n_components}] "
        f"SVM training completed in "
        f"{elapsed:.2f}s"
    )

    return (
        svm,
        scaler,
        pca,
    )


# ============================================================
# PCA + SVM:
# EVALUATE
# ============================================================

def evaluate_pca_svm(
    svm: SVC,
    scaler: StandardScaler,
    pca: PCA,
    test_df: pd.DataFrame,
    n_components: int,
    save: bool = True,
    train_size: int = 0,
    training_config: dict | None = None,
) -> dict:
    """
    Evaluate PCA + SVM on ALL held-out test samples.
    """

    X_test, y_test = _prepare_xy(
        test_df
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_pca = pca.transform(
        X_scaled
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start = time.time()

    y_pred = svm.predict(
        X_pca
    )

    y_proba = svm.predict_proba(
        X_pca
    )

    elapsed = (
        time.time() - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = np.sort(
        np.unique(
            np.concatenate(
                [
                    y_test,
                    y_pred,
                ]
            )
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Explained variance
    # --------------------------------------------------------

    explained_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "model": "PCA + Classical SVM",

        "experiment": (
            f"pca_{n_components}_svm"
        ),

        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "n_train": int(
            train_size
        ),

        "n_test": int(
            len(X_test)
        ),

        "original_features": int(
            X_test.shape[1]
        ),

        "pca_components": int(
            n_components
        ),

        "explained_variance": (
            explained_variance
        ),

        "svm_C": float(
            svm.C
        ),

        "svm_kernel": str(
            svm.kernel
        ),

        "svm_gamma": str(
            svm.gamma
        ),

        "inference_time_seconds": float(
            elapsed
        ),

        "classification_report": report_dict,
    }

    if training_config is not None:

        metrics[
            "training_config"
        ] = training_config

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        f"  PCA-{n_components} + SVM RESULTS"
    )

    print("=" * 60)

    print(
        f"  Accuracy     : "
        f"{accuracy:.4f}"
    )

    print(
        f"  Macro F1     : "
        f"{macro_f1:.4f}"
    )

    print(
        f"  Weighted F1  : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"  Macro Prec   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"  Macro Recall : "
        f"{macro_recall:.4f}"
    )

    print(
        f"  PCA variance : "
        f"{explained_variance:.4f}"
    )

    print(
        f"  Train samples: "
        f"{train_size}"
    )

    print(
        f"  Test samples : "
        f"{len(X_test)}"
    )

    print(
        f"  PCA features : "
        f"{n_components}"
    )

    print()

    print(
        "  Classification Report:"
    )

    print(
        report_text
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        _save_experiment_results(
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=svm.classes_,
            cm=cm,
            experiment_name=(
                f"pca_{n_components}_svm"
            ),
        )

    return metrics


# ============================================================
# PCA + SVM:
# COMPLETE EXPERIMENT
# ============================================================

def run_pca_svm_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int,
    tune: bool = True,
    save_model: bool = True,
) -> dict:
    """
    Run one complete PCA + SVM experiment.

    Example:

        run_pca_svm_experiment(
            train_df,
            test_df,
            n_components=8
        )
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    _validate_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Tune
    # --------------------------------------------------------

    if tune:

        tuning = tune_pca_svm(
            train_df=train_df,
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

        best_params = (
            tuning[
                "best_params"
            ]
        )

        C = best_params[
            "C"
        ]

        kernel = best_params[
            "kernel"
        ]

        gamma = best_params[
            "gamma"
        ]

    else:

        tuning = None

        C = DEFAULT_C

        kernel = DEFAULT_KERNEL

        gamma = DEFAULT_GAMMA

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------

    svm, scaler, pca = train_pca_svm(
        train_df=train_df,
        n_components=n_components,
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    training_config = {
        "experiment": (
            f"pca_{n_components}_svm"
        ),

        "pca_components": int(
            n_components
        ),

        "tuning": bool(
            tune
        ),

        "cv_folds": 5,

        "cv_strategy": (
            "StratifiedKFold"
        ),

        "cv_shuffle": True,

        "cv_random_state": (
            RANDOM_STATE
        ),

        "scaling_inside_cv": True,

        "pca_inside_cv": True,

        "scoring": "f1_macro",

        "C": float(
            C
        ),

        "kernel": str(
            kernel
        ),

        "gamma": str(
            gamma
        ),
    }

    if tuning is not None:

        training_config[
            "best_cv_macro_f1"
        ] = tuning[
            "best_cv_macro_f1"
        ]

        training_config[
            "search_time_seconds"
        ] = tuning[
            "search_time_seconds"
        ]

        training_config[
            "explained_variance_cv"
        ] = tuning[
            "explained_variance"
        ]

    # --------------------------------------------------------
    # Save PCA model
    # --------------------------------------------------------

    if save_model:

        _save_pca_svm_model(
            svm=svm,
            scaler=scaler,
            pca=pca,
            n_components=n_components,
            config=training_config,
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_pca_svm(
        svm=svm,
        scaler=scaler,
        pca=pca,
        test_df=test_df,
        n_components=n_components,
        save=True,
        train_size=len(train_df),
        training_config=training_config,
    )

    return metrics


# ============================================================
# RUN ALL PCA EXPERIMENTS
# ============================================================

def run_all_pca_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    components_list: list[int] | None = None,
    tune: bool = True,
) -> pd.DataFrame:
    """
    Run PCA-4, PCA-6 and PCA-8 experiments.

    Returns a comparison dataframe.
    """

    if components_list is None:

        components_list = (
            DEFAULT_PCA_COMPONENTS.copy()
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    _validate_features(
        train_df,
        test_df,
    )

    results = []

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    for n_components in components_list:

        print()

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"# STARTING PCA-{n_components} + SVM"
        )

        print(
            "#" * 70
        )

        metrics = run_pca_svm_experiment(
            train_df=train_df,
            test_df=test_df,
            n_components=n_components,
            tune=tune,
            save_model=True,
        )

        results.append(
            {
                "experiment": (
                    f"PCA-{n_components} + SVM"
                ),

                "pca_components": (
                    n_components
                ),

                "accuracy": (
                    metrics["accuracy"]
                ),

                "macro_f1": (
                    metrics["macro_f1"]
                ),

                "weighted_f1": (
                    metrics["weighted_f1"]
                ),

                "macro_precision": (
                    metrics[
                        "macro_precision"
                    ]
                ),

                "macro_recall": (
                    metrics[
                        "macro_recall"
                    ]
                ),

                "explained_variance": (
                    metrics[
                        "explained_variance"
                    ]
                ),

                "best_cv_macro_f1": (
                    metrics[
                        "training_config"
                    ].get(
                        "best_cv_macro_f1",
                        np.nan,
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # Dataframe
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by test Macro F1
    # --------------------------------------------------------

    comparison = comparison.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        METRICS_DIR
        / "pca_svm_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "  PCA + SVM COMPARISON"
    )

    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print("=" * 80)

    print()

    best = comparison.iloc[0]

    print(
        "Best PCA experiment:"
    )

    print(
        f"  {best['experiment']}"
    )

    print(
        f"  Accuracy : "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"  Macro F1 : "
        f"{best['macro_f1']:.4f}"
    )

    print()

    print(
        f"Comparison saved → "
        f"{comparison_path}"
    )

    return comparison


# ============================================================
# SAVE PCA + SVM MODEL
# ============================================================

def _save_pca_svm_model(
    svm: SVC,
    scaler: StandardScaler,
    pca: PCA,
    n_components: int,
    config: dict,
) -> None:
    """
    Save PCA, scaler and SVM.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / f"pca_{n_components}_svm.pkl"
    )

    scaler_path = (
        MODELS_DIR
        / f"pca_{n_components}_scaler.pkl"
    )

    pca_path = (
        MODELS_DIR
        / f"pca_{n_components}.pkl"
    )

    config_path = (
        MODELS_DIR
        / f"pca_{n_components}_svm_config.json"
    )

    # --------------------------------------------------------
    # Save SVM
    # --------------------------------------------------------

    with open(
        model_path,
        "wb",
    ) as file:

        pickle.dump(
            svm,
            file,
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
    # Save config
    # --------------------------------------------------------

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    print(
        f"[PCA-{n_components}] "
        f"Model saved → {model_path}"
    )

    print(
        f"[PCA-{n_components}] "
        f"Scaler saved → {scaler_path}"
    )

    print(
        f"[PCA-{n_components}] "
        f"PCA saved → {pca_path}"
    )


# ============================================================
# SAVE ORIGINAL SVM MODEL
# ============================================================

def save_classical_svm(
    svm: SVC,
    scaler: StandardScaler,
    config: dict | None = None,
) -> None:
    """
    Save original-feature SVM and scaler.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    with open(
        SVM_MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            svm,
            file,
        )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    with open(
        SVM_SCALER_PATH,
        "wb",
    ) as file:

        pickle.dump(
            scaler,
            file,
        )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    if config is not None:

        with open(
            SVM_CONFIG_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=2,
            )

    print(
        "[classical_svm] "
        f"Model saved → "
        f"{SVM_MODEL_PATH}"
    )


# ============================================================
# LOAD ORIGINAL SVM
# ============================================================

def load_classical_svm() -> tuple[
    SVC,
    StandardScaler,
]:
    """
    Load original-feature SVM.
    """

    if not SVM_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{SVM_MODEL_PATH}"
        )

    if not SVM_SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Scaler not found: "
            f"{SVM_SCALER_PATH}"
        )

    with open(
        SVM_MODEL_PATH,
        "rb",
    ) as file:

        svm = pickle.load(
            file
        )

    with open(
        SVM_SCALER_PATH,
        "rb",
    ) as file:

        scaler = pickle.load(
            file
        )

    return (
        svm,
        scaler,
    )


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

def _save_experiment_results(
    metrics: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    cm: np.ndarray,
    experiment_name: str,
) -> None:
    """
    Save metrics, predictions and confusion matrix.
    """

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_path = (
        METRICS_DIR
        / f"{experiment_name}_results.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "[metrics] "
        f"Saved → {metrics_path}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_data = {
        "true_label": y_test,
        "predicted": y_pred,
        "correct": (
            y_test == y_pred
        ),
    }

    prediction_df = pd.DataFrame(
        prediction_data
    )

    # --------------------------------------------------------
    # Probability columns
    # --------------------------------------------------------

    for index, cls in enumerate(
        classes
    ):

        prediction_df[
            f"prob_{cls}"
        ] = y_proba[
            :,
            index,
        ]

    prediction_path = (
        PREDICTIONS_DIR
        / f"{experiment_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        "[predictions] "
        f"Saved → {prediction_path}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        FIGURES_DIR
        / f"{experiment_name}_confusion_matrix.png"
    )

    _plot_confusion_matrix(
        cm=cm,
        labels=classes,
        title=(
            f"{experiment_name} "
            "Confusion Matrix"
        ),
        save_path=confusion_path,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray | list,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """
    Plot confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    plt.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(
            len(labels)
        ),
        yticks=np.arange(
            len(labels)
        ),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        cm.max() / 2.0
        if cm.size and cm.max() > 0
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=9,
            )

    fig.tight_layout()

    if save_path is not None:

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        print(
            "[plot] "
            f"Saved → {save_path}"
        )

    plt.close(
        fig
    )


# ============================================================
# EXISTING MAIN.PY STAGE
# ============================================================

def run_classical_stage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tune: bool = True,
) -> dict:
    """
    Existing classical stage used by main.py.

    IMPORTANT:

    This function intentionally keeps the original
    1025-feature SVM experiment.

    Therefore:

        python main.py --stage classical

    continues to produce the original baseline.

    PCA experiments are separate and do not alter
    the existing baseline.
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    _validate_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Tune
    # --------------------------------------------------------

    if tune:

        tuning = tune_svm(
            train_df=train_df,
            random_state=RANDOM_STATE,
        )

        best_params = (
            tuning[
                "best_params"
            ]
        )

        C = best_params[
            "C"
        ]

        kernel = best_params[
            "kernel"
        ]

        gamma = best_params[
            "gamma"
        ]

    else:

        tuning = None

        C = DEFAULT_C

        kernel = DEFAULT_KERNEL

        gamma = DEFAULT_GAMMA

    # --------------------------------------------------------
    # Train final
    # --------------------------------------------------------

    svm, scaler = train_classical_svm(
        train_df=train_df,
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    training_config = {
        "experiment": (
            "original_1025_svm"
        ),

        "tuning": bool(
            tune
        ),

        "cv_folds": 5,

        "cv_strategy": (
            "StratifiedKFold"
        ),

        "cv_shuffle": True,

        "cv_random_state": (
            RANDOM_STATE
        ),

        "scaling_inside_cv": True,

        "scoring": "f1_macro",

        "C": float(
            C
        ),

        "kernel": str(
            kernel
        ),

        "gamma": str(
            gamma
        ),
    }

    if tuning is not None:

        training_config[
            "best_cv_macro_f1"
        ] = tuning[
            "best_cv_macro_f1"
        ]

        training_config[
            "search_time_seconds"
        ] = tuning[
            "search_time_seconds"
        ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_classical_svm(
        svm=svm,
        scaler=scaler,
        config=training_config,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_classical_svm(
        svm=svm,
        scaler=scaler,
        test_df=test_df,
        save=True,
        train_size=len(train_df),
        training_config=training_config,
    )

    return metrics


# ============================================================
# OPTIONAL COMMAND-LINE PCA EXPERIMENT
# ============================================================

def _run_pca_experiments_from_csv() -> None:
    """
    Optional helper.

    This function allows PCA experiments to be run
    directly from this module.

    It does NOT affect main.py.

    It expects:

        data/processed/train.csv
        data/processed/test.csv
    """

    from src.config import (
        TEST_CSV,
        TRAIN_CSV,
    )

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found: "
            f"{TRAIN_CSV}"
        )

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found: "
            f"{TEST_CSV}"
        )

    print()

    print(
        "Loading training data..."
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    print(
        "Loading test data..."
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    run_all_pca_experiments(
        train_df=train_df,
        test_df=test_df,
        components_list=[
            4,
            6,
            8,
        ],
        tune=True,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    _run_pca_experiments_from_csv()