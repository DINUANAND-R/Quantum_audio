"""
qsvc_model.py — Quantum Support Vector Classifier (QSVC)

Pipeline:

    PCA features
          ↓
    ZZFeatureMap
          ↓
    Fidelity Quantum Kernel
          ↓
    Classical SVM
          ↓
    Emotion prediction

The quantum part computes the kernel matrix.
The final SVM optimisation is classical.

Compatible with:
    Qiskit >= 2.0
    qiskit-machine-learning >= 0.8
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    FEATURE_MAP_REPS,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_QUBITS,
    PREDICTIONS_DIR,
    QSVC_SHOTS,
    RANDOM_STATE,
)

from src.quantum_features import build_feature_map


logger = logging.getLogger(__name__)


# ============================================================
# MODEL PATHS
# ============================================================

_QSVC_PATH = MODELS_DIR / "qsvc.pkl"

_QSVC_CONFIG_PATH = (
    MODELS_DIR / "qsvc_config.json"
)


# ============================================================
# VALIDATION
# ============================================================

def _validate_data(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> None:
    """
    Validate X and y before quantum kernel computation.
    """

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(
            f"{name}: X must be a 2D array. "
            f"Received shape={X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X and y have different lengths. "
            f"X={len(X)}, y={len(y)}"
        )

    if len(X) == 0:
        raise ValueError(
            f"{name}: dataset is empty."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"{name}: X contains NaN or infinite values."
        )


def _validate_qubits(
    X: np.ndarray,
    n_qubits: int,
) -> None:
    """
    Ensure PCA dimensionality matches quantum circuit size.
    """

    if n_qubits <= 0:
        raise ValueError(
            "n_qubits must be greater than zero."
        )

    if X.shape[1] != n_qubits:
        raise ValueError(
            "\nQSVC dimension mismatch.\n"
            f"Expected {n_qubits} features "
            f"for {n_qubits} qubits.\n"
            f"Received {X.shape[1]} features.\n\n"
            "Make sure:\n"
            "PCA components == N_QUBITS"
        )


# ============================================================
# BUILD QSVC
# ============================================================

def build_qsvc(
    n_qubits: int = N_QUBITS,
    feature_map_reps: int = FEATURE_MAP_REPS,
):
    """
    Construct a QSVC using FidelityQuantumKernel.

    Important for Qiskit 2.x:
        Do NOT pass a sampler to FidelityQuantumKernel.

    The kernel is constructed as:

        K(x, y) = |<psi(x)|psi(y)>|^2
    """

    if n_qubits <= 0:
        raise ValueError(
            "n_qubits must be greater than zero."
        )

    if feature_map_reps <= 0:
        raise ValueError(
            "feature_map_reps must be greater than zero."
        )

    try:

        from qiskit_machine_learning.kernels import (
            FidelityQuantumKernel
        )

        from qiskit_machine_learning.algorithms import (
            QSVC as QiskitQSVC
        )

    except ImportError:

        try:

            from qiskit_machine_learning.kernels import (
                FidelityQuantumKernel
            )

            from qiskit_machine_learning.algorithms.classifiers import (
                QSVC as QiskitQSVC
            )

        except ImportError as exc:

            raise ImportError(
                "\nCould not import QSVC or "
                "FidelityQuantumKernel.\n\n"
                "Install using:\n"
                "python -m pip install -U "
                "qiskit-machine-learning "
                "qiskit-algorithms\n\n"
                f"Original error: {exc}"
            ) from exc

    # --------------------------------------------------------
    # Feature map
    # --------------------------------------------------------

    feature_map = build_feature_map(
        n_qubits=n_qubits,
        reps=feature_map_reps,
    )

    # --------------------------------------------------------
    # Quantum kernel
    # --------------------------------------------------------

    kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # --------------------------------------------------------
    # QSVC
    # --------------------------------------------------------

    qsvc = QiskitQSVC(
        quantum_kernel=kernel
    )

    print()
    print("=" * 60)
    print("QSVC CONFIGURATION")
    print("=" * 60)

    print(
        f"Qubits           : {n_qubits}"
    )

    print(
        f"Feature map reps : {feature_map_reps}"
    )

    print(
        f"Kernel           : FidelityQuantumKernel"
    )

    print(
        f"Shots            : {QSVC_SHOTS}"
    )

    print(
        f"Random state     : {RANDOM_STATE}"
    )

    print("=" * 60)

    return qsvc


# ============================================================
# TRAIN QSVC
# ============================================================

def train_qsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int = N_QUBITS,
    feature_map_reps: int = FEATURE_MAP_REPS,
) -> tuple:
    """
    Train the QSVC.

    Returns
    -------
    fitted_qsvc, training_time_seconds
    """

    X_train = np.asarray(
        X_train
    )

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    _validate_data(
        X_train,
        y_train,
        "QSVC training data",
    )

    _validate_qubits(
        X_train,
        n_qubits,
    )

    unique_classes = np.unique(
        y_train
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "QSVC requires at least two classes."
        )

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = np.arange(
        len(X_train)
    )

    rng.shuffle(
        indices
    )

    X_train = X_train[
        indices
    ]

    y_train = y_train[
        indices
    ]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    qsvc = build_qsvc(
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
    )

    print()
    print("=" * 60)
    print("QSVC TRAINING")
    print("=" * 60)

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Features         : {X_train.shape[1]}"
    )

    print(
        f"Classes          : {len(unique_classes)}"
    )

    print(
        f"Class labels     : {unique_classes}"
    )

    print()
    print(
        "[qsvc] Computing quantum kernel..."
    )

    print(
        "[qsvc] Kernel computation has "
        "approximately O(n²) pairwise cost."
    )

    print(
        "[qsvc] Training may take several minutes."
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start_time = time.time()

    qsvc.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time()
        - start_time
    )

    print()
    print(
        "[qsvc] Training complete."
    )

    print(
        f"[qsvc] Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"[qsvc] Training time: "
        f"{training_time / 60:.2f} minutes"
    )

    return (
        qsvc,
        training_time,
    )


# ============================================================
# EVALUATE QSVC
# ============================================================

def evaluate_qsvc(
    qsvc,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float = 0.0,
    n_qubits: int = N_QUBITS,
    feature_map_reps: int = FEATURE_MAP_REPS,
    save: bool = True,
) -> dict:
    """
    Evaluate QSVC on the complete supplied test set.
    """

    X_test = np.asarray(
        X_test
    )

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    _validate_data(
        X_test,
        y_test,
        "QSVC test data",
    )

    _validate_qubits(
        X_test,
        n_qubits,
    )

    print()
    print("=" * 60)
    print("QSVC EVALUATION")
    print("=" * 60)

    print(
        f"Test samples : {len(X_test)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_start = time.time()

    y_pred = qsvc.predict(
        X_test
    )

    prediction_time = (
        time.time()
        - prediction_start
    )

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

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
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = sorted(
        set(
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
        labels=classes,
    )

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {
        "model": "QSVC",

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

        "n_train": None,

        "n_test": int(
            len(y_test)
        ),

        "n_qubits": int(
            n_qubits
        ),

        "feature_map_reps": int(
            feature_map_reps
        ),

        "shots": QSVC_SHOTS,

        "training_time_s": float(
            training_time
        ),

        "prediction_time_s": float(
            prediction_time
        ),

        "random_state": int(
            RANDOM_STATE
        ),

        "classification_report":
            report_dict,
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("QSVC RESULTS")
    print("=" * 60)

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Macro F1     : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1  : {weighted_f1:.4f}"
    )

    print(
        f"Macro Prec   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall : {macro_recall:.4f}"
    )

    print(
        f"Train time   : "
        f"{training_time:.2f}s"
    )

    print(
        f"Predict time : "
        f"{prediction_time:.2f}s"
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        report_text
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        _save_qsvc_results(
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            classes=classes,
            cm=cm,
        )

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def _save_qsvc_results(
    metrics: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    cm: np.ndarray,
) -> None:
    """
    Save metrics, predictions and confusion matrix.
    """

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
        METRICS_DIR /
        "qsvc_results.json"
    )

    with open(
        metrics_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"[qsvc] Metrics → "
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted": y_pred,
            "correct": (
                y_test == y_pred
            ),
        }
    )

    prediction_path = (
        PREDICTIONS_DIR /
        "qsvc_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"[qsvc] Predictions → "
        f"{prediction_path}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        FIGURES_DIR /
        "qsvc_confusion_matrix.png"
    )

    _plot_confusion_matrix(
        cm=cm,
        labels=classes,
        title="QSVC - Confusion Matrix",
        save_path=confusion_path,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "QSVC - Confusion Matrix",
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
        if cm.size > 0
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=9,
            )

    fig.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        print(
            f"[plot] QSVC confusion matrix → "
            f"{save_path}"
        )

    plt.close(fig)


# ============================================================
# SAVE QSVC
# ============================================================

def save_qsvc(
    qsvc,
    n_qubits: int = N_QUBITS,
    feature_map_reps: int = FEATURE_MAP_REPS,
) -> None:
    """
    Save fitted QSVC and its configuration.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {
        "model": "QSVC",

        "n_qubits": int(
            n_qubits
        ),

        "feature_map_reps": int(
            feature_map_reps
        ),

        "shots": QSVC_SHOTS,

        "random_state": int(
            RANDOM_STATE
        ),
    }

    config_path = (
        MODELS_DIR /
        "qsvc_config.json"
    )

    with open(
        config_path,
        "w",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    print(
        f"[qsvc] Config → "
        f"{config_path}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    try:

        with open(
            _QSVC_PATH,
            "wb",
        ) as file:

            pickle.dump(
                qsvc,
                file,
            )

        print(
            f"[qsvc] Model → "
            f"{_QSVC_PATH}"
        )

    except Exception as exc:

        logger.exception(
            "Could not save QSVC model."
        )

        raise RuntimeError(
            f"Failed to save QSVC model: {exc}"
        ) from exc


# ============================================================
# LOAD QSVC
# ============================================================

def load_qsvc():
    """
    Load saved QSVC model.
    """

    if not _QSVC_PATH.exists():

        raise FileNotFoundError(
            f"\nQSVC model not found at:\n"
            f"{_QSVC_PATH}\n\n"
            "Train it first using:\n"
            "python main.py --stage qsvc"
        )

    with open(
        _QSVC_PATH,
        "rb",
    ) as file:

        qsvc = pickle.load(
            file
        )

    print(
        f"[qsvc] Loaded model → "
        f"{_QSVC_PATH}"
    )

    return qsvc


# ============================================================
# COMPLETE QSVC STAGE
# ============================================================

def run_qsvc_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = N_QUBITS,
    feature_map_reps: int = FEATURE_MAP_REPS,
) -> dict:
    """
    Complete QSVC pipeline:

        Validate
            ↓
        Train
            ↓
        Save
            ↓
        Evaluate
            ↓
        Save metrics
    """

    # --------------------------------------------------------
    # Convert input
    # --------------------------------------------------------

    X_train = np.asarray(
        X_train
    )

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    X_test = np.asarray(
        X_test
    )

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    _validate_data(
        X_train,
        y_train,
        "QSVC training data",
    )

    _validate_data(
        X_test,
        y_test,
        "QSVC test data",
    )

    _validate_qubits(
        X_train,
        n_qubits,
    )

    _validate_qubits(
        X_test,
        n_qubits,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    qsvc, training_time = train_qsvc(
        X_train=X_train,
        y_train=y_train,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_qsvc(
        qsvc=qsvc,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_qsvc(
        qsvc=qsvc,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
        save=True,
    )

    # --------------------------------------------------------
    # Add actual training size
    # --------------------------------------------------------

    metrics["n_train"] = int(
        len(X_train)
    )

    # --------------------------------------------------------
    # Re-save final metrics
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        METRICS_DIR /
        "qsvc_results.json"
    )

    with open(
        metrics_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"[qsvc] Final metrics → "
        f"{metrics_path}"
    )

    return metrics