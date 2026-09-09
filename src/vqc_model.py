"""
vqc_model.py — Variational Quantum Classifier (VQC)

Pipeline:

    PCA features
          ↓
    ZZFeatureMap
          ↓
    EfficientSU2
          ↓
    SPSA Optimizer
          ↓
    Multiclass VQC

Compatible with:
    Qiskit >= 2.0
    qiskit-machine-learning >= 0.8
    qiskit-algorithms
"""

from __future__ import annotations

import json
import logging
import pickle
import time

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
    ANSATZ_REPS,
    FEATURE_MAP_REPS,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_QUBITS,
    PREDICTIONS_DIR,
    VQC_MAX_ITER,
    VQC_SEED,
    VQC_SHOTS,
)

from src.quantum_features import (
    build_ansatz,
    build_feature_map,
)

logger = logging.getLogger(__name__)


# ============================================================
# MODEL PATHS
# ============================================================

_VQC_PATH = MODELS_DIR / "vqc_params.pkl"
_VQC_CONFIG_PATH = MODELS_DIR / "vqc_config.json"


# ============================================================
# VALIDATION
# ============================================================

def _validate_data(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> None:
    """
    Validate feature matrix and labels.
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
    The VQC expects the number of input features
    to equal the number of qubits.
    """

    if X.shape[1] != n_qubits:
        raise ValueError(
            "\nVQC dimension mismatch.\n"
            f"Expected {n_qubits} features "
            f"for {n_qubits} qubits, "
            f"but received {X.shape[1]} features.\n\n"
            "Make sure:\n"
            "PCA components == N_QUBITS"
        )


# ============================================================
# SAMPLER
# ============================================================

def _build_sampler(
    shots: int | None,
    random_state: int,
):
    """
    Build a Qiskit sampler.

    Qiskit 2.x:
        StatevectorSampler

    If shots is None:
        default StatevectorSampler behaviour.

    If shots is specified:
        finite-shot simulation.
    """

    try:

        from qiskit.primitives import StatevectorSampler

        if shots is None:

            sampler = StatevectorSampler(
                seed=random_state
            )

        else:

            sampler = StatevectorSampler(
                default_shots=shots,
                seed=random_state,
            )

        print(
            f"[vqc] Using StatevectorSampler "
            f"(shots={shots})"
        )

        return sampler

    except ImportError:
        pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    try:

        from qiskit.primitives import Sampler

        sampler = Sampler()

        print(
            "[vqc] Using legacy Qiskit Sampler"
        )

        return sampler

    except ImportError as exc:

        raise ImportError(
            "\nNo compatible Qiskit sampler found.\n\n"
            "Install/update using:\n"
            "python -m pip install -U qiskit "
            "qiskit-machine-learning "
            "qiskit-algorithms\n"
        ) from exc


# ============================================================
# BUILD VQC
# ============================================================

def build_vqc(
    n_qubits: int = N_QUBITS,
    max_iter: int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots: int | None = VQC_SHOTS,
):
    """
    Build an unfitted VQC model.
    """

    if n_qubits < 2:
        raise ValueError(
            "n_qubits must be >= 2."
        )

    if max_iter <= 0:
        raise ValueError(
            "max_iter must be > 0."
        )

    # --------------------------------------------------------
    # SPSA
    # --------------------------------------------------------

    try:

        from qiskit_algorithms.optimizers import SPSA

    except ImportError as exc:

        raise ImportError(
            "\nqiskit-algorithms is not installed.\n\n"
            "Install using:\n"
            "python -m pip install qiskit-algorithms"
        ) from exc

    # --------------------------------------------------------
    # Qiskit ML VQC
    # --------------------------------------------------------

    try:

        from qiskit_machine_learning.algorithms.classifiers import (
            VQC as QiskitVQC
        )

    except ImportError as exc:

        raise ImportError(
            "\nqiskit-machine-learning is not installed "
            "or the VQC API is unavailable.\n\n"
            "Install using:\n"
            "python -m pip install qiskit-machine-learning"
        ) from exc

    # --------------------------------------------------------
    # Feature map
    # --------------------------------------------------------

    feature_map = build_feature_map(
        n_qubits=n_qubits,
        reps=FEATURE_MAP_REPS,
    )

    # --------------------------------------------------------
    # Ansatz
    # --------------------------------------------------------

    ansatz = build_ansatz(
        n_qubits=n_qubits,
        reps=ANSATZ_REPS,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = SPSA(
        maxiter=max_iter
    )

    # --------------------------------------------------------
    # Sampler
    # --------------------------------------------------------

    sampler = _build_sampler(
        shots=shots,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    vqc = QiskitVQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VQC CONFIGURATION")
    print("=" * 60)

    print(
        f"Qubits              : {n_qubits}"
    )

    print(
        f"Feature map reps    : {FEATURE_MAP_REPS}"
    )

    print(
        f"Ansatz reps         : {ANSATZ_REPS}"
    )

    print(
        f"Trainable parameters: {ansatz.num_parameters}"
    )

    print(
        f"SPSA iterations     : {max_iter}"
    )

    print(
        f"Shots               : {shots}"
    )

    print(
        f"Random seed         : {random_state}"
    )

    print("=" * 60)

    return vqc


# ============================================================
# TRAIN VQC
# ============================================================

def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int = N_QUBITS,
    max_iter: int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots: int | None = VQC_SHOTS,
):
    """
    Train the VQC.

    Parameters
    ----------
    X_train:
        Shape = (samples, n_qubits)

    y_train:
        Class labels.
    """

    X_train = np.asarray(X_train)

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    _validate_data(
        X_train,
        y_train,
        "VQC training data",
    )

    _validate_qubits(
        X_train,
        n_qubits,
    )

    classes = np.unique(
        y_train
    )

    if len(classes) < 2:
        raise ValueError(
            "VQC requires at least two classes."
        )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_state
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
    # Print training information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VQC TRAINING")
    print("=" * 60)

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Features         : {X_train.shape[1]}"
    )

    print(
        f"Classes          : {len(classes)}"
    )

    print(
        f"Class labels     : {classes}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Build VQC
    # --------------------------------------------------------

    vqc = build_vqc(
        n_qubits=n_qubits,
        max_iter=max_iter,
        random_state=random_state,
        shots=shots,
    )

    print()
    print(
        "[vqc] Starting VQC optimization..."
    )

    print(
        f"[vqc] SPSA maximum iterations: {max_iter}"
    )

    print(
        "[vqc] Training may take several minutes."
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start_time = time.time()

    vqc.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time()
        - start_time
    )

    print()
    print(
        "[vqc] Training completed."
    )

    print(
        f"[vqc] Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"[vqc] Training time: "
        f"{training_time / 60:.2f} minutes"
    )

    return (
        vqc,
        training_time,
    )


# ============================================================
# EVALUATE VQC
# ============================================================

def evaluate_vqc(
    vqc,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float = 0.0,
    n_qubits: int = N_QUBITS,
    max_iter: int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots: int | None = VQC_SHOTS,
    save: bool = True,
):
    """
    Evaluate the VQC on the complete supplied test set.
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
        "VQC test data",
    )

    _validate_qubits(
        X_test,
        n_qubits,
    )

    print()
    print("=" * 60)
    print("VQC EVALUATION")
    print("=" * 60)

    print(
        f"Test samples : {len(X_test)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_start = time.time()

    y_pred = vqc.predict(
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
    # Confusion matrix
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

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "model": "VQC",

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
            FEATURE_MAP_REPS
        ),

        "ansatz_reps": int(
            ANSATZ_REPS
        ),

        "vqc_max_iter": int(
            max_iter
        ),

        "shots": shots,

        "training_time_s": float(
            training_time
        ),

        "prediction_time_s": float(
            prediction_time
        ),

        "random_state": int(
            random_state
        ),

        "classification_report":
            report_dict,
    }

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VQC RESULTS")
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
    print("Classification Report:")
    print(report_text)

    print("=" * 60)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        _save_vqc_results(
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

def _save_vqc_results(
    metrics,
    y_test,
    y_pred,
    classes,
    cm,
):
    """
    Save:
        metrics JSON
        predictions CSV
        confusion matrix
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
    # Metrics JSON
    # --------------------------------------------------------

    metrics_path = (
        METRICS_DIR /
        "vqc_results.json"
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
        f"[vqc] Metrics → "
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
        "vqc_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"[vqc] Predictions → "
        f"{prediction_path}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        FIGURES_DIR /
        "vqc_confusion_matrix.png"
    )

    _plot_confusion_matrix(
        cm=cm,
        labels=classes,
        title="VQC - Confusion Matrix",
        save_path=confusion_path,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def _plot_confusion_matrix(
    cm,
    labels,
    title,
    save_path=None,
):
    """
    Plot and save confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Purples,
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

    if cm.size > 0:

        threshold = (
            cm.max() / 2.0
        )

    else:

        threshold = 0

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
            f"[plot] VQC confusion matrix → "
            f"{save_path}"
        )

    plt.close(fig)


# ============================================================
# SAVE VQC MODEL
# ============================================================

def save_vqc(
    vqc,
    n_qubits: int = N_QUBITS,
    max_iter: int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots: int | None = VQC_SHOTS,
):
    """
    Save trained VQC and configuration.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {
        "model": "VQC",

        "n_qubits": int(
            n_qubits
        ),

        "feature_map_reps": int(
            FEATURE_MAP_REPS
        ),

        "ansatz_reps": int(
            ANSATZ_REPS
        ),

        "max_iter": int(
            max_iter
        ),

        "seed": int(
            random_state
        ),

        "shots": shots,
    }

    with open(
        _VQC_CONFIG_PATH,
        "w",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    print(
        f"[vqc] Config → "
        f"{_VQC_CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------

    try:

        with open(
            _VQC_PATH,
            "wb",
        ) as file:

            pickle.dump(
                vqc,
                file,
            )

        print(
            f"[vqc] Model → "
            f"{_VQC_PATH}"
        )

    except Exception as exc:

        logger.warning(
            "Could not pickle VQC: %s",
            exc,
        )

        print(
            "[vqc] Warning: complete VQC "
            "could not be pickled."
        )

        print(
            "[vqc] Configuration was saved."
        )


# ============================================================
# LOAD VQC
# ============================================================

def load_vqc():
    """
    Load saved VQC.
    """

    if not _VQC_PATH.exists():

        raise FileNotFoundError(
            f"\nVQC model not found:\n"
            f"{_VQC_PATH}\n\n"
            "Train the model first using:\n"
            "python main.py --stage vqc"
        )

    with open(
        _VQC_PATH,
        "rb",
    ) as file:

        vqc = pickle.load(
            file
        )

    print(
        f"[vqc] Loaded model → "
        f"{_VQC_PATH}"
    )

    return vqc


# ============================================================
# COMPLETE VQC STAGE
# ============================================================

def run_vqc_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = N_QUBITS,
    max_iter: int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots: int | None = VQC_SHOTS,
):
    """
    Run complete VQC pipeline.

        Train
          ↓
        Save
          ↓
        Evaluate
          ↓
        Save metrics
    """

    # --------------------------------------------------------
    # Convert arrays
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
        "VQC training data",
    )

    _validate_data(
        X_test,
        y_test,
        "VQC test data",
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

    vqc, training_time = train_vqc(
        X_train=X_train,
        y_train=y_train,
        n_qubits=n_qubits,
        max_iter=max_iter,
        random_state=random_state,
        shots=shots,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_vqc(
        vqc=vqc,
        n_qubits=n_qubits,
        max_iter=max_iter,
        random_state=random_state,
        shots=shots,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_vqc(
        vqc=vqc,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
        n_qubits=n_qubits,
        max_iter=max_iter,
        random_state=random_state,
        shots=shots,
        save=True,
    )

    # --------------------------------------------------------
    # Add actual training size
    # --------------------------------------------------------

    metrics["n_train"] = int(
        len(X_train)
    )

    # --------------------------------------------------------
    # Re-save metrics with n_train
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        METRICS_DIR /
        "vqc_results.json"
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
        f"[vqc] Final metrics → "
        f"{metrics_path}"
    )

    return metrics