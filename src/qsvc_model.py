"""
qsvc_model.py — Quantum Support Vector Classifier (QSVC).

HOW QSVC WORKS
---------------
  1. Classical PCA features → ZZFeatureMap → quantum state |ψ(x)⟩
  2. Quantum kernel  K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|²
     (The overlap of two quantum states serves as the kernel function)
  3. This kernel matrix is fed into a classical SVM
  4. The SVM finds the optimal hyperplane in the RKHS induced by the
     quantum kernel

The kernel computation is the only quantum step; the SVM optimisation
is entirely classical.

API: qiskit >= 2.0, qiskit-machine-learning >= 0.8
"""

from __future__ import annotations

import json
import logging
import pickle
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

_QSVC_PATH = MODELS_DIR / "qsvc.pkl"


# ---------------------------------------------------------------------------
# Build & train
# ---------------------------------------------------------------------------

def build_qsvc(n_qubits: int = N_QUBITS):
    """
    Construct a QSVC using the FidelityQuantumKernel.

    Qiskit Machine Learning >= 0.7 uses:
      FidelityQuantumKernel  (replaced QuantumKernel in older versions)

    Parameters
    ----------
    n_qubits : int

    Returns
    -------
    QSVC instance (unfitted)
    """
    try:
        # Qiskit Machine Learning >= 0.7 API
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC as QiskitQSVC

        feature_map = build_feature_map(n_qubits=n_qubits)
        kernel      = FidelityQuantumKernel(feature_map=feature_map)
        qsvc        = QiskitQSVC(quantum_kernel=kernel)
        print(f"[qsvc] Using FidelityQuantumKernel (Qiskit ML >= 0.7 API)")
        return qsvc

    except ImportError as e:
        raise ImportError(
            f"Could not import QSVC or FidelityQuantumKernel.\n"
            f"Ensure qiskit-machine-learning >= 0.7 is installed.\n"
            f"Original error: {e}"
        )


def train_qsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int = N_QUBITS,
) -> object:
    """
    Train the QSVC.

    Parameters
    ----------
    X_train  : np.ndarray  shape (n_samples, n_qubits)
    y_train  : np.ndarray  shape (n_samples,)
    n_qubits : int

    Returns
    -------
    Fitted QSVC
    """
    qsvc = build_qsvc(n_qubits=n_qubits)
    print(f"[qsvc] Training QSVC on {len(X_train)} samples, "
          f"{n_qubits} qubits (quantum kernel computation) …")
    print(f"[qsvc] NOTE: Kernel matrix computation has O(n²) cost; "
          f"may be slow for large datasets.")

    qsvc.fit(X_train, y_train)
    print("[qsvc] Training complete.")
    return qsvc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_qsvc(
    qsvc,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    classes: list | None = None,
    save: bool = True,
) -> dict:
    """
    Evaluate the trained QSVC and save results.

    Parameters
    ----------
    qsvc    : fitted QSVC
    X_test  : np.ndarray
    y_test  : np.ndarray
    classes : optional list of class names
    save    : bool — persist outputs

    Returns
    -------
    dict of metrics
    """
    print(f"[qsvc] Evaluating on {len(X_test)} test samples …")
    y_pred = qsvc.predict(X_test)

    if classes is None:
        classes = sorted(set(y_test))

    acc         = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_pre   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    macro_rec   = recall_score(   y_test, y_pred, average="macro",    zero_division=0)
    report      = classification_report(y_test, y_pred, zero_division=0)
    cm          = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))

    metrics = {
        "model"          : "QSVC",
        "accuracy"       : float(acc),
        "macro_f1"       : float(macro_f1),
        "weighted_f1"    : float(weighted_f1),
        "macro_precision": float(macro_pre),
        "macro_recall"   : float(macro_rec),
        "n_test"         : int(len(y_test)),
        "n_qubits"       : N_QUBITS,
        "feature_map_reps": int(2),
        "random_state"   : RANDOM_STATE,
    }

    print("\n" + "=" * 60)
    print("  QSVC RESULTS")
    print("=" * 60)
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Macro Prec   : {macro_pre:.4f}")
    print(f"  Macro Recall : {macro_rec:.4f}")
    print()
    print("  Classification Report:")
    print(report)
    print("=" * 60)

    if save:
        _save_qsvc_results(metrics, y_test, y_pred, list(classes), cm)

    return metrics


def _save_qsvc_results(
    metrics: dict,
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    classes: list,
    cm:      np.ndarray,
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "qsvc_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[qsvc] Metrics → {metrics_path}")

    pred_df = pd.DataFrame({"true_label": y_test, "predicted": y_pred,
                             "correct": y_test == y_pred})
    pred_path = PREDICTIONS_DIR / "qsvc_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[qsvc] Predictions → {pred_path}")

    _plot_confusion_matrix(cm, classes,
                           title="QSVC — Confusion Matrix",
                           save_path=FIGURES_DIR / "qsvc_confusion_matrix.png")


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Confusion matrix saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_qsvc(qsvc) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_QSVC_PATH, "wb") as f:
        pickle.dump(qsvc, f)
    print(f"[qsvc] Model saved → {_QSVC_PATH}")


def load_qsvc():
    if not _QSVC_PATH.exists():
        raise FileNotFoundError(
            f"QSVC model not found at {_QSVC_PATH}\n"
            f"Run:  python main.py --stage qsvc"
        )
    with open(_QSVC_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_qsvc_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_qubits: int = N_QUBITS,
) -> dict:
    """Train, evaluate, and save QSVC.  Returns metrics dict."""
    qsvc    = train_qsvc(X_train, y_train, n_qubits=n_qubits)
    save_qsvc(qsvc)
    metrics = evaluate_qsvc(qsvc, X_test, y_test, save=True)
    metrics["n_train"] = len(X_train)
    return metrics
