"""
vqc_model.py — Variational Quantum Classifier (VQC).

HOW VQC WORKS
--------------
Unlike QSVC (which uses a fixed quantum kernel), VQC is a parametric model:

  PCA features  →  ZZFeatureMap  →  EfficientSU2(θ)  →  Measurement
                   (encoding)        (trainable)           →  Class

  1. ZZFeatureMap encodes classical features into quantum states.
  2. EfficientSU2 applies parametric rotations and CNOT gates.
  3. A parity-based or softmax readout maps measurement outcomes to classes.
  4. A classical optimizer (SPSA) updates θ to minimise classification loss.

The gradient-free SPSA optimizer is preferred here because:
  a) It does not require parameter-shift gradient computation (2·n evals).
  b) It is noise-resilient, important on simulators with finite shots.
  c) It works with a multiclass softmax output map.

IMPORTANT LIMITATIONS
----------------------
  - VQC training is expensive on classical simulators: each forward pass
    requires running the quantum circuit.
  - For 4 qubits + EfficientSU2(reps=1), there are 16 parameters.
  - Convergence may be limited by local minima (barren plateaus).
  - These limitations are documented honestly; quantum advantage is NOT
    claimed without experimental evidence.

API: qiskit >= 2.0, qiskit-machine-learning >= 0.8
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
    ANSATZ_REPS,
    FEATURE_MAP_REPS,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_QUBITS,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    VQC_MAX_ITER,
    VQC_SEED,
    VQC_SHOTS,
)
from src.quantum_features import build_ansatz, build_feature_map

logger = logging.getLogger(__name__)

_VQC_PATH = MODELS_DIR / "vqc_params.pkl"


# ---------------------------------------------------------------------------
# VQC construction
# ---------------------------------------------------------------------------

def build_vqc(
    n_qubits:     int = N_QUBITS,
    max_iter:     int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots:        int | None = VQC_SHOTS,
):
    """
    Build a VQC using the Qiskit Machine Learning VQC class.

    Qiskit ML >= 0.8 API:
      VQC(feature_map, ansatz, optimizer, ...)

    Parameters
    ----------
    n_qubits     : int
    max_iter     : int  — SPSA iterations
    random_state : int
    shots        : int | None  (None → statevector; int → sampling)

    Returns
    -------
    VQC instance (unfitted)
    """
    try:
        from qiskit_algorithms.optimizers import SPSA
        from qiskit_machine_learning.algorithms.classifiers import VQC as QiskitVQC
    except ImportError:
        try:
            from qiskit.algorithms.optimizers import SPSA
            from qiskit_machine_learning.algorithms import VQC as QiskitVQC
        except ImportError as e:
            raise ImportError(
                f"Could not import VQC or SPSA.\n"
                f"Ensure qiskit-machine-learning >= 0.8 and qiskit-algorithms are installed.\n"
                f"Original error: {e}"
            )

    feature_map = build_feature_map(n_qubits=n_qubits, reps=FEATURE_MAP_REPS)
    ansatz      = build_ansatz(n_qubits=n_qubits, reps=ANSATZ_REPS)
    optimizer   = SPSA(maxiter=max_iter)

    # Sampler primitive – use StatevectorSampler (Qiskit 2.x) if shots=None
    sampler = _build_sampler(shots=shots, random_state=random_state)

    vqc = QiskitVQC(
        feature_map    = feature_map,
        ansatz         = ansatz,
        optimizer      = optimizer,
        sampler        = sampler,
    )

    n_params = ansatz.num_parameters
    print(f"[vqc] Built VQC: {n_qubits} qubits, {n_params} trainable params, "
          f"SPSA max_iter={max_iter}, shots={shots}")
    return vqc


def _build_sampler(shots: int | None, random_state: int):
    """
    Return the appropriate Qiskit Sampler primitive.

    Qiskit 2.x uses:
      StatevectorSampler  (no shots → exact statevector)  — preferred for CPU
      StatevectorSampler(default_shots=N)                 — sampling mode
    """
    try:
        from qiskit.primitives import StatevectorSampler
        if shots is None:
            sampler = StatevectorSampler(seed=random_state)
        else:
            sampler = StatevectorSampler(default_shots=shots, seed=random_state)
        print(f"[vqc] Using StatevectorSampler (Qiskit 2.x), shots={shots}")
        return sampler
    except ImportError:
        pass

    try:
        from qiskit.primitives import Sampler
        sampler = Sampler()
        print(f"[vqc] Using Sampler (Qiskit fallback), shots={shots}")
        return sampler
    except ImportError as e:
        raise ImportError(f"No suitable Qiskit Sampler found: {e}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vqc(
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    n_qubits:     int = N_QUBITS,
    max_iter:     int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots:        int | None = VQC_SHOTS,
) -> tuple:
    """
    Train the VQC.

    Parameters
    ----------
    X_train  : np.ndarray  shape (n_samples, n_qubits)
    y_train  : np.ndarray  shape (n_samples,) – string labels
    n_qubits, max_iter, random_state, shots : see build_vqc

    Returns
    -------
    (fitted_vqc, training_time_seconds)
    """
    vqc = build_vqc(n_qubits=n_qubits, max_iter=max_iter,
                    random_state=random_state, shots=shots)

    print(f"[vqc] Training VQC on {len(X_train)} samples …")
    print(f"[vqc] Expected iterations: {max_iter}  (SPSA uses 2 circuit evals/iter)")
    print(f"[vqc] This may take several minutes on a CPU simulator.")

    t0 = time.time()
    vqc.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"[vqc] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min).")
    return vqc, elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vqc(
    vqc,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    training_time: float = 0.0,
    save:         bool = True,
) -> dict:
    """Evaluate the VQC on the test set."""
    print(f"[vqc] Evaluating on {len(X_test)} test samples …")
    y_pred = vqc.predict(X_test)

    acc         = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_pre   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    macro_rec   = recall_score(   y_test, y_pred, average="macro",    zero_division=0)
    report      = classification_report(y_test, y_pred, zero_division=0)
    classes     = sorted(set(y_test))
    cm          = confusion_matrix(y_test, y_pred, labels=classes)

    metrics = {
        "model"          : "VQC",
        "accuracy"       : float(acc),
        "macro_f1"       : float(macro_f1),
        "weighted_f1"    : float(weighted_f1),
        "macro_precision": float(macro_pre),
        "macro_recall"   : float(macro_rec),
        "n_test"         : int(len(y_test)),
        "n_qubits"       : N_QUBITS,
        "vqc_max_iter"   : VQC_MAX_ITER,
        "training_time_s": float(training_time),
        "random_state"   : RANDOM_STATE,
    }

    print("\n" + "=" * 60)
    print("  VQC RESULTS")
    print("=" * 60)
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  Macro Prec   : {macro_pre:.4f}")
    print(f"  Macro Recall : {macro_rec:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    print()
    print("  Classification Report:")
    print(report)
    print("=" * 60)

    if save:
        _save_vqc_results(metrics, y_test, y_pred, classes, cm)

    return metrics


def _save_vqc_results(
    metrics, y_test, y_pred, classes, cm
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "vqc_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[vqc] Metrics → {metrics_path}")

    pred_df = pd.DataFrame({"true_label": y_test, "predicted": y_pred,
                             "correct": y_test == y_pred})
    pred_path = PREDICTIONS_DIR / "vqc_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[vqc] Predictions → {pred_path}")

    _plot_confusion_matrix(cm, classes,
                           title="VQC — Confusion Matrix",
                           save_path=FIGURES_DIR / "vqc_confusion_matrix.png")


def _plot_confusion_matrix(cm, labels, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
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
        print(f"[plot] VQC confusion matrix saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_vqc(vqc) -> None:
    """Save VQC weights/parameters (safe serialisation)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Try pickling the full model first
        with open(_VQC_PATH, "wb") as f:
            pickle.dump(vqc, f)
        print(f"[vqc] Model saved → {_VQC_PATH}")
    except Exception as exc:
        logger.warning("Could not pickle full VQC (%s). Saving config only.", exc)
        config = {
            "n_qubits" : N_QUBITS,
            "max_iter" : VQC_MAX_ITER,
            "seed"     : VQC_SEED,
            "shots"    : VQC_SHOTS,
            "note"     : "Full model serialisation failed; re-train to reproduce.",
        }
        with open(MODELS_DIR / "vqc_config.json", "w") as f:
            json.dump(config, f, indent=2)


def load_vqc():
    if not _VQC_PATH.exists():
        raise FileNotFoundError(
            f"VQC model not found at {_VQC_PATH}\n"
            f"Run:  python main.py --stage vqc"
        )
    with open(_VQC_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_vqc_stage(
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    n_qubits:     int = N_QUBITS,
    max_iter:     int = VQC_MAX_ITER,
    random_state: int = VQC_SEED,
    shots:        int | None = VQC_SHOTS,
) -> dict:
    """Train, evaluate, and save VQC.  Returns metrics dict."""
    vqc, elapsed = train_vqc(X_train, y_train, n_qubits=n_qubits,
                              max_iter=max_iter, random_state=random_state,
                              shots=shots)
    save_vqc(vqc)
    metrics = evaluate_vqc(vqc, X_test, y_test, training_time=elapsed, save=True)
    metrics["n_train"] = len(X_train)
    return metrics
