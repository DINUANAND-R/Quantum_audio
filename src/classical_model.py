"""
classical_model.py — Classical SVM baseline for speech emotion recognition.

Pipeline
--------
    Acoustic Features
          ↓
    StandardScaler     (fit on train only)
          ↓
    SVM (RBF kernel)
          ↓
    Emotion label

This baseline establishes the performance ceiling achievable without
quantum circuits, providing an honest comparison point.

All metrics are computed on the held-out, actor-independent test set.
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

logger = logging.getLogger(__name__)

_SVM_MODEL_PATH = MODELS_DIR / "classical_svm.pkl"
_SVM_SCALER_PATH = MODELS_DIR / "classical_svm_scaler.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classical_svm(
    train_df: pd.DataFrame,
    C: float = SVM_C,
    kernel: str = SVM_KERNEL,
    gamma: str | float = SVM_GAMMA,
    random_state: int = RANDOM_STATE,
) -> tuple[SVC, StandardScaler]:
    """
    Fit a StandardScaler and SVM on the training set.

    Parameters
    ----------
    train_df      : pd.DataFrame  — training feature matrix
    C, kernel, gamma : SVM hyperparameters
    random_state  : int

    Returns
    -------
    (fitted_svm, fitted_scaler)
    """
    feat_cols = _get_feature_cols(train_df)
    X_train   = train_df[feat_cols].values.astype(np.float64)
    y_train   = train_df["emotion"].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[classical_svm] Training SVM on {len(X_train)} samples, "
          f"{X_train.shape[1]} features …")
    print(f"[classical_svm] SVM config: C={C}, kernel={kernel}, gamma={gamma}")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
        probability=True,   # needed for soft fusion in multimodal branch
        decision_function_shape="ovr",
    )
    svm.fit(X_scaled, y_train)
    print("[classical_svm] Training complete.")
    return svm, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classical_svm(
    svm: SVC,
    scaler: StandardScaler,
    test_df: pd.DataFrame,
    save: bool = True,
) -> dict:
    """
    Evaluate the classical SVM on the test set and save results.

    Parameters
    ----------
    svm, scaler  : fitted objects
    test_df      : pd.DataFrame  — test feature matrix

    Returns
    -------
    dict of metrics
    """
    feat_cols = _get_feature_cols(test_df)
    X_test    = test_df[feat_cols].values.astype(np.float64)
    y_test    = test_df["emotion"].values

    X_test    = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled  = scaler.transform(X_test)

    y_pred    = svm.predict(X_scaled)
    y_proba   = svm.predict_proba(X_scaled)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    acc         = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_pre   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    macro_rec   = recall_score(   y_test, y_pred, average="macro",    zero_division=0)
    report      = classification_report(y_test, y_pred, zero_division=0)
    cm          = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))

    metrics = {
        "model"         : "Classical SVM",
        "accuracy"      : float(acc),
        "macro_f1"      : float(macro_f1),
        "weighted_f1"   : float(weighted_f1),
        "macro_precision": float(macro_pre),
        "macro_recall"  : float(macro_rec),
        "n_train"       : 0,   # patched by run_classical_stage after return
        "n_test"        : int(len(X_test)),
        "svm_C"         : SVM_C,
        "svm_kernel"    : SVM_KERNEL,
        "svm_gamma"     : str(SVM_GAMMA),
        "random_state"  : RANDOM_STATE,
    }

    print("\n" + "=" * 60)
    print("  CLASSICAL SVM RESULTS")
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
        _save_classical_results(metrics, y_test, y_pred, y_proba,
                                svm.classes_, cm)
    return metrics


def _save_classical_results(
    metrics: dict,
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    cm:      np.ndarray,
) -> None:
    """Persist metrics, predictions, and confusion matrix figure."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = METRICS_DIR / "classical_svm_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[classical_svm] Metrics  → {metrics_path}")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "true_label": y_test,
        "predicted" : y_pred,
        "correct"   : y_test == y_pred,
    })
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = y_proba[:, i]
    pred_path = PREDICTIONS_DIR / "classical_svm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[classical_svm] Predictions → {pred_path}")

    # Confusion matrix figure
    _plot_confusion_matrix(cm, classes,
                           title="Classical SVM — Confusion Matrix",
                           save_path=FIGURES_DIR / "classical_svm_confusion_matrix.png")


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray | list,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """Render and optionally save a confusion matrix heat-map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Confusion matrix saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Save / load model
# ---------------------------------------------------------------------------

def save_classical_svm(svm: SVC, scaler: StandardScaler) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SVM_MODEL_PATH,  "wb") as f:
        pickle.dump(svm,    f)
    with open(_SVM_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[classical_svm] Model saved → {_SVM_MODEL_PATH}")


def load_classical_svm() -> tuple[SVC, StandardScaler]:
    if not _SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVM model not found at {_SVM_MODEL_PATH}\n"
            f"Run:  python main.py --stage classical"
        )
    with open(_SVM_MODEL_PATH,  "rb") as f:
        svm    = pickle.load(f)
    with open(_SVM_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return svm, scaler


# ---------------------------------------------------------------------------
# Top-level stage runner
# ---------------------------------------------------------------------------

def run_classical_stage(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> dict:
    """Train, evaluate, and save the classical SVM.  Returns metrics dict."""
    svm, scaler = train_classical_svm(train_df)
    save_classical_svm(svm, scaler)
    metrics = evaluate_classical_svm(svm, scaler, test_df, save=True)
    # patch n_train
    metrics["n_train"] = len(train_df)
    return metrics
