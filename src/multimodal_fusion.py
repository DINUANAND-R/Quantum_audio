"""
multimodal_fusion.py — Late fusion of acoustic and text emotion predictions.

ARCHITECTURE
------------
                 ┌── Acoustic Features → PCA → Quantum Model → P(emotion | audio)
  Audio ────────┤
                 └── Whisper → Text → Text Model             → P(emotion | text)

Final prediction:
  P(final) = α · P(emotion | audio) + (1 - α) · P(emotion | text)

where α ∈ [0, 1] is a configurable fusion weight (default 0.5).

IMPORTANT
----------
Multimodal improvement is NOT claimed unless experimental results show it.
Results are reported honestly.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import FUSION_ALPHA, METRICS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def late_fusion(
    proba_audio: np.ndarray,
    proba_text:  np.ndarray,
    classes:     list[str],
    alpha: float = FUSION_ALPHA,
    y_true: np.ndarray | None = None,
) -> dict:
    """
    Combine audio and text probability vectors via weighted averaging.

    Parameters
    ----------
    proba_audio : np.ndarray  shape (n_samples, n_classes)  — audio branch probas
    proba_text  : np.ndarray  shape (n_samples, n_classes)  — text branch probas
    classes     : list[str]   — class names matching columns of proba arrays
    alpha       : float       — weight for audio branch
    y_true      : np.ndarray  — ground truth labels (for evaluation)

    Returns
    -------
    dict with 'y_pred', 'proba_fused', and optionally 'accuracy', 'macro_f1'
    """
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    proba_fused = alpha * proba_audio + (1.0 - alpha) * proba_text
    y_pred = np.array([classes[i] for i in np.argmax(proba_fused, axis=1)])

    result = {"alpha": alpha, "y_pred": y_pred, "proba_fused": proba_fused}

    if y_true is not None:
        acc      = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        result["accuracy"]  = float(acc)
        result["macro_f1"]  = float(macro_f1)
        print(f"[fusion] alpha={alpha:.2f}  Accuracy={acc:.4f}  Macro F1={macro_f1:.4f}")

    return result


def sweep_alpha(
    proba_audio: np.ndarray,
    proba_text:  np.ndarray,
    classes:     list[str],
    y_true:      np.ndarray,
    alphas: list[float] | None = None,
) -> pd.DataFrame:
    """
    Test several fusion weights and return a summary DataFrame.

    Parameters
    ----------
    alphas : list of α values to test (default: [0.0, 0.25, 0.5, 0.75, 1.0])

    Returns
    -------
    pd.DataFrame sorted by Macro F1 descending
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    rows = []
    for alpha in alphas:
        res = late_fusion(proba_audio, proba_text, classes, alpha=alpha, y_true=y_true)
        rows.append({"alpha": alpha,
                     "accuracy":  res.get("accuracy",  float("nan")),
                     "macro_f1":  res.get("macro_f1",  float("nan"))})

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n[fusion] Alpha sweep results:")
    print(df.to_string(index=False))

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "multimodal_fusion_sweep.csv"
    df.to_csv(path, index=False)
    print(f"[fusion] Sweep results → {path}")
    return df
