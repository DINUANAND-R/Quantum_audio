"""
dimensionality_reduction.py — PCA for quantum pre-processing.

WHY PCA IS NEEDED FOR QUANTUM CIRCUITS
---------------------------------------
A quantum circuit of N qubits encodes N real values (one per qubit) in
its rotation angles.  Classical audio features produce ~260-dimensional
vectors.  Feeding 260 features into 260 qubits would be computationally
intractable on classical simulators and on current NISQ hardware.

PCA compresses the feature space down to N_PCA_COMPONENTS dimensions
while retaining the directions of maximum variance.

IMPORTANT DATA-LEAKAGE RULE
-----------------------------
  StandardScaler and PCA are ALWAYS fitted on TRAINING data ONLY.
  The test set is transformed using the scaler and PCA fitted on train.
  This is enforced in the code below.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    N_PCA_COMPONENTS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# Where to persist fitted objects
_SCALER_PATH = MODELS_DIR / "scaler.pkl"
_PCA_PATH    = MODELS_DIR / "pca.pkl"


# ---------------------------------------------------------------------------
# Helper – identify feature columns
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only the numeric feature columns (feat_XXXX)."""
    return [c for c in df.columns if c.startswith("feat_")]


# ---------------------------------------------------------------------------
# Fit & transform
# ---------------------------------------------------------------------------

def fit_scaler_pca(
    train_df: pd.DataFrame,
    n_components: int = N_PCA_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[StandardScaler, PCA]:
    """
    Fit StandardScaler and PCA on TRAINING data.

    Parameters
    ----------
    train_df    : pd.DataFrame  — training feature matrix (contains feat_XXXX cols)
    n_components: int           — number of PCA components
    random_state: int

    Returns
    -------
    (fitted_scaler, fitted_pca)
    """
    feat_cols = _get_feature_cols(train_df)
    X_train   = train_df[feat_cols].values.astype(np.float64)

    # Replace any residual NaN/Inf with 0 (defensive)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)

    explained = np.sum(pca.explained_variance_ratio_) * 100
    logger.info(
        "PCA(%d components): explains %.2f%% of training variance.",
        n_components, explained,
    )
    print(f"[PCA] n_components={n_components}  "
          f"Explained variance={explained:.2f}%  "
          f"(of training set)")
    return scaler, pca


def transform(
    df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """
    Apply fitted scaler and PCA to a dataset.

    Parameters
    ----------
    df     : pd.DataFrame  — feature matrix
    scaler : fitted StandardScaler
    pca    : fitted PCA

    Returns
    -------
    np.ndarray  shape (n_samples, n_components)
    """
    feat_cols = _get_feature_cols(df)
    X         = df[feat_cols].values.astype(np.float64)
    X         = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled  = scaler.transform(X)
    X_pca     = pca.transform(X_scaled)
    return X_pca


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract the 'emotion' string labels from a feature DataFrame."""
    return df["emotion"].values


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------

def save_scaler_pca(scaler: StandardScaler, pca: PCA) -> None:
    """Pickle the fitted scaler and PCA to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(_PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    logger.info("Saved scaler → %s", _SCALER_PATH)
    logger.info("Saved PCA    → %s", _PCA_PATH)


def load_scaler_pca() -> tuple[StandardScaler, PCA]:
    """Load previously fitted scaler and PCA from models/."""
    if not _SCALER_PATH.exists() or not _PCA_PATH.exists():
        raise FileNotFoundError(
            f"Fitted scaler/PCA not found.\n"
            f"Run:  python main.py --stage pca   first."
        )
    with open(_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(_PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    return scaler, pca


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_explained_variance(pca: PCA, save: bool = True) -> None:
    """
    Plot cumulative explained variance ratio versus number of PCA components.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_all  = len(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PCA Explained Variance", fontsize=14, fontweight="bold")

    # -- individual component bars
    axes[0].bar(range(1, n_all + 1),
                pca.explained_variance_ratio_ * 100,
                color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Variance")

    # -- cumulative line
    axes[1].plot(range(1, n_all + 1), cumvar, "o-", color="#DD8452", linewidth=2)
    axes[1].axhline(y=90, color="grey", linestyle="--", alpha=0.7, label="90%")
    axes[1].axhline(y=95, color="red",  linestyle="--", alpha=0.7, label="95%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].set_ylim(0, 101)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / "pca_explained_variance.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PCA] Variance plot saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Full PCA stage
# ---------------------------------------------------------------------------

def run_pca_stage(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    n_components: int = N_PCA_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Fit scaler + PCA on train, transform both train and test.

    Returns
    -------
    X_train_pca, y_train, X_test_pca, y_test, scaler, pca
    """
    scaler, pca = fit_scaler_pca(train_df, n_components=n_components)
    save_scaler_pca(scaler, pca)
    plot_explained_variance(pca)

    X_train_pca = transform(train_df, scaler, pca)
    X_test_pca  = transform(test_df,  scaler, pca)
    y_train     = get_labels(train_df)
    y_test      = get_labels(test_df)

    print(f"[PCA] X_train_pca shape : {X_train_pca.shape}")
    print(f"[PCA] X_test_pca  shape : {X_test_pca.shape}")
    return X_train_pca, y_train, X_test_pca, y_test, scaler, pca
