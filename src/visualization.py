"""
visualization.py — Feature-level visualizations for the research pipeline.

Generates:
  1. Emotion class distribution bar chart
  2. Feature correlation heat-map (optional — can be slow for 260 features)
  3. PCA 2D scatter plot colored by emotion
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import EMOTION_MAP, FIGURES_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Emotion distribution
# ---------------------------------------------------------------------------

def plot_emotion_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of emotion class counts."""
    counts = df["emotion"].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white",
                  linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("RAVDESS — Emotion Class Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        out = FIGURES_DIR / "emotion_distribution.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[visualization] Emotion distribution → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Feature distributions (first N features as violins)
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    n_features: int = 10,
    save: bool = True,
) -> None:
    """Violin plots of the first *n_features* acoustic features."""
    feat_cols = [c for c in df.columns if c.startswith("feat_")][:n_features]
    if not feat_cols:
        logger.warning("No feat_* columns found; skipping feature distribution plot.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Feature Distributions (first 10 features)", fontsize=13, fontweight="bold")

    for ax, col in zip(axes.flatten(), feat_cols):
        emotions  = df["emotion"].unique()
        data      = [df.loc[df["emotion"] == e, col].dropna().values for e in sorted(emotions)]
        parts     = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(emotions) + 1))
        ax.set_xticklabels(sorted(emotions), rotation=45, ha="right", fontsize=7)
        ax.set_title(col, fontsize=8)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / "feature_distributions.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[visualization] Feature distributions → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. PCA 2D scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    df: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Project features to 2D via PCA and scatter-plot colored by emotion.
    This is a quick visual sanity check — does emotion-relevant structure exist?
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        return

    X = df[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    X_2d     = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    emotions = sorted(df["emotion"].unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(emotions))
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for e in emotions:
        mask = df["emotion"].values == e
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=e, alpha=0.6, s=25,
                   color=cmap(emotion_to_idx[e]))

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA of Acoustic Features (colored by emotion)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        out = FIGURES_DIR / "pca_scatter_2d.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[visualization] PCA 2D scatter → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Run all visualizations
# ---------------------------------------------------------------------------

def run_visualization_stage(df: pd.DataFrame) -> None:
    """Generate all feature-level visualizations."""
    print("[visualization] Generating feature visualizations …")
    plot_emotion_distribution(df)
    plot_feature_distributions(df)
    plot_pca_scatter(df)
    print("[visualization] Done.")
