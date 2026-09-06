"""
evaluation.py — Model comparison and final evaluation report.

Aggregates results from all trained models (Classical SVM, QSVC, VQC),
produces a comparison table, and saves a bar-chart figure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR

logger = logging.getLogger(__name__)

# Canonical metric files
_METRIC_FILES = {
    "Classical SVM": METRICS_DIR / "classical_svm_results.json",
    "QSVC"         : METRICS_DIR / "qsvc_results.json",
    "VQC"          : METRICS_DIR / "vqc_results.json",
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_metrics() -> dict[str, dict]:
    """Load all available model metric JSON files."""
    results = {}
    for model_name, path in _METRIC_FILES.items():
        if path.exists():
            with open(path) as f:
                results[model_name] = json.load(f)
            logger.info("Loaded metrics for %s", model_name)
        else:
            logger.warning("Metrics not found for %s at %s", model_name, path)
    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from metrics dicts.

    Returns
    -------
    pd.DataFrame with columns: Model, Accuracy, Macro F1, Weighted F1
    """
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Model"      : model_name,
            "Accuracy"   : m.get("accuracy",    float("nan")),
            "Macro F1"   : m.get("macro_f1",    float("nan")),
            "Weighted F1": m.get("weighted_f1", float("nan")),
            "Macro Prec" : m.get("macro_precision", float("nan")),
            "Macro Recall": m.get("macro_recall",   float("nan")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_model_comparison(df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart comparing Accuracy, Macro F1, and Weighted F1 across models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models  = df["Model"].tolist()
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    x       = np.arange(len(models))
    width   = 0.25
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = df[metric].tolist()
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=color, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classical vs Quantum Model Comparison\n"
                 "(RAVDESS Speech Emotion Recognition)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / "model_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[evaluation] Comparison plot → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_comparison_report(df: pd.DataFrame) -> None:
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 70)
    print("  CLASSICAL vs QUANTUM MODEL COMPARISON")
    print("  Dataset: RAVDESS Speech Emotion Recognition")
    print("=" * 70)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'W-F1':>10}")
    print("  " + "-" * 54)
    for _, row in df.iterrows():
        acc  = f"{row['Accuracy']:.4f}"  if not np.isnan(row['Accuracy'])    else "N/A"
        mf1  = f"{row['Macro F1']:.4f}"  if not np.isnan(row['Macro F1'])    else "N/A"
        wf1  = f"{row['Weighted F1']:.4f}" if not np.isnan(row['Weighted F1']) else "N/A"
        print(f"  {row['Model']:<20} {acc:>10} {mf1:>10} {wf1:>10}")
    print("=" * 70)

    # Identify best model
    if not df["Macro F1"].isna().all():
        best_idx = df["Macro F1"].idxmax()
        best     = df.loc[best_idx, "Model"]
        best_f1  = df.loc[best_idx, "Macro F1"]
        print(f"\n  Best model by Macro F1: {best}  ({best_f1:.4f})")
        print()
        if best != "Classical SVM":
            print("  NOTE: Quantum model outperformed classical baseline.")
            print("        Verify results carefully before drawing conclusions.")
        else:
            print("  NOTE: Classical SVM outperformed quantum models.")
            print("        This is expected for small qubit counts and limited iterations.")
            print("        Possible reasons for quantum under-performance:")
            print("          • Limited number of qubits (4) — insufficient expressivity")
            print("          • PCA compression loses emotion-relevant information")
            print("          • Few VQC iterations — optimizer may not have converged")
            print("          • Barren plateaus in the variational landscape")
            print("          • Classical simulator limitations (finite shots)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_evaluation_stage() -> pd.DataFrame:
    """Load all metrics, compare, and save the comparison CSV + figure."""
    results = load_all_metrics()

    if not results:
        print("[evaluation] No model results found.  Run training stages first.")
        return pd.DataFrame()

    df = build_comparison_table(results)

    # Save comparison CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[evaluation] Comparison table → {csv_path}")

    print_comparison_report(df)
    plot_model_comparison(df)
    return df
