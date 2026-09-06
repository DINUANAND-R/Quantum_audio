"""
dataset_split.py — Speaker-independent train/test split for RAVDESS.

CRITICAL: The same actor MUST NOT appear in both train and test sets.
          Splitting individual files randomly (ignoring speaker identity)
          would cause data leakage — the model could memorise actor-specific
          voice characteristics rather than learning emotional patterns.

Strategy
--------
  Training actors : Actor 01 – 20   (configurable via config.py)
  Testing  actors : Actor 21 – 24   (configurable via config.py)

Both actor splits are configurable in src/config.py:
  TRAIN_ACTORS = list(range(1, 21))
  TEST_ACTORS  = list(range(21, 25))

Output
------
  data/processed/train.csv
  data/processed/test.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    PROC_DATA_DIR,
    TEST_ACTORS,
    TEST_CSV,
    TRAIN_ACTORS,
    TRAIN_CSV,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

def speaker_independent_split(
    feature_df: pd.DataFrame,
    train_actors: list[int] = TRAIN_ACTORS,
    test_actors:  list[int] = TEST_ACTORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *feature_df* into train and test by actor ID.

    Verifies that there is NO overlap between train and test actor sets.

    Parameters
    ----------
    feature_df   : pd.DataFrame  — full feature dataset (must have 'actor_id')
    train_actors : list[int]
    test_actors  : list[int]

    Returns
    -------
    (train_df, test_df)
    """
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    overlap = set(train_actors) & set(test_actors)
    if overlap:
        raise ValueError(
            f"Actor overlap detected between train and test sets: {overlap}\n"
            f"This would cause data leakage.  Fix TRAIN_ACTORS / TEST_ACTORS "
            f"in src/config.py."
        )

    if "actor_id" not in feature_df.columns:
        raise KeyError("'actor_id' column not found in feature_df.")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    train_df = feature_df[feature_df["actor_id"].isin(train_actors)].copy()
    test_df  = feature_df[feature_df["actor_id"].isin(test_actors)].copy()

    # ------------------------------------------------------------------
    # Post-split verification (belt-and-suspenders)
    # ------------------------------------------------------------------
    actual_train_actors = set(train_df["actor_id"].unique())
    actual_test_actors  = set(test_df["actor_id"].unique())
    verified_overlap    = actual_train_actors & actual_test_actors

    if verified_overlap:
        raise RuntimeError(
            f"[BUG] Actor overlap survived the split: {verified_overlap}"
        )

    logger.info(
        "Split: %d train samples (%d actors) | %d test samples (%d actors)",
        len(train_df), len(actual_train_actors),
        len(test_df),  len(actual_test_actors),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train.csv and test.csv to data/processed/."""
    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv( TEST_CSV,  index=False)
    logger.info("Saved train split → %s", TRAIN_CSV)
    logger.info("Saved test  split → %s", TEST_CSV)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved train.csv and test.csv."""
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Split CSVs not found at:\n  {TRAIN_CSV}\n  {TEST_CSV}\n"
            f"Run:  python main.py --stage features   first."
        )
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------

def verify_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print a detailed verification report for the split."""
    print("=" * 60)
    print("  SPEAKER-INDEPENDENT SPLIT VERIFICATION")
    print("=" * 60)

    train_actors = sorted(train_df["actor_id"].unique())
    test_actors  = sorted(test_df["actor_id"].unique())
    overlap      = set(train_actors) & set(test_actors)

    print(f"  Training actors  : {train_actors}")
    print(f"  Testing  actors  : {test_actors}")
    print(f"  Actor overlap    : {overlap if overlap else '∅ (NONE — GOOD)'}")
    print()
    print(f"  Training samples : {len(train_df)}")
    print(f"  Testing  samples : {len(test_df)}")
    print()

    print("  Class distribution in TRAIN:")
    train_counts = train_df["emotion"].value_counts().sort_index()
    for emo, cnt in train_counts.items():
        print(f"    {emo:<12}: {cnt}")

    print("\n  Class distribution in TEST:")
    test_counts = test_df["emotion"].value_counts().sort_index()
    for emo, cnt in test_counts.items():
        print(f"    {emo:<12}: {cnt}")

    print()
    if overlap:
        print("  ⚠ WARNING: Actor overlap detected!  Data leakage risk!")
    else:
        print("  ✓ No actor overlap.  Speaker-independent split is VALID.")
    print("=" * 60)
