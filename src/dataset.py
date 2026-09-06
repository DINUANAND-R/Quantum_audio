"""
dataset.py — RAVDESS dataset discovery, filename parsing, and metadata extraction.

RAVDESS filename format (hyphen-separated):
  Modality - VocalChannel - Emotion - Intensity - Statement - Repetition - Actor.wav

Example:  03-01-05-01-02-01-12.wav
  [0] Modality      : 03 = audio-visual
  [1] VocalChannel  : 01 = speech
  [2] Emotion       : 05 = angry
  [3] Intensity     : 01 = normal, 02 = strong
  [4] Statement     : 01 = "Kids are talking by the door"
                      02 = "Dogs are sitting by the door"
  [5] Repetition    : 01 = 1st, 02 = 2nd
  [6] Actor         : 01–24 (odd = male, even = female)

Emotion mapping: see src/config.py  EMOTION_MAP
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    EMOTION_MAP,
    PROC_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ravdess_filename(filepath: Path) -> Optional[dict]:
    """
    Parse a RAVDESS WAV filename and return a dict of metadata fields.

    Returns None if the filename does not match the expected format.

    Parameters
    ----------
    filepath : Path
        Full path to a single .wav file.

    Returns
    -------
    dict with keys:
        file_path, modality, vocal_channel, emotion_id, emotion,
        intensity, statement, repetition, actor_id
    OR None on parse failure.
    """
    name = filepath.stem          # strip .wav
    parts = name.split("-")

    if len(parts) != 7:
        logger.warning("Unexpected filename format (expected 7 fields): %s", filepath.name)
        return None

    try:
        modality      = parts[0]
        vocal_channel = parts[1]
        emotion_id    = parts[2]
        intensity     = parts[3]
        statement     = parts[4]
        repetition    = parts[5]
        actor_id      = int(parts[6])
    except (ValueError, IndexError) as exc:
        logger.warning("Could not parse filename %s: %s", filepath.name, exc)
        return None

    emotion = EMOTION_MAP.get(emotion_id)
    if emotion is None:
        logger.warning("Unknown emotion code '%s' in file: %s", emotion_id, filepath.name)
        return None

    return {
        "file_path"    : str(filepath),
        "modality"     : modality,
        "vocal_channel": vocal_channel,
        "emotion_id"   : emotion_id,
        "emotion"      : emotion,
        "intensity"    : intensity,
        "statement"    : statement,
        "repetition"   : repetition,
        "actor_id"     : actor_id,
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_wav_files(root: Path) -> list[Path]:
    """
    Recursively find all .wav files under *root*.

    Parameters
    ----------
    root : Path
        Directory to search (typically RAW_DATA_DIR).

    Returns
    -------
    Sorted list of Path objects.
    """
    wavs = sorted(root.rglob("*.wav"))
    logger.info("Found %d WAV files under %s", len(wavs), root)
    return wavs


def build_metadata_dataframe(wav_files: list[Path]) -> pd.DataFrame:
    """
    Parse every WAV file path into a metadata row and return a DataFrame.

    Parameters
    ----------
    wav_files : list[Path]
        List of WAV file paths.

    Returns
    -------
    pd.DataFrame with one row per valid file.
    """
    records = []
    skipped = 0
    for fp in wav_files:
        meta = parse_ravdess_filename(fp)
        if meta is None:
            skipped += 1
            continue
        records.append(meta)

    if skipped > 0:
        logger.warning("Skipped %d files due to parse errors.", skipped)

    df = pd.DataFrame(records)
    logger.info("Parsed %d valid RAVDESS files (%d unique actors, %d emotions).",
                len(df),
                df["actor_id"].nunique() if not df.empty else 0,
                df["emotion"].nunique()  if not df.empty else 0)
    return df


# ---------------------------------------------------------------------------
# Top-level dataset loader
# ---------------------------------------------------------------------------

def load_ravdess_metadata(dataset_root: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Locate RAVDESS WAV files, parse metadata, and return a clean DataFrame.

    Exits with a helpful message if the dataset directory is missing or empty.

    Parameters
    ----------
    dataset_root : Path
        Root directory that contains Actor_01/ … Actor_24/ subdirectories.

    Returns
    -------
    pd.DataFrame
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"\n[DATASET MISSING]\n"
            f"Expected RAVDESS dataset at:\n  {dataset_root}\n\n"
            f"Please download the RAVDESS Speech Audio dataset and place it so that:\n"
            f"  {dataset_root}/Actor_01/\n"
            f"  {dataset_root}/Actor_02/\n"
            f"  ...\n"
            f"  {dataset_root}/Actor_24/\n"
            f"\nDownload link: https://zenodo.org/record/1188976\n"
            f"(Download 'Audio_Speech_Actors_01-24.zip' and extract here.)"
        )

    wav_files = find_wav_files(dataset_root)

    if len(wav_files) == 0:
        raise ValueError(
            f"\n[DATASET EMPTY]\n"
            f"No .wav files found under: {dataset_root}\n"
            f"The RAVDESS speech dataset should contain ~1440 .wav files.\n"
            f"Please verify the directory structure:\n"
            f"  {dataset_root}/Actor_01/*.wav\n"
            f"  {dataset_root}/Actor_02/*.wav\n"
            f"  ...\n"
        )

    df = build_metadata_dataframe(wav_files)

    # Verify we have speech-only files (vocal_channel == "01" and modality == "03")
    speech_df = df[(df["vocal_channel"] == "01") & (df["modality"] == "03")]
    if len(speech_df) != len(df):
        logger.info(
            "Filtering to speech-only files: %d → %d rows.", len(df), len(speech_df)
        )
        df = speech_df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------

def verify_dataset(dataset_root: Path = RAW_DATA_DIR) -> None:
    """
    Print a detailed verification report for the RAVDESS dataset.

    Call this from:  python main.py --stage dataset
    """
    print("=" * 60)
    print("  RAVDESS DATASET VERIFICATION")
    print("=" * 60)
    print(f"  Looking in: {dataset_root}")
    print()

    df = load_ravdess_metadata(dataset_root)

    print(f"  Total valid WAV files  : {len(df)}")
    print(f"  Unique actors          : {sorted(df['actor_id'].unique())}")
    print(f"  Number of actors       : {df['actor_id'].nunique()}")
    print(f"  Unique emotions        : {sorted(df['emotion'].unique())}")
    print(f"  Number of emotion classes: {df['emotion'].nunique()}")
    print()

    print("  Emotion distribution:")
    emotion_counts = df["emotion"].value_counts().sort_index()
    for emotion, count in emotion_counts.items():
        bar = "█" * (count // 5)
        print(f"    {emotion:<12}: {count:4d}  {bar}")
    print()

    print("  Actor distribution (sample count per actor):")
    actor_counts = df.groupby("actor_id").size()
    for actor_id, count in actor_counts.items():
        print(f"    Actor {actor_id:02d}: {count} files")
    print()

    print("  Sample filename parsing:")
    for _, row in df.head(5).iterrows():
        fname = Path(row["file_path"]).name
        print(f"    {fname}")
        print(f"      → emotion: {row['emotion']}, actor: {row['actor_id']}, "
              f"intensity: {row['intensity']}, statement: {row['statement']}")
    print()

    # Save metadata
    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = PROC_DATA_DIR / "ravdess_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"  Metadata saved to: {meta_path}")
    print("=" * 60)
    print("  DATASET VERIFICATION COMPLETE ✓")
    print("=" * 60)
