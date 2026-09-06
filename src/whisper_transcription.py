"""
whisper_transcription.py — Optional speech-to-text branch using OpenAI Whisper.

PURPOSE
-------
Acoustic features capture HOW something is said (pitch, energy, rhythm).
Linguistic content captures WHAT is said.  For some emotions, the words
spoken carry additional signal that acoustic features cannot capture.

This module provides an OPTIONAL second branch:

  Audio
    ↓
  Whisper ASR
    ↓
  Transcript (text)
    ↓
  Text emotion model  (future work / multimodal fusion)

IMPORTANT
---------
This branch is completely INDEPENDENT from the acoustic-quantum pipeline.
Running this stage is OPTIONAL.  The main acoustic pipeline works without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PROC_DATA_DIR, TRANSCRIPT_CSV, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)


def transcribe_dataset(
    metadata_df: pd.DataFrame,
    model_size: str = WHISPER_MODEL_SIZE,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Transcribe all audio files in *metadata_df* using Whisper.

    Parameters
    ----------
    metadata_df    : pd.DataFrame  — must have 'file_path', 'actor_id', 'emotion'
    model_size     : str  — 'tiny' | 'base' | 'small' | 'medium' | 'large'
    force_recompute: bool

    Returns
    -------
    pd.DataFrame with columns: file_path, actor_id, emotion, transcript
    """
    if TRANSCRIPT_CSV.exists() and not force_recompute:
        print(f"[whisper] Loading cached transcripts from {TRANSCRIPT_CSV}")
        return pd.read_csv(TRANSCRIPT_CSV)

    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "OpenAI Whisper is required for the STT branch.\n"
            "Install it with:  pip install openai-whisper"
        ) from exc

    print(f"[whisper] Loading Whisper model '{model_size}' …")
    model = whisper.load_model(model_size)
    print(f"[whisper] Model loaded.  Transcribing {len(metadata_df)} files …")

    records = []
    failed  = 0

    for idx, row in metadata_df.iterrows():
        filepath = Path(row["file_path"])
        if not filepath.exists():
            logger.warning("Audio file not found: %s", filepath)
            failed += 1
            continue

        try:
            result     = model.transcribe(str(filepath), language="en", fp16=False)
            transcript = result["text"].strip()
        except Exception as exc:
            logger.warning("Transcription failed for %s: %s", filepath.name, exc)
            transcript = ""
            failed    += 1

        records.append({
            "file_path": str(filepath),
            "actor_id" : row["actor_id"],
            "emotion"  : row["emotion"],
            "transcript": transcript,
        })

        if (idx + 1) % 50 == 0:
            print(f"  Transcribed {idx + 1}/{len(metadata_df)} files …")

    df = pd.DataFrame(records)
    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRANSCRIPT_CSV, index=False)
    print(f"[whisper] Saved {len(df)} transcripts → {TRANSCRIPT_CSV}")
    if failed > 0:
        print(f"[whisper] WARNING: {failed} transcription failures.")
    return df
