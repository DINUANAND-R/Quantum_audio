"""
preprocessing.py — Audio loading and preprocessing for RAVDESS.

Each audio file is:
  1. Loaded with librosa at a fixed sample rate
  2. Converted to mono
  3. Amplitude-normalized
  4. Trimmed or zero-padded to TARGET_DURATION seconds
  5. Validated for corruption / NaN / all-silence

No raw audio is modified on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.config import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    TARGET_DURATION,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_audio(
    filepath: Path | str,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    target_duration: float = TARGET_DURATION,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load a WAV file and return (waveform, sample_rate).

    Steps
    -----
    1. Attempt to load with librosa.
    2. Trim/pad to target_duration.
    3. Normalize amplitude to [-1, 1].
    4. Validate the waveform (no NaN, not all-zero).

    Returns None if any step fails.

    Parameters
    ----------
    filepath : Path | str
    sample_rate : int
    mono : bool
    target_duration : float  (seconds)

    Returns
    -------
    (np.ndarray, int) or None
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio preprocessing.\n"
            "Install it with:  pip install librosa"
        ) from exc

    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("Audio file not found: %s", filepath)
        return None

    try:
        waveform, sr = librosa.load(str(filepath), sr=sample_rate, mono=mono)
    except Exception as exc:
        logger.warning("Could not load %s: %s", filepath.name, exc)
        return None

    # ------------------------------------------------------------------
    # Trim silence from beginning and end
    # ------------------------------------------------------------------
    try:
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
    except Exception:
        pass  # non-fatal; continue with untrimed waveform

    # ------------------------------------------------------------------
    # Fix duration: trim or zero-pad to target_duration
    # ------------------------------------------------------------------
    target_samples = int(target_duration * sample_rate)
    if len(waveform) > target_samples:
        waveform = waveform[:target_samples]
    elif len(waveform) < target_samples:
        pad_width = target_samples - len(waveform)
        waveform  = np.pad(waveform, (0, pad_width), mode="constant")

    # ------------------------------------------------------------------
    # Amplitude normalization
    # ------------------------------------------------------------------
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp
    else:
        logger.warning("All-zero (silent) waveform: %s", filepath.name)
        return None

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.warning("NaN/Inf in waveform: %s", filepath.name)
        return None

    return waveform, sr


# ---------------------------------------------------------------------------
# Batch validator (non-destructive – just checks, doesn't save)
# ---------------------------------------------------------------------------

def validate_audio_files(file_paths: list[str | Path]) -> dict:
    """
    Try to load each file and report success/failure counts.

    Parameters
    ----------
    file_paths : list of str | Path

    Returns
    -------
    dict with keys 'valid', 'invalid', 'invalid_paths'
    """
    valid   = []
    invalid = []

    for fp in file_paths:
        result = load_audio(fp)
        if result is None:
            invalid.append(str(fp))
        else:
            valid.append(str(fp))

    logger.info("Audio validation: %d valid, %d invalid.", len(valid), len(invalid))
    return {"valid": valid, "invalid": invalid, "n_valid": len(valid), "n_invalid": len(invalid)}
