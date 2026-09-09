"""
preprocessing.py

Audio loading and preprocessing utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from src.config import (
    SAMPLE_RATE,
    TARGET_DURATION,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path: str | Path) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Load audio as mono waveform.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning("Audio file does not exist: %s", file_path)
        return None, None

    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=SAMPLE_RATE,
            mono=True,
        )

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            logger.warning("Empty audio: %s", file_path)
            return None, None

        return audio, sr

    except Exception as exc:
        logger.warning(
            "Could not load %s: %s",
            file_path,
            exc,
        )

        return None, None


# ============================================================
# REMOVE SILENCE
# ============================================================

def trim_silence(
    audio: np.ndarray,
    top_db: float = 35.0,
) -> np.ndarray:
    """
    Remove only strong leading/trailing silence.
    """

    if audio is None or len(audio) == 0:
        return audio

    try:
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
        )

        if len(trimmed) > 0:
            return trimmed

    except Exception:
        pass

    return audio


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Robust amplitude normalization.
    """

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    abs_audio = np.abs(audio)

    if np.max(abs_audio) < 1e-8:
        return audio

    scale = np.percentile(abs_audio, 99.5)

    if scale < 1e-8:
        scale = np.max(abs_audio)

    audio = audio / scale

    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


# ============================================================
# SELECT INFORMATIVE REGION
# ============================================================

def select_energy_region(
    audio: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Select the highest-energy region when the recording
    is longer than the target duration.
    """

    if len(audio) <= target_length:
        return audio

    if target_length <= 0:
        return audio[:target_length]

    hop = max(target_length // 4, 1)

    best_start = 0
    best_energy = -np.inf

    for start in range(
        0,
        len(audio) - target_length + 1,
        hop,
    ):
        segment = audio[start:start + target_length]

        energy = float(
            np.mean(segment ** 2)
        )

        if energy > best_energy:
            best_energy = energy
            best_start = start

    return audio[
        best_start:
        best_start + target_length
    ]


# ============================================================
# FIXED LENGTH
# ============================================================

def fix_duration(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration: float = TARGET_DURATION,
) -> np.ndarray:
    """
    Convert waveform to fixed duration.
    """

    target_length = int(sr * duration)

    if target_length <= 0:
        raise ValueError("Target duration must be positive.")

    if len(audio) > target_length:

        audio = select_energy_region(
            audio,
            target_length,
        )

    elif len(audio) < target_length:

        missing = target_length - len(audio)

        left = missing // 2
        right = missing - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    return audio[:target_length]


# ============================================================
# MAIN PREPROCESSOR
# ============================================================

def preprocess_audio(
    file_path: str | Path,
) -> np.ndarray | None:
    """
    Complete preprocessing pipeline.
    """

    audio, sr = load_audio(file_path)

    if audio is None:
        return None

    audio = trim_silence(
        audio,
        top_db=35.0,
    )

    audio = normalize_audio(audio)

    audio = fix_duration(
        audio,
        sr=sr,
        duration=TARGET_DURATION,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not np.any(np.abs(audio) > 1e-8):
        logger.warning(
            "Audio became silent after preprocessing: %s",
            file_path,
        )

    return audio.astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================

def validate_audio(audio: np.ndarray) -> bool:

    if audio is None:
        return False

    if len(audio) == 0:
        return False

    if not np.all(np.isfinite(audio)):
        return False

    if np.max(np.abs(audio)) < 1e-8:
        return False

    return True