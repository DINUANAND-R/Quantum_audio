"""
feature_extraction.py — Acoustic feature extraction from RAVDESS audio files.

For each audio file the following features are computed and summarised
with mean / std / min / max across time frames:

  1.  MFCC         (40 coefficients × 4 stats = 160 features)
  2.  Chroma       (12 × 4 = 48 features)
  3.  RMS Energy   (1 × 4 = 4 features)
  4.  Zero Crossing Rate  (1 × 4 = 4 features)
  5.  Spectral Centroid   (1 × 4 = 4 features)
  6.  Spectral Bandwidth  (1 × 4 = 4 features)
  7.  Spectral Rolloff    (1 × 4 = 4 features)
  8.  Spectral Contrast   (7 × 4 = 28 features)
  9.  Fundamental Frequency / Pitch (1 × 4 = 4 features) [best-effort]

Total ≈ 260 features (exact count printed at runtime).

These classical features summarise emotion-relevant acoustic cues:
  - Timbre    → MFCC, Chroma
  - Energy    → RMS
  - Rhythm    → ZCR
  - Spectrum  → Centroid, Bandwidth, Rolloff, Contrast
  - Pitch     → F0 via librosa.yin

After extraction the features are saved to features/audio_features.csv so
that the audio files do NOT need to be reprocessed on every run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_FILE,
    FEATURES_DIR,
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    RANDOM_STATE,
    SAMPLE_RATE,
    STAT_FUNCTIONS,
    TARGET_DURATION,
)
from src.preprocessing import load_audio

logger = logging.getLogger(__name__)

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(track: np.ndarray) -> np.ndarray:
    """
    Compute [mean, std, min, max] over the time-axis of a feature track.

    Parameters
    ----------
    track : np.ndarray  shape (n_features, n_frames) or (n_frames,)

    Returns
    -------
    1-D np.ndarray of statistics (length = n_features × 4)
    """
    if track.ndim == 1:
        track = track[np.newaxis, :]  # shape (1, n_frames)

    stats = np.concatenate([
        np.mean(track, axis=1),
        np.std( track, axis=1),
        np.min( track, axis=1),
        np.max( track, axis=1),
    ])
    return stats


# ---------------------------------------------------------------------------
# Single-file feature extractor
# ---------------------------------------------------------------------------

def extract_features_from_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> Optional[np.ndarray]:
    """
    Extract a fixed-length acoustic feature vector from a waveform.

    Parameters
    ----------
    waveform  : np.ndarray  (n_samples,)
    sr        : int  sample rate
    n_mfcc    : int  number of MFCC coefficients
    n_fft     : int  FFT window size
    hop_length: int  hop length

    Returns
    -------
    np.ndarray of shape (D,) or None on error.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required.  pip install librosa") from exc

    feature_parts: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # 1. MFCC  (timbre, vocal tract shape)
    # ------------------------------------------------------------------
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    feature_parts.append(_summarise(mfcc))

    # ------------------------------------------------------------------
    # 2. Chroma  (pitch class / harmonic content)
    # ------------------------------------------------------------------
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr,
                                         n_fft=n_fft, hop_length=hop_length)
    feature_parts.append(_summarise(chroma))

    # ------------------------------------------------------------------
    # 3. RMS Energy  (loudness / intensity)
    # ------------------------------------------------------------------
    rms = librosa.feature.rms(y=waveform, frame_length=n_fft,
                               hop_length=hop_length)
    feature_parts.append(_summarise(rms))

    # ------------------------------------------------------------------
    # 4. Zero Crossing Rate  (noisiness / voiced vs unvoiced)
    # ------------------------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y=waveform, hop_length=hop_length)
    feature_parts.append(_summarise(zcr))

    # ------------------------------------------------------------------
    # 5. Spectral Centroid  (brightness)
    # ------------------------------------------------------------------
    spec_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length)
    feature_parts.append(_summarise(spec_centroid))

    # ------------------------------------------------------------------
    # 6. Spectral Bandwidth  (spread around centroid)
    # ------------------------------------------------------------------
    spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
    feature_parts.append(_summarise(spec_bw))

    # ------------------------------------------------------------------
    # 7. Spectral Rolloff  (high-frequency energy concentration)
    # ------------------------------------------------------------------
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
    feature_parts.append(_summarise(rolloff))

    # ------------------------------------------------------------------
    # 8. Spectral Contrast  (valley vs peak in sub-bands)
    # ------------------------------------------------------------------
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
    feature_parts.append(_summarise(contrast))

    # ------------------------------------------------------------------
    # 9. Fundamental Frequency / Pitch  (best-effort via YIN)
    # ------------------------------------------------------------------
    try:
        # librosa.yin returns frame-wise F0 estimates
        f0 = librosa.yin(waveform, fmin=50, fmax=500,
                         sr=sr, hop_length=hop_length)
        # Replace unvoiced frames (f0==0 or very low) with NaN then interpolate
        f0 = f0.astype(float)
        f0[f0 < 50] = np.nan
        if np.all(np.isnan(f0)):
            f0 = np.zeros(len(f0))
        else:
            # linear interpolation over NaN gaps
            nans = np.isnan(f0)
            x    = np.arange(len(f0))
            f0[nans] = np.interp(x[nans], x[~nans], f0[~nans])
        feature_parts.append(_summarise(f0[np.newaxis, :]))
    except Exception as exc:
        logger.debug("Pitch extraction failed (non-fatal): %s", exc)
        feature_parts.append(np.zeros(4))   # fallback: 4 zeros

    # ------------------------------------------------------------------
    # Concatenate all parts into one flat vector
    # ------------------------------------------------------------------
    vector = np.concatenate(feature_parts)

    # Sanity check
    if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
        logger.warning("NaN/Inf detected in feature vector; replacing with 0.")
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    return vector


def extract_features_from_file(
    filepath: Path | str,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_duration: float = TARGET_DURATION,
) -> Optional[np.ndarray]:
    """
    Load audio from *filepath* then extract features.

    Returns None if loading or extraction fails.
    """
    result = load_audio(filepath, sample_rate=sample_rate,
                        target_duration=target_duration)
    if result is None:
        return None
    waveform, sr = result
    return extract_features_from_waveform(waveform, sr=sr, n_mfcc=n_mfcc,
                                           n_fft=n_fft, hop_length=hop_length)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_features_for_dataset(
    metadata_df: pd.DataFrame,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Extract features for every row in *metadata_df*.

    The result is cached at FEATURE_FILE.  Set force_recompute=True to
    ignore the cache and recompute from raw audio.

    Parameters
    ----------
    metadata_df     : pd.DataFrame  (output of dataset.load_ravdess_metadata)
    force_recompute : bool

    Returns
    -------
    pd.DataFrame  with feature columns + 'emotion', 'actor_id', 'file_path'
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURE_FILE.exists() and not force_recompute:
        logger.info("Loading cached features from %s", FEATURE_FILE)
        df = pd.read_csv(FEATURE_FILE)
        print(f"[feature_extraction] Loaded {len(df)} cached feature vectors "
              f"from {FEATURE_FILE}")
        return df

    print(f"[feature_extraction] Extracting features for {len(metadata_df)} files …")
    records = []
    failed  = 0

    for idx, row in metadata_df.iterrows():
        filepath = Path(row["file_path"])
        vector   = extract_features_from_file(filepath)

        if vector is None:
            logger.warning("Skipping %s (extraction failed).", filepath.name)
            failed += 1
            continue

        record = {"file_path": str(filepath),
                  "actor_id" : row["actor_id"],
                  "emotion"  : row["emotion"]}
        for i, v in enumerate(vector):
            record[f"feat_{i:04d}"] = v
        records.append(record)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(metadata_df)} files …")

    if failed > 0:
        print(f"[feature_extraction] WARNING: {failed} files failed extraction.")

    df = pd.DataFrame(records)
    df.to_csv(FEATURE_FILE, index=False)
    print(f"[feature_extraction] Saved {len(df)} feature vectors to {FEATURE_FILE}")
    print(f"[feature_extraction] Feature vector dimension: "
          f"{len(df.columns) - 3} features per file")
    return df
