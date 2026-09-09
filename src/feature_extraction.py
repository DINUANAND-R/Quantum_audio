"""
feature_extraction.py

Extract robust acoustic features for speech emotion recognition.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_FILE,
    FEATURES_DIR,
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
)
from src.preprocessing import preprocess_audio

logger = logging.getLogger(__name__)


# ============================================================
# STATISTICS
# ============================================================

def _statistics(values: np.ndarray) -> list[float]:
    """
    Calculate robust statistics.
    """

    values = np.asarray(values, dtype=np.float32)

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return [0.0] * 7

    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def _flatten_statistics(
    matrix: np.ndarray,
    prefix: str,
) -> dict[str, float]:

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    result = {}

    for i in range(matrix.shape[0]):

        stats = _statistics(
            matrix[i]
        )

        names = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
        ]

        for name, value in zip(
            names,
            stats,
        ):
            result[
                f"{prefix}{i+1}_{name}"
            ] = value

    return result


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict[str, float]:
    """
    Extract acoustic features from waveform.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    features: dict[str, float] = {}

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    features.update(
        _flatten_statistics(
            mfcc,
            "mfcc_",
        )
    )

    # --------------------------------------------------------
    # MFCC DELTA
    # --------------------------------------------------------

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    features.update(
        _flatten_statistics(
            mfcc_delta,
            "mfcc_delta_",
        )
    )

    # --------------------------------------------------------
    # MFCC DELTA DELTA
    # --------------------------------------------------------

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2,
    )

    features.update(
        _flatten_statistics(
            mfcc_delta2,
            "mfcc_delta2_",
        )
    )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    features.update(
        _flatten_statistics(
            chroma,
            "chroma_",
        )
    )

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features.update(
        _flatten_statistics(
            rms.reshape(1, -1),
            "rms_",
        )
    )

    # --------------------------------------------------------
    # ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features.update(
        _flatten_statistics(
            zcr.reshape(1, -1),
            "zcr_",
        )
    )

    # --------------------------------------------------------
    # SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features.update(
        _flatten_statistics(
            centroid.reshape(1, -1),
            "centroid_",
        )
    )

    # --------------------------------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features.update(
        _flatten_statistics(
            bandwidth.reshape(1, -1),
            "bandwidth_",
        )
    )

    # --------------------------------------------------------
    # SPECTRAL ROLLOFF
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features.update(
        _flatten_statistics(
            rolloff.reshape(1, -1),
            "rolloff_",
        )
    )

    # --------------------------------------------------------
    # SPECTRAL CONTRAST
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    features.update(
        _flatten_statistics(
            contrast,
            "contrast_",
        )
    )

    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    try:

        f0 = librosa.yin(
            audio,
            fmin=65,
            fmax=500,
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )

        valid_f0 = f0[
            np.isfinite(f0)
        ]

        valid_f0 = valid_f0[
            valid_f0 > 0
        ]

        if valid_f0.size > 0:

            pitch_stats = _statistics(
                valid_f0
            )

            names = [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "q25",
                "q75",
            ]

            for name, value in zip(
                names,
                pitch_stats,
            ):
                features[
                    f"pitch_{name}"
                ] = value

            features["pitch_range"] = float(
                np.max(valid_f0)
                - np.min(valid_f0)
            )

            features["voiced_ratio"] = float(
                len(valid_f0) / max(len(f0), 1)
            )

        else:

            for name in [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "q25",
                "q75",
            ]:
                features[
                    f"pitch_{name}"
                ] = 0.0

            features["pitch_range"] = 0.0
            features["voiced_ratio"] = 0.0

    except Exception:

        for name in [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
        ]:
            features[
                f"pitch_{name}"
            ] = 0.0

        features["pitch_range"] = 0.0
        features["voiced_ratio"] = 0.0

    # --------------------------------------------------------
    # ENERGY DYNAMICS
    # --------------------------------------------------------

    energy = rms

    if len(energy) > 1:

        energy_diff = np.diff(
            energy
        )

        features[
            "energy_change_mean"
        ] = float(
            np.mean(energy_diff)
        )

        features[
            "energy_change_std"
        ] = float(
            np.std(energy_diff)
        )

        features[
            "energy_change_abs_mean"
        ] = float(
            np.mean(np.abs(energy_diff))
        )

    else:

        features[
            "energy_change_mean"
        ] = 0.0

        features[
            "energy_change_std"
        ] = 0.0

        features[
            "energy_change_abs_mean"
        ] = 0.0

    # --------------------------------------------------------
    # GLOBAL AUDIO STATISTICS
    # --------------------------------------------------------

    features["audio_mean"] = float(
        np.mean(audio)
    )

    features["audio_std"] = float(
        np.std(audio)
    )

    features["audio_min"] = float(
        np.min(audio)
    )

    features["audio_max"] = float(
        np.max(audio)
    )

    features["audio_rms"] = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    for key, value in features.items():

        if not np.isfinite(value):
            features[key] = 0.0

    return features


# ============================================================
# SINGLE FILE
# ============================================================

def extract_features_from_file(
    file_path: str | Path,
) -> dict[str, float] | None:

    audio = preprocess_audio(
        file_path
    )

    if audio is None:
        return None

    try:

        return extract_features(
            audio,
            SAMPLE_RATE,
        )

    except Exception as exc:

        logger.warning(
            "Feature extraction failed for %s: %s",
            file_path,
            exc,
        )

        return None


# ============================================================
# DATASET
# ============================================================

def extract_features_for_dataset(
    metadata_df: pd.DataFrame,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Extract features for complete dataset.
    """

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        FEATURE_FILE.exists()
        and not force_recompute
    ):

        print(
            f"[feature_extraction] "
            f"Loading cached features from "
            f"{FEATURE_FILE}"
        )

        df = pd.read_csv(
            FEATURE_FILE
        )

        if "actor_id" not in df.columns:
            raise ValueError(
                "Cached feature file does not contain actor_id. "
                "Run with --force-recompute."
            )

        return df

    rows = []

    failed = 0

    for index, row in metadata_df.iterrows():

        file_path = row["file_path"]

        extracted = extract_features_from_file(
            file_path
        )

        if extracted is None:

            failed += 1

            print(
                f"[feature_extraction] "
                f"Skipping {Path(file_path).name}"
            )

            continue

        result = {}

        # Metadata
        for column in metadata_df.columns:
            result[column] = row[column]

        # Features
        for key, value in extracted.items():
            result[f"feat_{key}"] = value

        rows.append(result)

        if (index + 1) % 100 == 0:
            print(
                f"[feature_extraction] "
                f"Processed {index + 1}/"
                f"{len(metadata_df)}"
            )

    if not rows:

        raise RuntimeError(
            "No audio features were extracted. "
            "Check the RAVDESS audio files and paths."
        )

    feature_df = pd.DataFrame(
        rows
    )

    feature_columns = [
        c
        for c in feature_df.columns
        if c.startswith("feat_")
    ]

    feature_df[
        feature_columns
    ] = feature_df[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    feature_df.to_csv(
        FEATURE_FILE,
        index=False,
    )

    print(
        f"\n[feature_extraction] "
        f"Saved {len(feature_df)} feature vectors"
    )

    print(
        f"[feature_extraction] "
        f"Failed files: {failed}"
    )

    print(
        f"[feature_extraction] "
        f"Feature dimension: "
        f"{len(feature_columns)}"
    )

    return feature_df