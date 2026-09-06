"""
tests/test_dataset.py — Unit tests for dataset parsing and splitting.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import parse_ravdess_filename
from src.config import EMOTION_MAP


# ---------------------------------------------------------------------------
# 1. Filename parsing
# ---------------------------------------------------------------------------

class TestFilenameParser:

    def test_valid_filename(self, tmp_path):
        """Standard RAVDESS filename should parse correctly."""
        fname = tmp_path / "03-01-05-01-02-01-12.wav"
        fname.touch()
        result = parse_ravdess_filename(fname)
        assert result is not None
        assert result["emotion_id"] == "05"
        assert result["emotion"]    == "angry"
        assert result["actor_id"]   == 12
        assert result["modality"]   == "03"
        assert result["statement"]  == "02"

    def test_all_emotion_codes(self, tmp_path):
        """Every emotion code in EMOTION_MAP must parse to a string label."""
        for code, label in EMOTION_MAP.items():
            fname = tmp_path / f"03-01-{code}-01-01-01-01.wav"
            fname.touch()
            result = parse_ravdess_filename(fname)
            assert result is not None, f"Failed for code {code}"
            assert result["emotion"] == label

    def test_actor_extraction(self, tmp_path):
        """Actor IDs 01–24 must be extracted as integers."""
        for actor in range(1, 25):
            fname = tmp_path / f"03-01-01-01-01-01-{actor:02d}.wav"
            fname.touch()
            result = parse_ravdess_filename(fname)
            assert result is not None
            assert result["actor_id"] == actor

    def test_invalid_filename_too_few_parts(self, tmp_path):
        """Filenames with wrong number of parts should return None."""
        fname = tmp_path / "03-01-05.wav"
        fname.touch()
        assert parse_ravdess_filename(fname) is None

    def test_unknown_emotion_code(self, tmp_path):
        """Unknown emotion code (e.g. 99) should return None."""
        fname = tmp_path / "03-01-99-01-01-01-01.wav"
        fname.touch()
        assert parse_ravdess_filename(fname) is None


# ---------------------------------------------------------------------------
# 2. Speaker-independent split
# ---------------------------------------------------------------------------

class TestSpeakerSplit:

    def _make_dummy_df(self):
        """Create a dummy feature DataFrame with 24 actors."""
        records = []
        for actor in range(1, 25):
            for _ in range(10):
                records.append({
                    "file_path": f"/fake/actor_{actor:02d}.wav",
                    "actor_id" : actor,
                    "emotion"  : "happy",
                    "feat_0000": np.random.rand(),
                })
        return pd.DataFrame(records)

    def test_no_overlap(self):
        """After split, train and test actor sets must be disjoint."""
        from src.dataset_split import speaker_independent_split
        df = self._make_dummy_df()
        train_df, test_df = speaker_independent_split(df)
        train_actors = set(train_df["actor_id"].unique())
        test_actors  = set(test_df["actor_id"].unique())
        assert train_actors & test_actors == set(), \
            f"Actor overlap: {train_actors & test_actors}"

    def test_split_counts(self):
        """All rows must appear in exactly one split."""
        from src.dataset_split import speaker_independent_split
        df = self._make_dummy_df()
        train_df, test_df = speaker_independent_split(df)
        assert len(train_df) + len(test_df) == len(df)

    def test_overlap_raises(self):
        """Overlapping train/test actor lists must raise ValueError."""
        from src.dataset_split import speaker_independent_split
        df = self._make_dummy_df()
        with pytest.raises(ValueError, match="overlap"):
            speaker_independent_split(df,
                                      train_actors=list(range(1, 22)),
                                      test_actors=[20, 21, 22])


# ---------------------------------------------------------------------------
# 3. Feature extraction sanity
# ---------------------------------------------------------------------------

class TestFeatureExtraction:

    def test_output_shape(self):
        """Feature vector should be 1-D and non-empty."""
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not installed")

        from src.feature_extraction import extract_features_from_waveform

        sr       = 22050
        waveform = np.random.randn(sr * 3).astype(np.float32)
        waveform = waveform / np.max(np.abs(waveform))
        vector   = extract_features_from_waveform(waveform, sr=sr)

        assert vector is not None
        assert vector.ndim == 1
        assert len(vector) > 0
        print(f"\n  Feature vector length: {len(vector)}")

    def test_no_nan_inf(self):
        """Feature vector must contain no NaN or Inf values."""
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not installed")

        from src.feature_extraction import extract_features_from_waveform

        sr       = 22050
        waveform = np.random.randn(sr * 3).astype(np.float32)
        waveform = waveform / np.max(np.abs(waveform))
        vector   = extract_features_from_waveform(waveform, sr=sr)

        assert not np.any(np.isnan(vector)), "NaN values in feature vector"
        assert not np.any(np.isinf(vector)), "Inf values in feature vector"


# ---------------------------------------------------------------------------
# 4. PCA output dimensions
# ---------------------------------------------------------------------------

class TestPCA:

    def test_pca_output_shape(self):
        """PCA output must have correct number of components."""
        from src.dimensionality_reduction import fit_scaler_pca, transform

        n_samples    = 50
        n_features   = 100
        n_components = 4

        # Build dummy DataFrame
        feat_cols = {f"feat_{i:04d}": np.random.randn(n_samples)
                     for i in range(n_features)}
        df = pd.DataFrame(feat_cols)
        df["emotion"]   = "happy"
        df["actor_id"]  = 1
        df["file_path"] = "dummy.wav"

        scaler, pca = fit_scaler_pca(df, n_components=n_components)
        X_pca       = transform(df, scaler, pca)

        assert X_pca.shape == (n_samples, n_components), \
            f"Expected ({n_samples}, {n_components}), got {X_pca.shape}"

    def test_no_test_leakage(self):
        """Scaler and PCA must be fitted ONLY on train data."""
        from src.dimensionality_reduction import fit_scaler_pca, transform

        n_train, n_test, n_features, n_comp = 40, 10, 50, 4

        def _make_df(n):
            fc = {f"feat_{i:04d}": np.random.randn(n) for i in range(n_features)}
            df = pd.DataFrame(fc)
            df["emotion"] = "happy"; df["actor_id"] = 1; df["file_path"] = "x.wav"
            return df

        train_df, test_df = _make_df(n_train), _make_df(n_test)
        scaler, pca = fit_scaler_pca(train_df, n_components=n_comp)

        # Transform test with train-fitted objects (correct)
        X_test = transform(test_df, scaler, pca)
        assert X_test.shape == (n_test, n_comp)
