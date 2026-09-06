"""
config.py — Central configuration for Quantum-Audio-Emotion.

All hyperparameters, paths, and toggles live here.
No magic numbers should appear anywhere else in the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolves correctly regardless of where the script is called)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DATA_DIR   = DATA_DIR / "raw" / "RAVDESS"
PROC_DATA_DIR  = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Feature paths
# ---------------------------------------------------------------------------
FEATURES_DIR   = PROJECT_ROOT / "features"
FEATURE_FILE   = FEATURES_DIR / "audio_features.csv"
TRAIN_CSV      = PROC_DATA_DIR / "train.csv"
TEST_CSV       = PROC_DATA_DIR / "test.csv"
TRANSCRIPT_CSV = PROC_DATA_DIR / "transcripts.csv"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR     = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Results paths
# ---------------------------------------------------------------------------
RESULTS_DIR    = PROJECT_ROOT / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
METRICS_DIR    = RESULTS_DIR / "metrics"
PREDICTIONS_DIR= RESULTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 22050   # Hz  – librosa default; covers full speech band
N_FFT            = 2048    # FFT window size in samples
HOP_LENGTH       = 512     # Hop between frames (≈23 ms at 22 kHz)
TARGET_DURATION  = 3.0     # seconds – clips longer than this are trimmed/padded
N_MFCC           = 40      # Number of Mel-Frequency Cepstral Coefficients

# ---------------------------------------------------------------------------
# Acoustic feature extraction
# ---------------------------------------------------------------------------
# Statistical summaries computed per feature track
STAT_FUNCTIONS = ["mean", "std", "min", "max"]

# ---------------------------------------------------------------------------
# RAVDESS emotion mapping  (from filename position 3, 1-indexed)
# ---------------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Numeric labels for ML models
EMOTION_LABEL_MAP = {
    "neutral":   0,
    "calm":      1,
    "happy":     2,
    "sad":       3,
    "angry":     4,
    "fearful":   5,
    "disgust":   6,
    "surprised": 7,
}

# ---------------------------------------------------------------------------
# Speaker-independent split
# ---------------------------------------------------------------------------
# Actors 1–24; last N actors are reserved for testing
TRAIN_ACTORS = list(range(1, 21))   # Actor 01–20  → training
TEST_ACTORS  = list(range(21, 25))  # Actor 21–24  → testing

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
N_PCA_COMPONENTS     = 4    # Number of PCA components fed to quantum circuits
PCA_COMPONENTS_SWEEP = [4, 6, 8]  # Optional sweep for experiments

# ---------------------------------------------------------------------------
# Quantum configuration
# ---------------------------------------------------------------------------
N_QUBITS         = 4        # Must equal N_PCA_COMPONENTS when using ZZFeatureMap
FEATURE_MAP_REPS = 2        # ZZFeatureMap repetitions (entanglement depth)
ANSATZ_REPS      = 1        # EfficientSU2 repetitions

# ---------------------------------------------------------------------------
# VQC training
# ---------------------------------------------------------------------------
VQC_MAX_ITER  = 100         # Maximum SPSA iterations (keep low for CPU feasibility)
VQC_SHOTS     = None        # None → statevector simulation (exact); int → sampling
VQC_SEED      = 42

# ---------------------------------------------------------------------------
# QSVC
# ---------------------------------------------------------------------------
QSVC_SHOTS    = None        # None → statevector kernel

# ---------------------------------------------------------------------------
# Classical SVM
# ---------------------------------------------------------------------------
SVM_C         = 10.0
SVM_KERNEL    = "rbf"
SVM_GAMMA     = "scale"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------
# FAST_MODE: subsamples the dataset and uses fewer VQC iterations.
#            Useful for quick smoke-tests.
# FULL_MODE: uses the full dataset (still PCA-compressed for quantum models).
FAST_MODE             = "fast"
FULL_MODE             = "full"
DEFAULT_MODE          = FULL_MODE
FAST_MODE_SAMPLE_LIMIT = 200   # Max training samples in fast mode
FAST_VQC_MAX_ITER     = 30    # Max SPSA iterations in fast mode

# ---------------------------------------------------------------------------
# Whisper (optional speech-to-text branch)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large

# ---------------------------------------------------------------------------
# Multimodal fusion
# ---------------------------------------------------------------------------
FUSION_ALPHA = 0.5   # weight for acoustic branch: P_final = α·P_audio + (1-α)·P_text
