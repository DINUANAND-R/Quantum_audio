"""
config.py
Central configuration for Quantum-Audio-Emotion.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "RAVDESS"

PROC_DATA_DIR = DATA_DIR / "processed"

FEATURES_DIR = PROJECT_ROOT / "features"

FEATURE_FILE = FEATURES_DIR / "audio_features.csv"

TRAIN_CSV = PROC_DATA_DIR / "train.csv"

TEST_CSV = PROC_DATA_DIR / "test.csv"

TRANSCRIPT_CSV = PROC_DATA_DIR / "transcripts.csv"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR = RESULTS_DIR / "metrics"

PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# ============================================================
# AUDIO PROCESSING
# ============================================================

SAMPLE_RATE = 22050

N_FFT = 2048

HOP_LENGTH = 512

TARGET_DURATION = 3.0

N_MFCC = 40


# ============================================================
# FEATURE EXTRACTION
# ============================================================

STAT_FUNCTIONS = [
    "mean",
    "std",
    "min",
    "max"
]


# ============================================================
# EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


EMOTION_LABEL_MAP = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}


EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]


# ============================================================
# SPEAKER-INDEPENDENT DATA SPLIT
# ============================================================

# Actors 1-20 -> Training
# Actors 21-24 -> Testing

TRAIN_ACTORS = list(range(1, 21))

TEST_ACTORS = list(range(21, 25))


# ============================================================
# PCA CONFIGURATION
# ============================================================

# Main PCA dimension used by quantum models.
#
# We use 8 because the quantum models below use 8 qubits.
N_PCA_COMPONENTS = 8


# PCA experiments to compare.
#
# This allows us to test:
# PCA-4
# PCA-6
# PCA-8
# PCA-12
# PCA-16

PCA_COMPONENTS_SWEEP = [
    4,
    6,
    8,
    12,
    16
]


# ============================================================
# QUANTUM CONFIGURATION
# ============================================================

# Number of qubits used by quantum models.
#
# PCA-8 -> 8 features -> 8 qubits

N_QUBITS = 8


# Quantum feature map
FEATURE_MAP_REPS = 1


# Variational ansatz
ANSATZ_REPS = 1


# ============================================================
# VQC CONFIGURATION
# ============================================================

# Maximum number of optimizer iterations
VQC_MAX_ITER = 150


# Faster configuration for development/testing
FAST_VQC_MAX_ITER = 30


# None means Statevector simulation
# instead of shot-based sampling.
VQC_SHOTS = None


# Reproducibility
VQC_SEED = 42


# ============================================================
# QSVC CONFIGURATION
# ============================================================

# None means exact/statevector-style simulation
# where supported by the selected quantum backend.
QSVC_SHOTS = None


# ============================================================
# CLASSICAL SVM CONFIGURATION
# ============================================================

SVM_C = 10.0

SVM_KERNEL = "rbf"

SVM_GAMMA = "scale"


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# EXECUTION MODES
# ============================================================

FAST_MODE = "fast"

FULL_MODE = "full"

DEFAULT_MODE = FULL_MODE


# Number of samples used in fast/development mode
FAST_MODE_SAMPLE_LIMIT = 200


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

WHISPER_MODEL_SIZE = "base"


# ============================================================
# MULTIMODAL FUSION
# ============================================================

# Weight used when combining acoustic and transcript features.
#
# Final fusion:
#
# fused_score =
#     FUSION_ALPHA * acoustic_score
#     +
#     (1 - FUSION_ALPHA) * text_score

FUSION_ALPHA = 0.5