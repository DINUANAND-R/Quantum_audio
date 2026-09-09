"""
main.py

Command-line entry point for Quantum-Audio-Emotion.

Pipeline:

    Dataset
       ↓
    Feature Extraction
       ↓
    Speaker-Independent Split
       ↓
    Classical SVM
       ↓
    PCA
       ↓
    QSVC
       ↓
    VQC
       ↓
    Evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

from src.config import (
    DEFAULT_MODE,
    FAST_MODE,
    FAST_MODE_SAMPLE_LIMIT,
    FAST_VQC_MAX_ITER,
    FULL_MODE,
    N_PCA_COMPONENTS,
    N_QUBITS,
    RAW_DATA_DIR,
    RANDOM_STATE,
    VQC_MAX_ITER,
    VQC_SEED,
    VQC_SHOTS,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)-8s "
        "%(name)s: "
        "%(message)s"
    ),
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")


# ============================================================
# DATASET
# ============================================================

def stage_dataset(args):
    """
    Verify that the RAVDESS dataset is available.
    """

    from src.dataset import verify_dataset

    print("\n[main] === DATASET STAGE ===")

    verify_dataset(RAW_DATA_DIR)


# ============================================================
# BALANCED SAMPLING
# ============================================================

def _balanced_sample(
    df: pd.DataFrame,
    samples_per_class: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create a balanced subset while preserving all classes.

    This is used only for FAST/development mode.

    The final FULL experiment should use the complete
    speaker-independent training and test sets.
    """

    if samples_per_class <= 0:
        raise ValueError(
            "samples_per_class must be greater than zero."
        )

    if "emotion" not in df.columns:
        raise KeyError(
            "'emotion' column is required for balanced sampling."
        )

    class_counts = df["emotion"].value_counts()

    insufficient = class_counts[
        class_counts < samples_per_class
    ]

    if len(insufficient) > 0:
        raise ValueError(
            "Not enough samples for balanced sampling.\n"
            f"Required per class: {samples_per_class}\n"
            f"Available:\n{class_counts}"
        )

    sampled_parts = []

    for emotion in sorted(
        df["emotion"].unique()
    ):

        class_df = df[
            df["emotion"] == emotion
        ]

        sampled = class_df.sample(
            n=samples_per_class,
            random_state=random_state,
        )

        sampled_parts.append(
            sampled
        )

    result = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    # Shuffle final result
    result = result.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    return result


# ============================================================
# FEATURES
# ============================================================

def stage_features(args):
    """
    Extract acoustic features and create the
    speaker-independent train/test split.

    IMPORTANT:
        Actors 1-20  → training
        Actors 21-24 → testing

    The test speakers are never used for training.
    """

    from src.dataset import (
        load_ravdess_metadata,
    )

    from src.dataset_split import (
        save_splits,
        speaker_independent_split,
        verify_split,
    )

    from src.feature_extraction import (
        extract_features_for_dataset,
    )

    from src.visualization import (
        run_visualization_stage,
    )

    print(
        "\n[main] === FEATURE EXTRACTION STAGE ==="
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_df = load_ravdess_metadata(
        RAW_DATA_DIR
    )

    print(
        f"[main] Metadata samples: "
        f"{len(metadata_df)}"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature_df = extract_features_for_dataset(
        metadata_df,
        force_recompute=args.force_recompute,
    )

    if "actor_id" not in feature_df.columns:
        raise KeyError(
            "actor_id column missing from feature dataframe."
        )

    if "emotion" not in feature_df.columns:
        raise KeyError(
            "emotion column missing from feature dataframe."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # First create the speaker-independent split.
    #
    # Do NOT sample the complete dataframe before splitting,
    # because that can accidentally remove important speakers.
    # --------------------------------------------------------

    train_df, test_df = speaker_independent_split(
        feature_df
    )

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    verify_split(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # FAST MODE
    #
    # Only reduce TRAINING data.
    #
    # NEVER reduce the final test set.
    # --------------------------------------------------------

    if args.mode == FAST_MODE:

        print(
            "\n[main] FAST MODE enabled."
        )

        print(
            "[main] Reducing training data only."
        )

        # Use a balanced subset.
        #
        # Example:
        # 25 samples × 8 emotions = 200 samples.
        #

        class_count = (
            train_df["emotion"]
            .nunique()
        )

        samples_per_class = max(
            1,
            FAST_MODE_SAMPLE_LIMIT
            // class_count,
        )

        max_available = (
            train_df["emotion"]
            .value_counts()
            .min()
        )

        samples_per_class = min(
            samples_per_class,
            int(max_available),
        )

        train_df = _balanced_sample(
            train_df,
            samples_per_class=samples_per_class,
            random_state=RANDOM_STATE,
        )

        print(
            f"[main] FAST training samples: "
            f"{len(train_df)}"
        )

        print(
            f"[main] Test samples retained: "
            f"{len(test_df)}"
        )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    save_splits(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    try:

        # Visualize complete feature dataframe.
        run_visualization_stage(
            feature_df
        )

    except Exception as exc:

        logger.warning(
            "Visualization skipped: %s",
            exc,
        )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print(
        "\n[main] Speaker-independent split ready."
    )

    print(
        f"[main] Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"[main] Test samples: "
        f"{len(test_df)}"
    )

    print(
        "\n[main] Training emotion distribution:"
    )

    print(
        train_df["emotion"]
        .value_counts()
        .sort_index()
    )

    print(
        "\n[main] Test emotion distribution:"
    )

    print(
        test_df["emotion"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# CLASSICAL SVM
# ============================================================

def stage_classical(args):
    """
    Train and evaluate the classical SVM baseline.
    """

    from src.classical_model import (
        run_classical_stage,
    )

    from src.dataset_split import (
        load_splits,
    )

    print(
        "\n[main] === CLASSICAL SVM STAGE ==="
    )

    train_df, test_df = load_splits()

    metrics = run_classical_stage(
        train_df,
        test_df,
    )

    _print_stage_summary(
        "Classical SVM",
        metrics,
    )


# ============================================================
# PCA
# ============================================================

def stage_pca(args):
    """
    Run PCA using only training data for fitting.

    Test data is transformed using the already fitted
    scaler and PCA.
    """

    from src.dataset_split import (
        load_splits,
    )

    from src.dimensionality_reduction import (
        run_pca_stage,
    )

    print(
        "\n[main] === PCA STAGE ==="
    )

    n_components = (
        args.n_components
        if args.n_components is not None
        else N_PCA_COMPONENTS
    )

    train_df, test_df = load_splits()

    X_train, y_train, X_test, y_test, _, _ = (
        run_pca_stage(
            train_df,
            test_df,
            n_components=n_components,
        )
    )

    print(
        f"[main] PCA components: "
        f"{n_components}"
    )

    print(
        f"[main] X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"[main] X_test shape: "
        f"{X_test.shape}"
    )


# ============================================================
# QSVC
# ============================================================

def stage_qsvc(args):
    """
    Train and evaluate QSVC.

    Important:
        The complete held-out test set is used.

    Only training data may be reduced in FAST mode.
    """

    from src.dataset_split import (
        load_splits,
    )

    from src.dimensionality_reduction import (
        run_pca_stage,
    )

    from src.qsvc_model import (
        run_qsvc_stage,
    )

    print(
        "\n[main] === QSVC STAGE ==="
    )

    n_components = (
        args.n_components
        if args.n_components is not None
        else N_PCA_COMPONENTS
    )

    n_qubits = (
        args.n_qubits
        if args.n_qubits is not None
        else N_QUBITS
    )

    # --------------------------------------------------------
    # QSVC requires:
    #
    # number of PCA components == number of qubits
    # --------------------------------------------------------

    if n_components != n_qubits:

        raise ValueError(
            "\nQSVC configuration error.\n"
            f"PCA components = {n_components}\n"
            f"Qubits          = {n_qubits}\n\n"
            "For the current QSVC implementation:\n"
            "PCA components must equal number of qubits."
        )

    # --------------------------------------------------------
    # Load speaker-independent split
    # --------------------------------------------------------

    train_df, test_df = load_splits()

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
        _,
    ) = run_pca_stage(
        train_df,
        test_df,
        n_components=n_components,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    _print_quantum_config(
        n_components=n_components,
        n_qubits=n_qubits,
    )

    print(
        f"[main] QSVC training samples: "
        f"{len(X_train)}"
    )

    print(
        f"[main] QSVC test samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Train + evaluate
    # --------------------------------------------------------

    metrics = run_qsvc_stage(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_qubits=n_qubits,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    _print_stage_summary(
        "QSVC",
        metrics,
    )


# ============================================================
# VQC
# ============================================================

def stage_vqc(args):
    """
    Train and evaluate VQC.

    The final evaluation always uses the complete
    held-out test set.
    """

    from src.dataset_split import (
        load_splits,
    )

    from src.dimensionality_reduction import (
        run_pca_stage,
    )

    from src.vqc_model import (
        run_vqc_stage,
    )

    print(
        "\n[main] === VQC STAGE ==="
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    n_components = (
        args.n_components
        if args.n_components is not None
        else N_PCA_COMPONENTS
    )

    n_qubits = (
        args.n_qubits
        if args.n_qubits is not None
        else N_QUBITS
    )

    max_iter = (
        args.max_iter
        if args.max_iter is not None
        else (
            FAST_VQC_MAX_ITER
            if args.mode == FAST_MODE
            else VQC_MAX_ITER
        )
    )

    # --------------------------------------------------------
    # VQC dimension check
    # --------------------------------------------------------

    if n_components != n_qubits:

        raise ValueError(
            "\nVQC configuration error.\n"
            f"PCA components = {n_components}\n"
            f"Qubits          = {n_qubits}\n\n"
            "For the current VQC implementation:\n"
            "PCA components must equal number of qubits."
        )

    # --------------------------------------------------------
    # Load speaker-independent split
    # --------------------------------------------------------

    train_df, test_df = load_splits()

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
        _,
    ) = run_pca_stage(
        train_df,
        test_df,
        n_components=n_components,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    _print_quantum_config(
        n_components=n_components,
        n_qubits=n_qubits,
        max_iter=max_iter,
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Do NOT randomly select 100 test samples.
    #
    # The complete 240-sample held-out test set must be used
    # for the final VQC evaluation.
    # --------------------------------------------------------

    print(
        f"[main] VQC training samples: "
        f"{len(X_train)}"
    )

    print(
        f"[main] VQC test samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Train + evaluate
    # --------------------------------------------------------

    metrics = run_vqc_stage(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_qubits=n_qubits,
        max_iter=max_iter,
        random_state=VQC_SEED,
        shots=VQC_SHOTS,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    _print_stage_summary(
        "VQC",
        metrics,
    )


# ============================================================
# EVALUATION
# ============================================================

def stage_evaluate(args):
    """
    Compare saved model results.
    """

    from src.evaluation import (
        run_evaluation_stage,
    )

    print(
        "\n[main] === EVALUATION STAGE ==="
    )

    run_evaluation_stage()


# ============================================================
# WHISPER
# ============================================================

def stage_whisper(args):
    """
    Optional Whisper transcription stage.
    """

    from src.dataset import (
        load_ravdess_metadata,
    )

    from src.whisper_transcription import (
        transcribe_dataset,
    )

    print(
        "\n[main] === WHISPER STAGE ==="
    )

    metadata_df = load_ravdess_metadata(
        RAW_DATA_DIR
    )

    transcribe_dataset(
        metadata_df,
        force_recompute=args.force_recompute,
    )


# ============================================================
# ALL
# ============================================================

def stage_all(args):
    """
    Run the complete pipeline.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        " QUANTUM-AUDIO-EMOTION"
    )

    print(
        " FULL PIPELINE"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    stage_dataset(args)

    # --------------------------------------------------------
    # 2. Features + split
    # --------------------------------------------------------

    stage_features(args)

    # --------------------------------------------------------
    # 3. Classical baseline
    # --------------------------------------------------------

    stage_classical(args)

    # --------------------------------------------------------
    # 4. PCA
    # --------------------------------------------------------

    stage_pca(args)

    # --------------------------------------------------------
    # 5. QSVC
    # --------------------------------------------------------

    stage_qsvc(args)

    # --------------------------------------------------------
    # 6. VQC
    # --------------------------------------------------------

    stage_vqc(args)

    # --------------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------------

    stage_evaluate(args)

    print(
        "\n"
        + "=" * 60
    )

    print(
        " PIPELINE COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# HELPERS
# ============================================================

def _print_quantum_config(
    n_components: int,
    n_qubits: int,
    max_iter: int | None = None,
):
    """
    Print quantum experiment configuration.
    """

    print(
        "\n[main] Quantum configuration:"
    )

    print(
        f"  PCA components : "
        f"{n_components}"
    )

    print(
        f"  Qubits         : "
        f"{n_qubits}"
    )

    if max_iter is not None:

        print(
            f"  VQC iterations : "
            f"{max_iter}"
        )


def _print_stage_summary(
    name: str,
    metrics: dict,
):
    """
    Print common model summary.
    """

    print(
        f"\n[main] {name} complete"
    )

    print(
        f"  Accuracy : "
        f"{metrics.get('accuracy', 'N/A')}"
    )

    print(
        f"  Macro F1 : "
        f"{metrics.get('macro_f1', 'N/A')}"
    )

    print(
        f"  Weighted F1 : "
        f"{metrics.get('weighted_f1', 'N/A')}"
    )

    if "n_train" in metrics:

        print(
            f"  Train samples : "
            f"{metrics.get('n_train')}"
        )

    if "n_test" in metrics:

        print(
            f"  Test samples : "
            f"{metrics.get('n_test')}"
        )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser():
    """
    Build command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Quantum-Classical "
            "Speech Emotion Recognition"
        )
    )

    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    parser.add_argument(
        "--stage",
        choices=[
            "dataset",
            "features",
            "classical",
            "pca",
            "qsvc",
            "vqc",
            "evaluate",
            "whisper",
            "all",
        ],
        default="dataset",
        help="Pipeline stage to execute.",
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            FAST_MODE,
            FULL_MODE,
        ],
        default=DEFAULT_MODE,
        help=(
            "fast = reduced training data; "
            "full = complete training data."
        ),
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            "Number of PCA components. "
            "For quantum models this must equal "
            "the number of qubits."
        ),
    )

    # --------------------------------------------------------
    # Qubits
    # --------------------------------------------------------

    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Number of quantum circuit qubits.",
    )

    # --------------------------------------------------------
    # VQC iterations
    # --------------------------------------------------------

    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum VQC SPSA iterations.",
    )

    # --------------------------------------------------------
    # Feature recomputation
    # --------------------------------------------------------

    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Force feature/transcription "
            "recomputation."
        ),
    )

    # --------------------------------------------------------
    # Verbose
    # --------------------------------------------------------

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main CLI entry point.
    """

    parser = build_parser()

    args = parser.parse_args()

    # --------------------------------------------------------
    # Verbose logging
    # --------------------------------------------------------

    if args.verbose:

        logging.getLogger().setLevel(
            logging.DEBUG
        )

    # --------------------------------------------------------
    # Global random seed
    # --------------------------------------------------------

    np.random.seed(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Stage mapping
    # --------------------------------------------------------

    stages = {

        "dataset":
            stage_dataset,

        "features":
            stage_features,

        "classical":
            stage_classical,

        "pca":
            stage_pca,

        "qsvc":
            stage_qsvc,

        "vqc":
            stage_vqc,

        "evaluate":
            stage_evaluate,

        "whisper":
            stage_whisper,

        "all":
            stage_all,
    }

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    try:

        stages[
            args.stage
        ](args)

    except KeyboardInterrupt:

        print(
            "\n[main] Pipeline interrupted."
        )

        sys.exit(0)

    except Exception as exc:

        logger.exception(
            "Pipeline failed: %s",
            exc,
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()