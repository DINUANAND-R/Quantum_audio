"""
main.py — Command-line entry point for the Quantum-Audio-Emotion pipeline.

Usage
-----
  python main.py --stage dataset
  python main.py --stage features
  python main.py --stage classical
  python main.py --stage pca
  python main.py --stage qsvc
  python main.py --stage vqc
  python main.py --stage evaluate
  python main.py --stage whisper          # optional STT branch
  python main.py --stage all              # run complete pipeline

Mode flags
----------
  python main.py --stage all --mode fast  # subsample + fewer iterations
  python main.py --stage all --mode full  # full dataset (default)

Other flags
-----------
  --n-components 4     Override PCA components
  --n-qubits 4         Override number of qubits
  --max-iter 100       Override VQC max iterations
  --force-recompute    Re-extract features even if cache exists
  --verbose            Enable DEBUG logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running from any subdirectory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

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

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_dataset(args) -> None:
    """Stage 1: Verify RAVDESS dataset."""
    from src.dataset import verify_dataset
    verify_dataset(RAW_DATA_DIR)


def stage_features(args) -> None:
    """Stage 2–4: Extract acoustic features + speaker-independent split."""
    from src.dataset import load_ravdess_metadata
    from src.dataset_split import save_splits, speaker_independent_split, verify_split
    from src.feature_extraction import extract_features_for_dataset
    from src.visualization import run_visualization_stage

    print("\n[main] === FEATURE EXTRACTION STAGE ===")
    metadata_df = load_ravdess_metadata(RAW_DATA_DIR)
    feature_df  = extract_features_for_dataset(
        metadata_df, force_recompute=args.force_recompute
    )

    # Apply fast-mode subsampling if requested
    if args.mode == FAST_MODE:
        n = FAST_MODE_SAMPLE_LIMIT
        print(f"\n[main] FAST MODE: sampling {n} training rows from full dataset.")
        train_actors_df = feature_df[feature_df["actor_id"] <= 20]
        test_actors_df  = feature_df[feature_df["actor_id"] >  20]
        n_train = min(n, len(train_actors_df))
        train_actors_df = train_actors_df.sample(n=n_train, random_state=RANDOM_STATE)
        feature_df = pd.concat([train_actors_df, test_actors_df])

    train_df, test_df = speaker_independent_split(feature_df)
    verify_split(train_df, test_df)
    save_splits(train_df, test_df)

    run_visualization_stage(feature_df)
    _print_dataset_summary(train_df, test_df, args)


def stage_classical(args) -> None:
    """Stage 7: Train and evaluate Classical SVM."""
    import pandas as pd
    from src.classical_model import run_classical_stage
    from src.dataset_split import load_splits

    print("\n[main] === CLASSICAL SVM STAGE ===")
    train_df, test_df = load_splits()
    metrics = run_classical_stage(train_df, test_df)
    _print_stage_summary("Classical SVM", metrics)


def stage_pca(args) -> None:
    """Stage 8: Fit PCA on training data and transform both splits."""
    import pandas as pd
    from src.dataset_split import load_splits
    from src.dimensionality_reduction import run_pca_stage

    print("\n[main] === PCA STAGE ===")
    n_comp = args.n_components or N_PCA_COMPONENTS
    train_df, test_df = load_splits()
    X_train, y_train, X_test, y_test, scaler, pca = run_pca_stage(
        train_df, test_df, n_components=n_comp
    )
    print(f"[main] PCA complete.  "
          f"X_train: {X_train.shape},  X_test: {X_test.shape}")


def stage_qsvc(args) -> None:
    """Stage 10: Train and evaluate QSVC."""
    import pandas as pd
    from src.dataset_split import load_splits
    from src.dimensionality_reduction import (fit_scaler_pca, load_scaler_pca,
                                               save_scaler_pca, transform,
                                               get_labels, run_pca_stage)
    from src.qsvc_model import run_qsvc_stage

    print("\n[main] === QSVC STAGE ===")
    n_comp    = args.n_components or N_PCA_COMPONENTS
    n_qubits  = args.n_qubits     or N_QUBITS
    train_df, test_df = load_splits()

    X_train, y_train, X_test, y_test, _, _ = run_pca_stage(
        train_df, test_df, n_components=n_comp
    )

    _print_quantum_config(args, n_comp, n_qubits, mode=args.mode)

    metrics = run_qsvc_stage(X_train, y_train, X_test, y_test, n_qubits=n_qubits)
    _print_stage_summary("QSVC", metrics)


def stage_vqc(args) -> None:
    """Stage 11: Train and evaluate VQC."""
    import pandas as pd
    from src.dataset_split import load_splits
    from src.dimensionality_reduction import run_pca_stage
    from src.vqc_model import run_vqc_stage

    print("\n[main] === VQC STAGE ===")
    n_comp   = args.n_components or N_PCA_COMPONENTS
    n_qubits = args.n_qubits     or N_QUBITS
    max_iter = args.max_iter     or (FAST_VQC_MAX_ITER if args.mode == FAST_MODE
                                     else VQC_MAX_ITER)

    train_df, test_df = load_splits()
    X_train, y_train, X_test, y_test, _, _ = run_pca_stage(
        train_df, test_df, n_components=n_comp
    )

    _print_quantum_config(args, n_comp, n_qubits, mode=args.mode, max_iter=max_iter)

    metrics = run_vqc_stage(
        X_train, y_train, X_test, y_test,
        n_qubits=n_qubits, max_iter=max_iter,
        random_state=VQC_SEED, shots=VQC_SHOTS,
    )
    _print_stage_summary("VQC", metrics)


def stage_evaluate(args) -> None:
    """Stage 12: Compare all models."""
    print("\n[main] === EVALUATION / COMPARISON STAGE ===")
    from src.evaluation import run_evaluation_stage
    run_evaluation_stage()


def stage_whisper(args) -> None:
    """Stage 13 (optional): Transcribe dataset with Whisper."""
    print("\n[main] === WHISPER STT STAGE (OPTIONAL) ===")
    from src.dataset import load_ravdess_metadata
    from src.whisper_transcription import transcribe_dataset

    metadata_df = load_ravdess_metadata(RAW_DATA_DIR)
    transcribe_dataset(metadata_df, force_recompute=args.force_recompute)


def stage_all(args) -> None:
    """Run the complete pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  QUANTUM-AUDIO-EMOTION — FULL PIPELINE")
    print("=" * 60)
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  PCA Comp : {args.n_components or N_PCA_COMPONENTS}")
    print(f"  Qubits   : {args.n_qubits     or N_QUBITS}")
    print()

    stage_dataset(args)
    stage_features(args)
    stage_classical(args)
    stage_pca(args)
    stage_qsvc(args)
    stage_vqc(args)
    stage_evaluate(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_dataset_summary(train_df, test_df, args) -> None:
    print(f"\n[main] Dataset summary:")
    print(f"  Training samples : {len(train_df)}")
    print(f"  Test samples     : {len(test_df)}")
    print(f"  Feature columns  : {sum(1 for c in train_df.columns if c.startswith('feat_'))}")


def _print_quantum_config(args, n_comp, n_qubits, mode, max_iter=None) -> None:
    print(f"\n[main] Quantum configuration:")
    print(f"  Mode            : {mode.upper()}")
    print(f"  PCA components  : {n_comp}")
    print(f"  Qubits          : {n_qubits}")
    if max_iter is not None:
        print(f"  VQC max_iter    : {max_iter}")


def _print_stage_summary(name: str, metrics: dict) -> None:
    print(f"\n[main] {name} complete:")
    print(f"  Accuracy  : {metrics.get('accuracy',  'N/A')}")
    print(f"  Macro F1  : {metrics.get('macro_f1',  'N/A')}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum_audio_emotion",
        description="Hybrid Quantum-Classical Speech Emotion Recognition (RAVDESS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage dataset
  python main.py --stage features
  python main.py --stage classical
  python main.py --stage pca
  python main.py --stage qsvc
  python main.py --stage vqc
  python main.py --stage evaluate
  python main.py --stage all --mode fast
  python main.py --stage all --mode full
  python main.py --stage whisper
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["dataset", "features", "classical", "pca",
                 "qsvc", "vqc", "evaluate", "whisper", "all"],
        default="dataset",
        help="Pipeline stage to run (default: dataset)",
    )
    parser.add_argument(
        "--mode",
        choices=[FAST_MODE, FULL_MODE],
        default=DEFAULT_MODE,
        help=f"Execution mode: '{FAST_MODE}' subsamples data; "
             f"'{FULL_MODE}' uses everything (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=f"Number of PCA components (default: {N_PCA_COMPONENTS})",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help=f"Number of qubits (default: {N_QUBITS})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"VQC SPSA max iterations (default: {VQC_MAX_ITER})",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Re-extract features even if a cached file exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    np.random.seed(RANDOM_STATE)

    # Dispatch to the correct stage
    stages = {
        "dataset"  : stage_dataset,
        "features" : stage_features,
        "classical": stage_classical,
        "pca"      : stage_pca,
        "qsvc"     : stage_qsvc,
        "vqc"      : stage_vqc,
        "evaluate" : stage_evaluate,
        "whisper"  : stage_whisper,
        "all"      : stage_all,
    }

    runner = stages[args.stage]
    try:
        runner(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unhandled exception in stage '%s': %s", args.stage, exc)
        sys.exit(1)


if __name__ == "__main__":
    # pandas imported here to avoid circular import risk from inside functions
    import pandas as pd   # noqa: F401
    main()
