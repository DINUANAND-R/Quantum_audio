"""
quantum_features.py — Quantum feature map construction for Qiskit 2.x.

HOW QUANTUM ENCODING WORKS
----------------------------
Classical PCA features (real numbers) → quantum circuit angles

A ZZFeatureMap encodes N classical values into N qubits by:
  1. Applying Hadamard gates to put qubits in superposition: |+⟩^⊗N
  2. Applying Pauli-Z rotations: Rz(2·x_i) on qubit i
  3. Applying entangling ZZ interactions: Rz(2·(π - x_i)(π - x_j)) on pairs
  4. Repeating (reps) times

This creates a quantum state whose inner product (quantum kernel) between
two data points is hard to compute classically for large N — the potential
source of quantum advantage for kernel-based ML.

For the QSVC, the kernel matrix K[i,j] = |⟨ψ(x_i)|ψ(x_j)⟩|² is computed
and used in a classical SVM.

For the VQC, a variational ansatz (EfficientSU2) adds trainable parameters
on top of the feature map.

API compatibility: Qiskit 2.x  (qiskit >= 2.0,  qiskit-machine-learning >= 0.8)
"""

from __future__ import annotations

import logging

from qiskit.circuit.library import EfficientSU2, ZZFeatureMap

from src.config import ANSATZ_REPS, FEATURE_MAP_REPS, N_QUBITS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def build_feature_map(
    n_qubits: int = N_QUBITS,
    reps: int = FEATURE_MAP_REPS,
) -> ZZFeatureMap:
    """
    Build a ZZFeatureMap.

    The ZZFeatureMap was introduced in Havlíček et al. (Nature 2019) as a
    feature map that is conjectured to be classically hard to simulate for
    large n_qubits.  For small n_qubits used here, it serves as a structured
    quantum encoding of classical features.

    Parameters
    ----------
    n_qubits : int  — number of qubits = number of PCA components
    reps     : int  — number of ZZ entanglement layers

    Returns
    -------
    ZZFeatureMap circuit
    """
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    logger.info(
        "Built ZZFeatureMap: n_qubits=%d, reps=%d, depth=%d",
        n_qubits, reps, fm.decompose().depth()
    )
    print(f"[quantum_features] ZZFeatureMap: {n_qubits} qubits, {reps} reps")
    return fm


# ---------------------------------------------------------------------------
# Ansatz (variational circuit for VQC)
# ---------------------------------------------------------------------------

def build_ansatz(
    n_qubits: int = N_QUBITS,
    reps: int = ANSATZ_REPS,
) -> EfficientSU2:
    """
    Build an EfficientSU2 variational ansatz.

    EfficientSU2 consists of:
      - Layers of single-qubit SU(2) rotations (Ry, Rz gates)
      - Linear entanglement via CNOT gates

    The trainable parameters (θ) in these gates are optimised by the
    classical optimizer (SPSA) to minimise the classification loss.

    Parameters
    ----------
    n_qubits : int
    reps     : int  — number of SU(2) layers

    Returns
    -------
    EfficientSU2 circuit
    """
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement="linear")
    n_params = ansatz.num_parameters
    logger.info(
        "Built EfficientSU2 ansatz: n_qubits=%d, reps=%d, n_params=%d",
        n_qubits, reps, n_params,
    )
    print(f"[quantum_features] EfficientSU2 ansatz: {n_qubits} qubits, "
          f"{reps} reps, {n_params} trainable parameters")
    return ansatz
