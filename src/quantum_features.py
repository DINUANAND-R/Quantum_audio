"""
quantum_features.py

Quantum feature map and variational ansatz.
"""

from __future__ import annotations

from qiskit.circuit.library import (
    EfficientSU2,
    ZZFeatureMap,
)


def build_feature_map(
    n_qubits: int,
    reps: int = 1,
):

    if n_qubits <= 0:
        raise ValueError(
            "n_qubits must be positive."
        )

    if reps <= 0:
        raise ValueError(
            "reps must be positive."
        )

    return ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement="linear",
    )


def build_ansatz(
    n_qubits: int,
    reps: int = 2,
):

    if n_qubits <= 0:
        raise ValueError(
            "n_qubits must be positive."
        )

    if reps <= 0:
        raise ValueError(
            "reps must be positive."
        )

    return EfficientSU2(
        num_qubits=n_qubits,
        reps=reps,
        entanglement="linear",
    )