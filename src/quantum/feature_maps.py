"""
Quantum Feature Maps Module.

This module provides factory functions for creating quantum feature maps
that encode classical data into quantum states. These feature maps are
the foundation of quantum kernel methods.

Mathematical Background:
-----------------------
A quantum feature map Φ: X → H maps classical data x ∈ X to a quantum state
|Φ(x)⟩ in Hilbert space H. The encoding is done through parameterized
quantum circuits where data values control rotation angles.

Common Encoding Strategies:
--------------------------
1. **Angle Encoding**: x → R_y(x) or R_z(x)
   - Maps each feature to a rotation angle
   - Simple but limited expressivity

2. **ZZ Feature Map**: Uses entangling gates between qubits
   - First layer: Single-qubit rotations parameterized by data
   - Second layer: Two-qubit ZZ interactions: exp(-i(π-x_i)(π-x_j)ZZ)
   - Creates correlations that capture non-linear relationships

Why Entanglement Matters:
------------------------
Without entanglement, each qubit evolves independently → separable states.
With entanglement, qubits influence each other → can capture complex 
correlations invisible to classical distance metrics (the "interference 
patterns" we observe in kernel heatmaps).

Author: Mostafa Yaser
License: MIT
"""

from typing import Literal, Optional
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit import QuantumCircuit


# Type alias for entanglement patterns
EntanglementType = Literal["linear", "full", "circular", "sca"]


def create_zz_feature_map(
    num_features: int,
    reps: int = 2,
    entanglement: EntanglementType = "linear"
) -> ZZFeatureMap:
    """
    Create a ZZ Feature Map for quantum kernel computation.

    The ZZ Feature Map is one of the most expressive quantum feature maps,
    using ZZ interactions to create entanglement-induced correlations.

    Parameters
    ----------
    num_features : int
        Number of input features (= number of qubits).
        Each classical feature is encoded into one qubit.
    
    reps : int, default=2
        Number of repetitions of the feature map circuit.
        More reps = deeper circuit = higher expressivity but harder to train.
        - reps=1: Shallow, quick to simulate
        - reps=2: Good balance (recommended)
        - reps=3+: Very expressive but may overfit
    
    entanglement : {"linear", "full", "circular", "sca"}, default="linear"
        Pattern of entangling gates between qubits:
        - "linear": Each qubit entangles with its neighbor (i ↔ i+1)
        - "full": All-to-all connectivity (exponential gates)
        - "circular": Linear + wrap-around (n ↔ 0)
        - "sca": Shifted-circular-alternating

    Returns
    -------
    feature_map : ZZFeatureMap
        Configured Qiskit ZZFeatureMap circuit.

    Examples
    --------
    >>> fm = create_zz_feature_map(num_features=4, reps=2, entanglement="linear")
    >>> print(f"Circuit depth: {fm.depth()}")
    >>> print(f"Qubits: {fm.num_qubits}")
    
    Notes
    -----
    Circuit structure for 2 qubits, reps=1:
    
        ┌───┐┌─────────────┐                              ┌───┐┌─────────────┐
    q0: ┤ H ├┤ P(2.0*x[0]) ├──■──────────────────────────┤ H ├┤ P(2.0*x[0]) ├
        ├───┤├─────────────┤┌─┴─┐┌────────────────────┐┌─┴───┴┴─────────────┤
    q1: ┤ H ├┤ P(2.0*x[1]) ├┤ X ├┤ P(2.0*(π-x[0])(π-x[1])) ├┤ X ├ ...
        └───┘└─────────────┘└───┘└────────────────────┘└───┘
    
    The ZZ interaction term (π-x[0])(π-x[1]) creates non-linear feature
    combinations that cannot be captured by classical linear methods.
    """
    return ZZFeatureMap(
        feature_dimension=num_features,
        reps=reps,
        entanglement=entanglement
    )


def create_z_feature_map(
    num_features: int,
    reps: int = 2
) -> ZFeatureMap:
    """
    Create a Z Feature Map (simpler, no entanglement).

    The Z Feature Map uses only single-qubit Z rotations without any
    entangling gates. This is less expressive than ZZ but faster to simulate.

    Parameters
    ----------
    num_features : int
        Number of input features (= number of qubits).
    
    reps : int, default=2
        Number of repetitions.

    Returns
    -------
    feature_map : ZFeatureMap
        Configured Qiskit ZFeatureMap circuit.

    Notes
    -----
    Without entanglement, this feature map produces separable states.
    The kernel matrix will be a product of single-qubit overlaps:
    
        K(x, y) = ∏ᵢ |⟨φᵢ(xᵢ)|φᵢ(yᵢ)⟩|²
    
    This limits expressivity but provides a good baseline for comparison.
    """
    return ZFeatureMap(
        feature_dimension=num_features,
        reps=reps
    )


def create_pauli_feature_map(
    num_features: int,
    reps: int = 2,
    paulis: Optional[list] = None,
    entanglement: EntanglementType = "linear"
) -> PauliFeatureMap:
    """
    Create a general Pauli Feature Map with configurable Pauli gates.

    The Pauli Feature Map generalizes ZZ by allowing any combination
    of Pauli operators (X, Y, Z) in the encoding.

    Parameters
    ----------
    num_features : int
        Number of input features (= number of qubits).
    
    reps : int, default=2
        Number of repetitions.
    
    paulis : list of str, optional
        Pauli strings to use, e.g., ["Z", "ZZ", "ZZZ"].
        Default: ["Z", "ZZ"] (equivalent to ZZFeatureMap).
    
    entanglement : str, default="linear"
        Entanglement pattern (same options as ZZFeatureMap).

    Returns
    -------
    feature_map : PauliFeatureMap
        Configured Qiskit PauliFeatureMap circuit.

    Examples
    --------
    >>> # Create a feature map with Z, ZZ, and YY interactions
    >>> fm = create_pauli_feature_map(4, paulis=["Z", "ZZ", "YY"])
    """
    if paulis is None:
        paulis = ["Z", "ZZ"]
    
    return PauliFeatureMap(
        feature_dimension=num_features,
        reps=reps,
        paulis=paulis,
        entanglement=entanglement
    )


def get_feature_map_info(feature_map: QuantumCircuit) -> dict:
    """
    Extract useful information about a feature map circuit.

    Parameters
    ----------
    feature_map : QuantumCircuit
        A Qiskit quantum circuit (feature map).

    Returns
    -------
    info : dict
        Dictionary containing:
        - num_qubits: Number of qubits
        - depth: Circuit depth
        - num_parameters: Number of data parameters
        - gate_counts: Count of each gate type
    """
    return {
        "num_qubits": feature_map.num_qubits,
        "depth": feature_map.depth(),
        "num_parameters": feature_map.num_parameters,
        "gate_counts": dict(feature_map.count_ops())
    }


# Module exports
__all__ = [
    "create_zz_feature_map",
    "create_z_feature_map", 
    "create_pauli_feature_map",
    "get_feature_map_info",
    "EntanglementType"
]
