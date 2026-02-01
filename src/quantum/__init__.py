"""
Quantum Kernel Methods Module.

This package provides quantum computing implementations for kernel-based
machine learning, enabling comparison with classical approaches.

Modules
-------
feature_maps
    Quantum feature map factories (ZZ, Z, Pauli).
fidelity_kernel
    Quantum kernel computation using state fidelity.

Key Concepts
------------
Quantum kernels exploit the exponential size of Hilbert space (2^n dimensions
for n qubits) to create feature spaces that may provide advantages over
classical kernels for certain problems.

Examples
--------
>>> from src.quantum import QuantumKernelComputer, create_zz_feature_map
>>> import numpy as np

>>> # Create and use a quantum kernel
>>> qkc = QuantumKernelComputer.from_zz_feature_map(num_features=4, reps=2)
>>> X = np.random.randn(10, 4)  # Scale this to [0, 2π] in practice
>>> K = qkc.compute_kernel_matrix(X)
>>> print(f"Kernel shape: {K.shape}")
"""

from src.quantum.feature_maps import (
    create_zz_feature_map,
    create_z_feature_map,
    create_pauli_feature_map,
    get_feature_map_info,
    EntanglementType
)
from src.quantum.fidelity_kernel import (
    QuantumKernelComputer,
    compute_quantum_kernel
)

__all__ = [
    # Feature Maps
    "create_zz_feature_map",
    "create_z_feature_map",
    "create_pauli_feature_map",
    "get_feature_map_info",
    "EntanglementType",
    # Kernel Computation
    "QuantumKernelComputer",
    "compute_quantum_kernel",
]
