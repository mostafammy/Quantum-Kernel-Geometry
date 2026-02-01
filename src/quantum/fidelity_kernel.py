"""
Quantum Fidelity Kernel Module.

This module provides a high-level interface for computing quantum kernel
matrices using the fidelity (state overlap) between quantum states.

Mathematical Background:
-----------------------
The quantum kernel is defined as the fidelity between two quantum states:

    K(x, y) = |⟨Φ(x)|Φ(y)⟩|²

Where |Φ(x)⟩ is the quantum state produced by encoding classical data x
through a quantum feature map.

Why This Works:
--------------
1. The feature map Φ maps data to an exponentially large Hilbert space (2^n dimensions)
2. Fidelity measures "overlap" or similarity in this huge space
3. Quantum interference allows non-local correlations that classical kernels miss

The Measurement Process:
-----------------------
Computing K(x, y) = |⟨Φ(x)|Φ(y)⟩|² requires:
1. Prepare |Φ(x)⟩ using the feature map
2. Apply the inverse feature map Φ†(y)
3. Measure probability of |00...0⟩ state

On current (NISQ) hardware, this is estimated through repeated sampling.
On simulators, this can be computed exactly using statevector simulation.

Author: Mostafa Yaser
License: MIT
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.quantum.feature_maps import create_zz_feature_map, EntanglementType


class QuantumKernelComputer:
    """
    High-level interface for quantum kernel computation.

    This class wraps the Qiskit FidelityQuantumKernel with educational
    documentation and convenient factory methods.

    Attributes
    ----------
    feature_map : QuantumCircuit
        The quantum feature map circuit used for encoding.
    kernel : FidelityQuantumKernel
        The underlying Qiskit kernel object.
    
    Examples
    --------
    >>> from src.quantum.fidelity_kernel import QuantumKernelComputer
    >>> import numpy as np
    
    >>> # Create a quantum kernel for 4-dimensional data
    >>> qkc = QuantumKernelComputer.from_zz_feature_map(num_features=4)
    
    >>> # Compute kernel matrix
    >>> X = np.random.randn(20, 4)  # 20 samples, 4 features
    >>> K = qkc.compute_kernel_matrix(X)
    >>> print(K.shape)  # (20, 20)
    
    Notes
    -----
    Computational Complexity:
    - Time: O(n² × shots) where n = number of samples
    - The quantum circuit runs n² times (once per kernel entry)
    - Each run requires `shots` measurements on real hardware
    
    On simulators with statevector mode, this is computed exactly
    without statistical sampling noise.
    """

    def __init__(self, feature_map: QuantumCircuit):
        """
        Initialize a QuantumKernelComputer with a given feature map.

        Parameters
        ----------
        feature_map : QuantumCircuit
            The quantum circuit that encodes classical data into quantum states.
            Must have parameters that will be bound to input features.
        """
        self.feature_map = feature_map
        self.kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    @classmethod
    def from_zz_feature_map(
        cls,
        num_features: int,
        reps: int = 2,
        entanglement: EntanglementType = "linear"
    ) -> "QuantumKernelComputer":
        """
        Factory method to create a kernel computer with ZZ feature map.

        Parameters
        ----------
        num_features : int
            Number of input features (= number of qubits).
        reps : int, default=2
            Number of feature map repetitions.
        entanglement : str, default="linear"
            Entanglement pattern between qubits.

        Returns
        -------
        QuantumKernelComputer
            Configured quantum kernel computer.

        Examples
        --------
        >>> qkc = QuantumKernelComputer.from_zz_feature_map(
        ...     num_features=8,
        ...     reps=2,
        ...     entanglement="linear"
        ... )
        """
        feature_map = create_zz_feature_map(
            num_features=num_features,
            reps=reps,
            entanglement=entanglement
        )
        return cls(feature_map)
    
    def compute_kernel_matrix(
        self,
        X: NDArray[np.floating],
        Y: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """
        Compute the quantum kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First dataset. Features must match the feature map dimension.
            Values should typically be scaled to [0, 2π] for rotation gates.
        
        Y : ndarray of shape (n_samples_Y, n_features), optional
            Second dataset. If None, computes K(X, X).

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Quantum kernel matrix where K[i,j] = |⟨Φ(X[i])|Φ(Y[j])⟩|²

        Notes
        -----
        Properties of the quantum kernel matrix:
        1. Symmetric: K[i,j] = K[j,i] (when X = Y)
        2. Positive semi-definite: All eigenvalues ≥ 0
        3. Diagonal equals 1: K[i,i] = 1 (self-overlap is perfect)
        """
        if Y is None:
            return self.kernel.evaluate(x_vec=X)
        else:
            return self.kernel.evaluate(x_vec=X, y_vec=Y)
    
    def get_circuit_info(self) -> dict:
        """
        Get information about the underlying quantum circuit.

        Returns
        -------
        info : dict
            Circuit statistics including depth, gate counts, etc.
        """
        return {
            "num_qubits": self.feature_map.num_qubits,
            "depth": self.feature_map.depth(),
            "num_parameters": self.feature_map.num_parameters,
            "gate_counts": dict(self.feature_map.count_ops()),
            "hilbert_dim": 2 ** self.feature_map.num_qubits
        }


def compute_quantum_kernel(
    X: NDArray[np.floating],
    Y: Optional[NDArray[np.floating]] = None,
    num_qubits: Optional[int] = None,
    reps: int = 2,
    entanglement: EntanglementType = "linear"
) -> NDArray[np.floating]:
    """
    Convenience function to compute a quantum kernel matrix.

    This is a functional interface to QuantumKernelComputer for quick use.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data. Feature values should be in [0, 2π].
    
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second dataset. If None, computes K(X, X).
    
    num_qubits : int, optional
        Number of qubits to use. Defaults to n_features.
    
    reps : int, default=2
        Feature map repetitions.
    
    entanglement : str, default="linear"
        Entanglement pattern.

    Returns
    -------
    K : ndarray
        Quantum kernel matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from src.quantum.fidelity_kernel import compute_quantum_kernel
    
    >>> # Prepare data
    >>> X = np.random.randn(20, 4)
    >>> scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    >>> X_scaled = scaler.fit_transform(X)
    
    >>> # Compute kernel
    >>> K = compute_quantum_kernel(X_scaled)
    >>> print(f"Kernel shape: {K.shape}")
    """
    if num_qubits is None:
        num_qubits = X.shape[1]
    
    qkc = QuantumKernelComputer.from_zz_feature_map(
        num_features=num_qubits,
        reps=reps,
        entanglement=entanglement
    )
    
    return qkc.compute_kernel_matrix(X, Y)


# Module exports
__all__ = [
    "QuantumKernelComputer",
    "compute_quantum_kernel"
]
