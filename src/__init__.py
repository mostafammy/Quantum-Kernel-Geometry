"""
Quantum Kernel Geometry Analysis Package.

A research package for analyzing and comparing the geometric properties
of classical vs quantum kernel methods in machine learning.

Subpackages
-----------
classical
    Classical kernel implementations (RBF) and spectral analysis.
quantum
    Quantum feature maps and fidelity kernel computation.
visualization
    Publication-quality plots for kernels and eigenvalue spectra.

Quick Start
-----------
>>> import numpy as np
>>> from sklearn.preprocessing import MinMaxScaler
>>> from src.classical import compute_rbf_kernel, compute_eigenspectrum
>>> from src.quantum import compute_quantum_kernel
>>> from src.visualization import plot_kernel_comparison, plot_spectrum_comparison

>>> # Generate data
>>> X = np.random.randn(20, 4)
>>> scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
>>> X_scaled = scaler.fit_transform(X)

>>> # Compute kernels
>>> K_rbf = compute_rbf_kernel(X_scaled, gamma=1.0)
>>> K_quantum = compute_quantum_kernel(X_scaled)

>>> # Visualize
>>> plot_kernel_comparison(K_rbf, K_quantum)

Author: Mostafa Yaser
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Mostafa Yaser"

# Convenience imports
from src.classical import compute_rbf_kernel, compute_eigenspectrum
from src.quantum import QuantumKernelComputer, compute_quantum_kernel
from src.visualization import plot_kernel_comparison, plot_spectrum_comparison
