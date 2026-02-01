"""
Classical Kernel Methods Module.

This package provides classical machine learning kernel implementations
for comparison with quantum kernel methods.

Modules
-------
rbf_kernel
    Radial Basis Function (Gaussian) kernel implementation.
spectral_analysis
    Eigenvalue spectrum analysis for kernel expressivity.

Examples
--------
>>> from src.classical import compute_rbf_kernel, compute_eigenspectrum
>>> import numpy as np
>>> X = np.random.randn(20, 4)
>>> K = compute_rbf_kernel(X, gamma=1.0)
>>> spectrum = compute_eigenspectrum(K)
>>> print(f"Effective dimension: {len(spectrum[spectrum > 0.01])}")
"""

from src.classical.rbf_kernel import compute_rbf_kernel, suggest_gamma
from src.classical.spectral_analysis import (
    compute_eigenspectrum,
    compute_effective_rank,
    compute_participation_ratio,
    compare_spectra
)

__all__ = [
    # RBF Kernel
    "compute_rbf_kernel",
    "suggest_gamma",
    # Spectral Analysis
    "compute_eigenspectrum",
    "compute_effective_rank",
    "compute_participation_ratio",
    "compare_spectra",
]
