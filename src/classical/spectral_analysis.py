"""
Spectral Analysis Module for Kernel Matrices.

This module provides utilities for analyzing the eigenvalue spectrum of
kernel matrices, which reveals the "expressivity" or "effective dimension"
of the feature space induced by the kernel.

Mathematical Background:
-----------------------
Given a kernel matrix K, its eigenvalue decomposition is:

    K = VΛV^T

Where:
    - V contains eigenvectors (orthonormal basis of the feature space)
    - Λ is diagonal with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0

Key Insight:
-----------
The eigenvalue spectrum reveals how "spread out" the data is in feature space:
- Fast decay (λᵢ → 0 quickly): Low effective dimension, compressed representation
- Slow decay (λᵢ stays high): High effective dimension, expressive representation

This is the mathematical foundation for comparing Classical vs Quantum kernels.

Author: Mostafa Yaser
License: MIT
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def compute_eigenspectrum(
    K: NDArray[np.floating],
    normalize: bool = True
) -> NDArray[np.floating]:
    """
    Compute the sorted eigenvalue spectrum of a kernel matrix.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric positive semi-definite kernel matrix.
    
    normalize : bool, default=True
        If True, normalize so the largest eigenvalue equals 1.0.
        This enables fair comparison between different kernels.

    Returns
    -------
    eigenvalues : ndarray of shape (n,)
        Eigenvalues sorted in descending order (largest first).
        If normalized, eigenvalues[0] = 1.0.

    Examples
    --------
    >>> from src.classical.rbf_kernel import compute_rbf_kernel
    >>> from src.classical.spectral_analysis import compute_eigenspectrum
    >>> import numpy as np
    >>> X = np.random.randn(20, 5)
    >>> K = compute_rbf_kernel(X)
    >>> spectrum = compute_eigenspectrum(K)
    >>> spectrum[0]  # Largest eigenvalue (normalized)
    1.0

    Notes
    -----
    For valid kernel matrices (Mercer kernels), all eigenvalues should be ≥ 0.
    Small negative values may appear due to numerical precision; these are
    typically clipped to 0 in downstream analysis.
    """
    # Use eigvalsh for symmetric matrices (more numerically stable)
    eigenvalues = np.linalg.eigvalsh(K)
    
    # Sort in descending order (largest first)
    eigenvalues = eigenvalues[::-1]
    
    if normalize and eigenvalues[0] > 0:
        eigenvalues = eigenvalues / eigenvalues[0]
    
    return eigenvalues


def compute_effective_rank(
    eigenvalues: NDArray[np.floating],
    threshold: float = 0.01
) -> int:
    """
    Compute the effective rank (numerical rank) of a kernel matrix.

    The effective rank counts how many eigenvalues are "significant",
    which measures the dimensionality of the space actually being used.

    Parameters
    ----------
    eigenvalues : ndarray of shape (n,)
        Eigenvalues, assumed to be normalized (max = 1.0) and sorted descending.
    
    threshold : float, default=0.01
        Eigenvalues below this fraction of the maximum are considered negligible.
        - threshold=0.01: Count eigenvalues > 1% of max
        - threshold=0.001: More permissive (higher rank)

    Returns
    -------
    effective_rank : int
        Number of eigenvalues above the threshold.

    Examples
    --------
    >>> eigenvalues = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
    >>> compute_effective_rank(eigenvalues, threshold=0.01)
    4
    >>> compute_effective_rank(eigenvalues, threshold=0.1)
    3

    Notes
    -----
    This is related to but different from:
    - Matrix rank (count of non-zero eigenvalues)
    - Nuclear norm (sum of eigenvalues)
    - Participation ratio (sum² / sum of squares)
    """
    max_eigenvalue = np.max(eigenvalues)
    if max_eigenvalue == 0:
        return 0
    
    normalized = eigenvalues / max_eigenvalue
    return int(np.sum(normalized > threshold))


def compute_participation_ratio(eigenvalues: NDArray[np.floating]) -> float:
    """
    Compute the participation ratio (PR) of the eigenvalue spectrum.

    The participation ratio is a continuous measure of effective dimensionality,
    defined as:

        PR = (Σλᵢ)² / Σλᵢ²

    Properties:
    - PR = 1 when only one eigenvalue is non-zero (1D)
    - PR = n when all n eigenvalues are equal (maximum dimension)

    Parameters
    ----------
    eigenvalues : ndarray of shape (n,)
        Non-negative eigenvalues (not necessarily normalized).

    Returns
    -------
    pr : float
        Participation ratio, ranging from 1 to n.

    Examples
    --------
    >>> # Uniform spectrum (maximum PR)
    >>> uniform = np.ones(10) / 10
    >>> compute_participation_ratio(uniform)
    10.0
    >>> # Single dominant eigenvalue (minimum PR)
    >>> peaked = np.array([1.0, 0.0, 0.0, 0.0])
    >>> compute_participation_ratio(peaked)
    1.0
    """
    eigenvalues = np.abs(eigenvalues)  # Handle small negative numerical errors
    sum_sq = np.sum(eigenvalues) ** 2
    sq_sum = np.sum(eigenvalues ** 2)
    
    if sq_sum == 0:
        return 0.0
    
    return sum_sq / sq_sum


def compare_spectra(
    spectrum_a: NDArray[np.floating],
    spectrum_b: NDArray[np.floating],
    name_a: str = "A",
    name_b: str = "B"
) -> dict:
    """
    Compare two eigenvalue spectra and return summary statistics.

    Parameters
    ----------
    spectrum_a, spectrum_b : ndarray
        Eigenvalue spectra to compare (should be normalized).
    name_a, name_b : str
        Names for the spectra (for the output dictionary).

    Returns
    -------
    comparison : dict
        Dictionary containing:
        - effective_rank_a, effective_rank_b: Effective ranks
        - participation_ratio_a, participation_ratio_b: PR values
        - rank_ratio: effective_rank_b / effective_rank_a
        - crossover_index: Where spectrum B exceeds spectrum A (if any)
    """
    eff_rank_a = compute_effective_rank(spectrum_a)
    eff_rank_b = compute_effective_rank(spectrum_b)
    pr_a = compute_participation_ratio(spectrum_a)
    pr_b = compute_participation_ratio(spectrum_b)
    
    # Find crossover point (where B > A)
    min_len = min(len(spectrum_a), len(spectrum_b))
    crossover = None
    for i in range(min_len):
        if spectrum_b[i] > spectrum_a[i]:
            crossover = i
            break
    
    return {
        f"effective_rank_{name_a.lower()}": eff_rank_a,
        f"effective_rank_{name_b.lower()}": eff_rank_b,
        f"participation_ratio_{name_a.lower()}": pr_a,
        f"participation_ratio_{name_b.lower()}": pr_b,
        "rank_ratio": eff_rank_b / eff_rank_a if eff_rank_a > 0 else float('inf'),
        "crossover_index": crossover
    }


# Module exports
__all__ = [
    "compute_eigenspectrum",
    "compute_effective_rank", 
    "compute_participation_ratio",
    "compare_spectra"
]
