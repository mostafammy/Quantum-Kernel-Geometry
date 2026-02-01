"""
Classical RBF (Radial Basis Function) Kernel Module.

This module provides an educational wrapper around scikit-learn's RBF kernel
implementation, with detailed documentation explaining the mathematical
foundations and geometric interpretations.

Mathematical Background:
-----------------------
The RBF kernel (also known as Gaussian kernel) is defined as:

    K(x, y) = exp(-γ ||x - y||²)

Where:
    - x, y are input vectors
    - γ (gamma) controls the kernel width
    - ||x - y||² is the squared Euclidean distance

Geometric Interpretation:
------------------------
The RBF kernel measures similarity based on LOCAL DISTANCE. Points that are
close in Euclidean space have high similarity (→ 1), while distant points
have low similarity (→ 0). This creates "myopic" or nearsighted behavior -
the kernel only sees local neighborhoods.

Author: Mostafa Yaser
License: MIT
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel


def compute_rbf_kernel(
    X: NDArray[np.floating],
    Y: Optional[NDArray[np.floating]] = None,
    gamma: float = 1.0
) -> NDArray[np.floating]:
    """
    Compute the RBF (Gaussian) kernel matrix between samples.

    This function wraps sklearn's rbf_kernel with educational documentation
    and type hints for clarity.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        Feature matrix for the first set of samples.
        Each row represents one data point.
    
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Feature matrix for the second set of samples.
        If None, computes K(X, X) - the kernel matrix of X with itself.
    
    gamma : float, default=1.0
        Kernel coefficient. Controls the "width" of the Gaussian:
        - Small γ (e.g., 0.1): Wide kernel, more points considered "similar"
        - Large γ (e.g., 10): Narrow kernel, only very close points similar
        
        Rule of thumb: γ = 1 / (2σ²) where σ is the RBF bandwidth.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix where K[i, j] = exp(-γ ||X[i] - Y[j]||²)
        Values range from 0 (dissimilar) to 1 (identical).

    Examples
    --------
    >>> import numpy as np
    >>> from src.classical.rbf_kernel import compute_rbf_kernel
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> K = compute_rbf_kernel(X, gamma=1.0)
    >>> K.shape
    (3, 3)
    >>> np.allclose(np.diag(K), 1.0)  # Self-similarity is always 1
    True

    Notes
    -----
    The RBF kernel is a Mercer kernel (positive semi-definite), meaning:
    1. K is symmetric: K(x, y) = K(y, x)
    2. All eigenvalues are ≥ 0
    3. It implicitly maps data to an infinite-dimensional Hilbert space

    See Also
    --------
    sklearn.metrics.pairwise.rbf_kernel : The underlying implementation
    """
    return sklearn_rbf_kernel(X, Y, gamma=gamma)


def suggest_gamma(X: NDArray[np.floating], method: str = "median") -> float:
    """
    Suggest an appropriate gamma value based on the data distribution.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data to analyze.
    
    method : str, default="median"
        Method for computing gamma:
        - "median": γ = 1 / median(||x_i - x_j||²) (robust to outliers)
        - "mean": γ = 1 / mean(||x_i - x_j||²) (sensitive to outliers)
        - "sklearn": γ = 1 / n_features (sklearn's default)

    Returns
    -------
    gamma : float
        Suggested gamma value for the RBF kernel.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> gamma = suggest_gamma(X, method="median")
    >>> print(f"Suggested gamma: {gamma:.4f}")
    """
    if method == "sklearn":
        return 1.0 / X.shape[1]
    
    # Compute pairwise squared distances
    from sklearn.metrics.pairwise import euclidean_distances
    distances_sq = euclidean_distances(X, squared=True)
    
    # Get upper triangle (exclude diagonal and duplicates)
    upper_tri = distances_sq[np.triu_indices_from(distances_sq, k=1)]
    
    if method == "median":
        return 1.0 / np.median(upper_tri)
    elif method == "mean":
        return 1.0 / np.mean(upper_tri)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median', 'mean', or 'sklearn'.")


# Module exports
__all__ = ["compute_rbf_kernel", "suggest_gamma"]
