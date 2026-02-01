"""
Heatmap Visualization Module for Kernel Matrices.

This module provides publication-quality heatmap visualizations for
kernel matrices, enabling side-by-side comparison of classical vs
quantum kernels.

Visualization Philosophy:
------------------------
Good scientific visualization should:
1. Be immediately interpretable (colorbar, labels, title)
2. Enable fair comparison (consistent scales, colormaps)
3. Highlight the key finding (interference patterns vs diagonal blur)

Author: Mostafa Yaser
License: MIT
"""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns


def plot_kernel_heatmap(
    K: NDArray[np.floating],
    ax: Optional[plt.Axes] = None,
    title: str = "Kernel Matrix",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    xlabel: str = "Sample Index",
    ylabel: str = "Sample Index",
    show_colorbar: bool = True,
    annot: bool = False
) -> plt.Axes:
    """
    Plot a single kernel matrix as a heatmap.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric kernel matrix with values typically in [0, 1].
    
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    
    title : str, default="Kernel Matrix"
        Title for the heatmap.
    
    cmap : str, default="viridis"
        Colormap to use. Recommended:
        - "viridis": Classical kernels (smooth gradients)
        - "magma": Quantum kernels (highlights interference)
        - "coolwarm": For difference matrices (diverging)
    
    vmin, vmax : float, default=0.0, 1.0
        Color scale limits. Use consistent values for comparison.
    
    xlabel, ylabel : str
        Axis labels.
    
    show_colorbar : bool, default=True
        Whether to show the color scale bar.
    
    annot : bool, default=False
        Whether to annotate cells with values (for small matrices).

    Returns
    -------
    ax : matplotlib Axes
        The axes containing the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.random.rand(10, 10)
    >>> K = (K + K.T) / 2  # Make symmetric
    >>> ax = plot_kernel_heatmap(K, title="Example Kernel")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(
        K,
        ax=ax,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
        annot=annot,
        fmt=".2f" if annot else None
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    return ax


def plot_kernel_comparison(
    K_classical: NDArray[np.floating],
    K_quantum: NDArray[np.floating],
    classical_title: str = "Classical RBF Kernel\n(Geometry = Local Distance)",
    quantum_title: str = "Quantum Fidelity Kernel\n(Geometry = Hilbert Space Interference)",
    figsize: Tuple[float, float] = (16, 7),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> matplotlib.figure.Figure:
    """
    Create a side-by-side comparison of classical and quantum kernel matrices.

    This is the flagship visualization showing the "Fog vs Crystal" phenomenon:
    - Classical RBF shows diagonal blur (local distance dominates)
    - Quantum shows interference patterns (non-local correlations)

    Parameters
    ----------
    K_classical : ndarray of shape (n, n)
        Classical kernel matrix (e.g., RBF).
    
    K_quantum : ndarray of shape (n, n)
        Quantum kernel matrix (e.g., Fidelity).
    
    classical_title, quantum_title : str
        Titles for each subplot.
    
    figsize : tuple, default=(16, 7)
        Figure size in inches.
    
    save_path : str, optional
        If provided, saves the figure to this path.
    
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib Figure
        The comparison figure.

    Examples
    --------
    >>> from src.classical import compute_rbf_kernel
    >>> from src.quantum import compute_quantum_kernel
    >>> 
    >>> K_rbf = compute_rbf_kernel(X_scaled)
    >>> K_quantum = compute_quantum_kernel(X_scaled)
    >>> fig = plot_kernel_comparison(K_rbf, K_quantum)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Classical kernel (viridis - smooth transitions)
    plot_kernel_heatmap(
        K_classical,
        ax=axes[0],
        title=classical_title,
        cmap="viridis",
        xlabel="Sample Index",
        ylabel="Sample Index"
    )
    
    # Quantum kernel (magma - highlights interference patterns)
    plot_kernel_heatmap(
        K_quantum,
        ax=axes[1],
        title=quantum_title,
        cmap="magma",
        xlabel="Sample Index",
        ylabel="Sample Index"
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Figure saved to: {save_path}")
    
    return fig


def plot_kernel_difference(
    K1: NDArray[np.floating],
    K2: NDArray[np.floating],
    title: str = "Kernel Difference (K2 - K1)",
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Plot the element-wise difference between two kernel matrices.

    Useful for understanding WHERE the kernels disagree most.

    Parameters
    ----------
    K1, K2 : ndarray of shape (n, n)
        Kernel matrices to compare.
    
    title : str
        Figure title.
    
    figsize : tuple
        Figure size.
    
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    diff = K2 - K1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 0
    max_abs = max(abs(diff.min()), abs(diff.max()))
    
    sns.heatmap(
        diff,
        ax=ax,
        cmap="coolwarm",
        square=True,
        vmin=-max_abs,
        vmax=max_abs,
        center=0
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Sample Index", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


# Module exports
__all__ = [
    "plot_kernel_heatmap",
    "plot_kernel_comparison",
    "plot_kernel_difference"
]
