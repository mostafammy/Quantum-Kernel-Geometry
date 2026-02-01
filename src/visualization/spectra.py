"""
Eigenvalue Spectrum Visualization Module.

This module provides functions for visualizing eigenvalue spectra,
which reveal the "expressivity" or effective dimensionality of
kernel-induced feature spaces.

Key Visualization:
-----------------
Log-scale eigenvalue plots show:
- Fast decay = compressed representation (low expressivity)
- Slow decay = high-dimensional representation (high expressivity)

The "crossover point" where quantum exceeds classical demonstrates
the advantage of quantum feature spaces.

Author: Mostafa Yaser
License: MIT
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure


def plot_eigenvalue_spectrum(
    eigenvalues: NDArray[np.floating],
    ax: Optional[plt.Axes] = None,
    label: str = "Eigenvalues",
    color: str = "blue",
    marker: str = "o",
    linestyle: str = "-",
    log_scale: bool = True
) -> plt.Axes:
    """
    Plot a single eigenvalue spectrum.

    Parameters
    ----------
    eigenvalues : ndarray of shape (n,)
        Eigenvalues, ideally sorted in descending order and normalized.
    
    ax : matplotlib Axes, optional
        Axes to plot on. Creates new figure if None.
    
    label : str
        Legend label for this spectrum.
    
    color : str
        Line/marker color.
    
    marker : str
        Marker style (e.g., 'o', 's', '^').
    
    linestyle : str
        Line style (e.g., '-', '--', '-.').
    
    log_scale : bool, default=True
        Whether to use log scale on y-axis (recommended for seeing
        the full dynamic range).

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.arange(len(eigenvalues))
    
    ax.plot(
        indices,
        eigenvalues,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=label,
        markersize=6
    )
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.set_xlabel("Component Index", fontsize=12)
    ax.set_ylabel("Eigenvalue Magnitude" + (" (Log Scale)" if log_scale else ""), fontsize=12)
    ax.grid(True, which="both", alpha=0.5, linestyle="-")
    ax.legend(fontsize=12)
    
    return ax


def plot_spectrum_comparison(
    spectrum_classical: NDArray[np.floating],
    spectrum_quantum: NDArray[np.floating],
    classical_label: str = "Classical RBF (Fast Decay)",
    quantum_label: str = "Quantum Fidelity (High Rank)",
    title: str = "Eigenvalue Spectrum: Expressivity Comparison",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> matplotlib.figure.Figure:
    """
    Plot classical vs quantum eigenvalue spectra on the same axes.

    This visualization demonstrates the mathematical proof of quantum
    expressivity advantage: quantum eigenvalues decay slower, indicating
    higher effective rank.

    Parameters
    ----------
    spectrum_classical : ndarray
        Classical kernel eigenvalues (normalized, sorted descending).
    
    spectrum_quantum : ndarray
        Quantum kernel eigenvalues (normalized, sorted descending).
    
    classical_label, quantum_label : str
        Legend labels.
    
    title : str
        Figure title.
    
    figsize : tuple
        Figure size in inches.
    
    save_path : str, optional
        If provided, saves figure to this path.
    
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    >>> from src.classical import compute_rbf_kernel, compute_eigenspectrum
    >>> from src.quantum import compute_quantum_kernel
    >>> 
    >>> K_rbf = compute_rbf_kernel(X_scaled)
    >>> K_quantum = compute_quantum_kernel(X_scaled)
    >>> 
    >>> spec_rbf = compute_eigenspectrum(K_rbf)
    >>> spec_quantum = compute_eigenspectrum(K_quantum)
    >>> 
    >>> fig = plot_spectrum_comparison(spec_rbf, spec_quantum)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Classical spectrum (purple, dashed)
    plot_eigenvalue_spectrum(
        spectrum_classical,
        ax=ax,
        label=classical_label,
        color="purple",
        marker="o",
        linestyle="--"
    )
    
    # Quantum spectrum (orange, solid)
    plot_eigenvalue_spectrum(
        spectrum_quantum,
        ax=ax,
        label=quantum_label,
        color="orange",
        marker="s",
        linestyle="-"
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Figure saved to: {save_path}")
    
    return fig


def plot_multi_spectrum_comparison(
    spectra: Dict[str, NDArray[np.floating]],
    colors: Optional[Dict[str, str]] = None,
    title: str = "Multi-Kernel Eigenvalue Comparison",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Plot multiple eigenvalue spectra for comparing many kernels.

    Parameters
    ----------
    spectra : dict
        Dictionary mapping kernel name to eigenvalue array.
        Example: {"RBF": spec_rbf, "2 Qubits": spec_2q, "8 Qubits": spec_8q}
    
    colors : dict, optional
        Dictionary mapping kernel name to color.
    
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
    if colors is None:
        # Default color cycle
        default_colors = plt.cm.tab10.colors
        colors = {name: default_colors[i % 10] for i, name in enumerate(spectra.keys())}
    
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, spec) in enumerate(spectra.items()):
        ax.plot(
            np.arange(len(spec)),
            spec,
            marker=markers[i % len(markers)],
            linestyle="-" if "Quantum" in name or "qubit" in name.lower() else "--",
            color=colors.get(name, f"C{i}"),
            label=name,
            markersize=5,
            alpha=0.8
        )
    
    ax.set_yscale("log")
    ax.set_xlabel("Component Index", fontsize=12)
    ax.set_ylabel("Eigenvalue Magnitude (Log Scale)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_rank_vs_qubits(
    qubit_counts: List[int],
    effective_ranks: List[int],
    classical_rank: Optional[int] = None,
    title: str = "Effective Rank vs Qubit Count",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Plot how effective rank scales with qubit count.

    This demonstrates the "Rank Explosion" phenomenon as qubits increase.

    Parameters
    ----------
    qubit_counts : list of int
        Number of qubits tested.
    
    effective_ranks : list of int
        Corresponding effective ranks.
    
    classical_rank : int, optional
        If provided, adds a horizontal dashed line for classical comparison.
    
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
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        qubit_counts,
        effective_ranks,
        marker="s",
        linestyle="-",
        color="orange",
        linewidth=2,
        markersize=10,
        label="Quantum Kernel"
    )
    
    if classical_rank is not None:
        ax.axhline(
            y=classical_rank,
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"Classical RBF (rank={classical_rank})"
        )
    
    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    
    # Annotate Hilbert space dimension
    for q, r in zip(qubit_counts, effective_ranks):
        ax.annotate(
            f"2^{q}={2**q}",
            (q, r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            alpha=0.7
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


# Module exports
__all__ = [
    "plot_eigenvalue_spectrum",
    "plot_spectrum_comparison",
    "plot_multi_spectrum_comparison",
    "plot_rank_vs_qubits"
]
