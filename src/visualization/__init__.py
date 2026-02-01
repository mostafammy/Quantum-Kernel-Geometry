"""
Visualization Module for Quantum Kernel Analysis.

This package provides publication-quality plotting functions for
visualizing kernel matrices and eigenvalue spectra.

Modules
-------
heatmaps
    Kernel matrix heatmap visualizations.
spectra
    Eigenvalue spectrum plots.

Examples
--------
>>> from src.visualization import plot_kernel_comparison, plot_spectrum_comparison
>>> 
>>> # Compare kernel heatmaps
>>> fig1 = plot_kernel_comparison(K_classical, K_quantum)
>>> 
>>> # Compare eigenvalue spectra
>>> fig2 = plot_spectrum_comparison(spec_classical, spec_quantum)
>>> 
>>> plt.show()
"""

from src.visualization.heatmaps import (
    plot_kernel_heatmap,
    plot_kernel_comparison,
    plot_kernel_difference
)
from src.visualization.spectra import (
    plot_eigenvalue_spectrum,
    plot_spectrum_comparison,
    plot_multi_spectrum_comparison,
    plot_rank_vs_qubits
)

__all__ = [
    # Heatmaps
    "plot_kernel_heatmap",
    "plot_kernel_comparison",
    "plot_kernel_difference",
    # Spectra
    "plot_eigenvalue_spectrum",
    "plot_spectrum_comparison",
    "plot_multi_spectrum_comparison",
    "plot_rank_vs_qubits",
]
