# ⚛️ Quantum Kernel Geometry: Inductive Bias & Expressivity Analysis

![Status](https://img.shields.io/badge/Status-Active_Research-success)
![Focus](https://img.shields.io/badge/Focus-Information_Geometry-blueviolet)
![Tech](https://img.shields.io/badge/Stack-Qiskit_|_Scikit--Learn-blue)

**Author:** Mostafa Yaser (Daphi)  
**Topic:** Quantum Machine Learning, Hilbert Space Geometry, Kernel Methods  

---

## 🧪 Abstract
This repository investigates the geometric properties of Quantum Kernel methods compared to classical Radial Basis Function (RBF) kernels. By analyzing the kernel matrices produced by mapping data into high-dimensional Hilbert spaces, we demonstrate two fundamental phenomena:
1.  **Geometric Divergence:** Quantum feature maps induce "Interference Patterns" that capture non-local structural similarities, unlike the distance-based locality of RBF.
2.  **The Qubit-Rank Relation:** We prove a direct causal link between qubit count and model expressivity, demonstrating a "Rank Bottleneck" at low qubits ($N=2$) and a "Rank Explosion" at higher qubits ($N=8$).

---

## 👁️ Visual Proof: The "Fog" vs. The "Crystal"

We compared a classical **RBF Kernel** against a **Quantum Fidelity Kernel** on a structured dataset.

* **Classical RBF (Left):** Exhibits a "diagonal blur." It relies on Euclidean distance, meaning similarity decays smoothly. It is effectively "myopic" (nearsighted).
* **Quantum Fidelity (Right):** Exhibits a sharp, checkerboard-like interference pattern. High similarity scores appear between distant data points, proving the model is capturing **entanglement-induced correlations** invisible to classical metrics.

![Fog vs Crystal Comparison](results/figures/kernel_comparison.png)
*(Figure 1: Side-by-side heatmap comparison of Kernel Matrices)*

---

## 📉 Mathematical Proof: Bottleneck & Explosion

Visuals are subjective; Eigenvalues are objective. We analyzed the **Eigenvalue Spectrum** of the kernel matrices to measure the Effective Dimension (Rank) of the feature space.

### Experiment A: The 2-Qubit Bottleneck
At $N=2$ qubits, the Hilbert space dimension is small ($2^2 = 4$).
* **Observation:** The Quantum Rank crashes early.
* **Conclusion:** Low-qubit quantum models act as **Dimensionality Compressors**. They force data into a constrained subspace, which can be useful for regularization but limits expressivity.

### Experiment B: The 8-Qubit Rank Explosion
At $N=8$ qubits, the Hilbert space dimension expands ($2^8 = 256$).
* **Observation:** The Quantum Eigenvalues stay higher for longer, matching and eventually surpassing the classical RBF tail.
* **Conclusion:** Increasing hardware scale unlocks massive expressivity, allowing the model to capture complex, high-dimensional manifolds.

![Rank Explosion](results/figures/high_dim_scaling.png)
*(Figure 2: Log-scale Eigenvalue Spectrum showing the Rank Explosion at 8 Qubits)*

---

## 📂 Repository Structure

| Module | Description |
| :--- | :--- |
| `src/` | Modular implementation of Quantum Feature Maps and Geometric Analysis tools. |
| `notebooks/02_...` | **Visual Analysis:** Generates the "Fog vs Crystal" heatmaps. |
| `notebooks/03_...` | **Bottleneck Analysis:** Demonstrates the rank limits of 2-qubit systems. |
| `notebooks/04_...` | **Scaling Analysis:** Demonstrates the rank explosion of 8-qubit systems. |
| `results/figures/` | Contains the generated proof plots and artifacts. |

---

## 🚀 Reproduction

To reproduce these scientific findings, clone the repository and run the experiments:

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run the Visual Comparison
python notebooks/02_kernel_comparison_script.py

# 3. Run the Scaling Experiment (Rank Explosion)
python notebooks/04_high_dim_scaling.py
