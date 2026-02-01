"""
02_kernel_comparison_script.py
------------------------------
Compares Classical (RBF) vs Quantum (Fidelity) Kernel Geometries.
Saves output to results/figures/kernel_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Ensure output directory exists
os.makedirs("../results/figures", exist_ok=True)

# --- 1. SETUP & DATA GENERATION ---
print("🧪 Generating Synthetic Structured Data...")
# Concentric circles: Impossible to separate with a straight line (Linear Kernel)
# But easy for RBF and Quantum Kernels to map
X, y = make_circles(n_samples=20, factor=0.5, noise=0.05, random_state=42)

# Sort for cleaner visualization (Class 0 then Class 1)
sort_idx = np.argsort(y)
X = X[sort_idx]
y = y[sort_idx]

# Scale to (0, 2π) for Quantum Rotation
scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_scaled = scaler.fit_transform(X)

print(f" - Data Shape: {X_scaled.shape}")

# --- 2. CLASSICAL RBF KERNEL ---
print("\n📐 Computing Classical RBF Kernel...")
K_classical = rbf_kernel(X_scaled, gamma=1.0)

# --- 3. QUANTUM FIDELITY KERNEL ---
print("\n⚛️ Computing Quantum Fidelity Kernel...")
print(" - Configuring ZZFeatureMap (2 Qubits, Deep Entanglement)...")
# Reps=2 + Linear Entanglement = Global Correlations
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Evaluate Kernel (Run Simulation)
K_quantum = q_kernel.evaluate(x_vec=X_scaled)

# --- 4. VISUALIZATION ---
print("\n🎨 Generating Comparison Heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot Classical
sns.heatmap(K_classical, ax=axes[0], cmap='viridis', square=True, vmin=0, vmax=1)
axes[0].set_title("Classical RBF Kernel\n(Geometry = Local Distance)", fontsize=14)
axes[0].set_xlabel("Patient Index")
axes[0].set_ylabel("Patient Index")

# Plot Quantum
sns.heatmap(K_quantum, ax=axes[1], cmap='magma', square=True, vmin=0, vmax=1)
axes[1].set_title("Quantum Fidelity Kernel\n(Geometry = Hilbert Space Interference)", fontsize=14)
axes[1].set_xlabel("Patient Index")
axes[1].set_ylabel("Patient Index")

plt.tight_layout()

# Save

output_path = "../results/figures/kernel_comparison.png"
plt.savefig(output_path, dpi=300)
print(f"\n✅ SUCCESS: Visualization Saved to {output_path}")
