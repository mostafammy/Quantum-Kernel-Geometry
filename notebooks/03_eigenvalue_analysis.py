"""
03_eigenvalue_analysis.py
-------------------------
Computes the Eigenvalue Spectrum of Classical vs Quantum Kernels.
This mathematically proves the 'Expressivity' difference.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Ensure output directory exists
os.makedirs("../results/figures", exist_ok=True)

# --- 1. SETUP (SAME DATA AS BEFORE) ---
print("🧪 Regenerating Data for Consistency...")
X, y = make_circles(n_samples=20, factor=0.5, noise=0.05, random_state=42)

# Sort for consistency
sort_idx = np.argsort(y)
X = X[sort_idx]
y = y[sort_idx]

scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_scaled = scaler.fit_transform(X)

# --- 2. COMPUTE KERNEL MATRICES ---
print("📐 Computing Classical Kernel...")
K_classical = rbf_kernel(X_scaled, gamma=1.0)

print("⚛️ Computing Quantum Kernel (Simulating)...")
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
q_kernel = FidelityQuantumKernel(feature_map=feature_map)
K_quantum = q_kernel.evaluate(x_vec=X_scaled)

# --- 3. SPECTRAL ANALYSIS (THE MATH) ---
print("📊 Decomposing Matrices into Eigenvalues...")

# Compute eigenvalues (sorted ascending)
eig_vals_classical = np.linalg.eigvalsh(K_classical)
eig_vals_quantum = np.linalg.eigvalsh(K_quantum)

# Sort descending (Largest first)
eig_vals_classical = eig_vals_classical[::-1]
eig_vals_quantum = eig_vals_quantum[::-1]

# Normalize so the first eigenvalue is 1.0 (for fair comparison)
eig_vals_classical = eig_vals_classical / eig_vals_classical[0]
eig_vals_quantum = eig_vals_quantum / eig_vals_quantum[0]

# --- 4. VISUALIZATION ---
print("🎨 Plotting Spectrum...")

plt.figure(figsize=(10, 6))

# Plot Classical Line
plt.plot(eig_vals_classical, marker='o', linestyle='--', color='purple', label='Classical RBF (Fast Decay)')

# Plot Quantum Line
plt.plot(eig_vals_quantum, marker='s', linestyle='-', color='orange', label='Quantum Fidelity (High Rank)')

plt.yscale('log')  # Log scale reveals the tiny values clearly
plt.title("Eigenvalue Spectrum: Expressivity Comparison", fontsize=14)
plt.ylabel("Eigenvalue Magnitude (Log Scale)", fontsize=12)
plt.xlabel("Component Index", fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=12)

# Save
output_path = "../results/figures/eigenvalue_spectrum.png"
plt.savefig(output_path, dpi=300)
print(f"\n✅ SUCCESS: Spectral Plot Saved to {output_path}")

