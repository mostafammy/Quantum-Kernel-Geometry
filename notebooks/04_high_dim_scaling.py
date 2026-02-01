"""
04_high_dim_scaling.py
----------------------
The "Scale-Up" Experiment.
We increase from 2 Qubits to 8 Qubits to observe Rank Explosion.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# --- PATH SETUP (Fixing the 'Lost File' Bug) ---
# This ensures we save to the right folder relative to where you run the script
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. GENERATE HIGH-DIMENSIONAL DATA ---
print("🧪 Generating 8-Dimensional Complexity...")
# We generate 40 samples with 8 features
# This requires more "room" to represent than 2D circles
n_features = 8
n_samples = 40

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=6,  # 6 features actually matter (Complex!)
    n_redundant=0,
    random_state=42
)

# Scale to (0, 2π) for Quantum Rotation
scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_scaled = scaler.fit_transform(X)

print(f" - Input Space: {n_features} Dimensions")
print(f" - Hilbert Space: 2^{n_features} = {2**n_features} Dimensions")

# --- 2. CLASSICAL RBF KERNEL ---
print("\n📐 Computing Classical RBF...")
# We use a specific gamma to prevent RBF from just being Identity
K_classical = rbf_kernel(X_scaled, gamma=0.5)

# --- 3. QUANTUM KERNEL (8 QUBITS) ---
print("\n⚛️ Computing Quantum Kernel (8 Qubits)...")
print("   (This might take 10-20 seconds on CPU...)")

# 8 Qubits matches the 8 Features
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2, entanglement='linear')
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Evaluate
K_quantum = q_kernel.evaluate(x_vec=X_scaled)

# --- 4. SPECTRAL ANALYSIS ---
print("\n📊 Calculating Eigenvalues...")

# Get Eigenvalues
eig_vals_classical = np.linalg.eigvalsh(K_classical)[::-1]
eig_vals_quantum = np.linalg.eigvalsh(K_quantum)[::-1]

# Normalize top eigenvalue to 1.0
eig_vals_classical = eig_vals_classical / eig_vals_classical[0]
eig_vals_quantum = eig_vals_quantum / eig_vals_quantum[0]

# --- 5. PLOTTING ---
print("🎨 Saving Comparison...")

plt.figure(figsize=(10, 6))

# Classical
plt.plot(eig_vals_classical, marker='o', linestyle='--', color='purple', label='Classical RBF')

# Quantum
plt.plot(eig_vals_quantum, marker='s', linestyle='-', color='orange', label=f'Quantum ({n_features} Qubits)')

plt.yscale('log')
plt.title(f"Rank Explosion: {n_features} Qubits vs Classical", fontsize=14)
plt.ylabel("Eigenvalue Magnitude (Log Scale)")
plt.xlabel("Component Index")
plt.legend()
plt.grid(True, which="both", alpha=0.5)

# Save correctly using the robust path
output_path = os.path.join(OUTPUT_DIR, "high_dim_scaling.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Done! Saved to: {output_path}")
