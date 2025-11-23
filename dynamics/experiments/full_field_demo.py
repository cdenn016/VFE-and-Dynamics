#!/usr/bin/env python3
"""
Complete Field Theory Demo
===========================

Demonstrates the FULL dynamical theory with all three kinetic terms:

    L = T_μ + T_Σ + T_φ - V

This is your complete LaTeX theory running!

Usage:
    python examples/full_field_demo.py

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dynamics.field_theory import (
    FullFieldHamiltonian,
    FullFieldState,
    create_simple_potential,
    create_initial_state,
    field_energy_summary
)
from dynamics.product_manifold_integrator import ProductManifoldVerlet
from geometry.lie_algebra import LieGroup


def main():
    print("\n" + "="*70)
    print("COMPLETE FIELD THEORY DEMONSTRATION")
    print("="*70)
    print("\nRunning the FULL dynamical theory:")
    print("  L = T_μ + T_Σ + T_φ - V")
    print("\nAll THREE kinetic terms with proper geometric structures!")
    print("="*70)

    # Parameters
    K = 3  # Latent dimension
    gauge_group = LieGroup.SO3

    # Create target (equilibrium point)
    mu_target = np.array([1.0, -0.5, 0.3])
    A = np.random.randn(K, K)
    Sigma_target = A @ A.T + 2 * np.eye(K)

    # Potential (with weaker coupling for stability)
    potential = create_simple_potential(mu_target, Sigma_target, weights=(0.5, 0.05, 0.01))

    # Hamiltonian
    Sigma_prior = 3 * np.eye(K)
    hamiltonian = FullFieldHamiltonian(
        potential=potential,
        Sigma_prior=Sigma_prior,
        gauge_group=gauge_group
    )

    # Initial state - start closer to equilibrium
    state = create_initial_state(K, gauge_group)
    # Move closer to target
    state.mu = mu_target + np.random.randn(K) * 0.3
    state.Sigma = Sigma_target + np.random.randn(K, K) * 0.1
    state.Sigma = 0.5 * (state.Sigma + state.Sigma.T)  # Symmetrize
    # Ensure SPD
    eigvals, eigvecs = np.linalg.eigh(state.Sigma)
    eigvals = np.maximum(eigvals, 0.5)
    state.Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Give it small initial momentum for stability
    state.pi_mu = np.random.randn(K) * 0.05
    state.pi_Sigma = np.random.randn(K, K) * 0.005
    state.pi_Sigma = 0.5 * (state.pi_Sigma + state.pi_Sigma.T)
    state.pi_phi = np.random.randn(3) * 0.02

    print(f"\nInitial state:")
    print(f"  μ = {state.mu}")
    print(f"  Σ diagonal = {np.diag(state.Sigma)}")
    print(f"  φ = {state.phi}")

    field_energy_summary(hamiltonian, state)

    # Integrate using GEOMETRIC INTEGRATOR on product manifold
    print("\n" + "="*70)
    print("Using ProductManifoldVerlet (symplectic on ℝ^K × SPD(K) × 𝔰𝔬(3))")
    print("="*70)

    integrator = ProductManifoldVerlet(hamiltonian)
    history = integrator.integrate(state, t_end=10.0, dt=0.005, save_interval=20)

    # Extract final state
    state = history['final_state']

    print(f"\nFinal state:")
    print(f"  μ = {state.mu}")
    print(f"  Σ diagonal = {np.diag(state.Sigma)}")
    print(f"  φ = {state.phi}")

    field_energy_summary(hamiltonian, state)

    # Energy drift
    E0 = history['energy'][0]
    E_final = history['energy'][-1]
    drift = E_final - E0

    print(f"\nEnergy conservation:")
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Final energy:   {E_final:.6f}")
    print(f"  Drift:          {drift:+.6e}")
    print(f"  Relative drift: {100*drift/E0:.2f}%")

    # Plot results
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    t = history['t']
    mu = history['mu']
    Sigma = history['Sigma']
    phi = history['phi']

    # Row 1: Mean evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(K):
        ax1.plot(t, mu[:, i], label=f'μ_{i}', alpha=0.7)
        ax1.axhline(mu_target[i], color=f'C{i}', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('μ')
    ax1.set_title('Mean Evolution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Covariance diagonal evolution
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(K):
        Sigma_diag = np.array([S[i, i] for S in Sigma])
        ax2.plot(t, Sigma_diag, label=f'Σ_{i}{i}', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Σ_ii')
    ax2.set_title('Covariance Diagonal')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Gauge field evolution
    ax3 = fig.add_subplot(gs[0, 2])
    n_gauge = phi.shape[1]
    for i in range(n_gauge):
        ax3.plot(t, phi[:, i], label=f'φ_{i}', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('φ')
    ax3.set_title('Gauge Field')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Row 2: Energy components
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(t, history['energy'], 'b-', linewidth=2, label='Total H')
    ax4.axhline(E0, color='r', linestyle='--', label='Initial')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy')
    ax4.set_title('Total Energy (Should be Conserved)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Row 3: Phase space projections
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(mu[:, 0], mu[:, 1], 'b-', alpha=0.7)
    ax5.plot(mu[0, 0], mu[0, 1], 'go', markersize=10, label='Start')
    ax5.plot(mu[-1, 0], mu[-1, 1], 'ro', markersize=10, label='End')
    ax5.plot(mu_target[0], mu_target[1], 'k*', markersize=15, label='Target')
    ax5.set_xlabel('μ₀')
    ax5.set_ylabel('μ₁')
    ax5.set_title('Mean Phase Space')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Covariance trace evolution
    ax6 = fig.add_subplot(gs[2, 1])
    Sigma_trace = np.array([np.trace(S) for S in Sigma])
    ax6.plot(t, Sigma_trace, 'b-', linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('tr(Σ)')
    ax6.set_title('Covariance Trace')
    ax6.grid(alpha=0.3)

    # Gauge field norm
    ax7 = fig.add_subplot(gs[2, 2])
    phi_norm = np.linalg.norm(phi, axis=1)
    ax7.plot(t, phi_norm, 'b-', linewidth=2)
    ax7.set_xlabel('Time')
    ax7.set_ylabel('||φ||')
    ax7.set_title('Gauge Field Norm')
    ax7.grid(alpha=0.3)

    plt.suptitle('Complete Field Theory: T_μ + T_Σ + T_φ', fontsize=16, y=0.995)

    out_path = Path("_results/full_field_theory.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved {out_path}")

    print("\n" + "="*70)
    print("COMPLETE! Your full LaTeX theory is now running!")
    print("="*70)
    print("\nAll THREE geometric structures:")
    print("  ✓ Fisher metric (mean)")
    print("  ✓ SPD manifold (covariance)")
    print("  ✓ Lie algebra (gauge field)")
    print("\nThis is the COMPLETE dynamical theory!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()