# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:55:35 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Dynamics Demo: Hamiltonian & Lagrangian Belief Evolution
=========================================================

Demonstrates the full second-order dynamics framework for belief evolution.

Key Insight:
-----------
Fisher metric (∇²KL) provides natural kinetic terms:

    KL ≈ const + force·δθ + (1/2)δθ^T·MASS·δθ
                            \_________________/
                            kinetic energy!

This gives:
    Hamiltonian: H = (1/2)p^T M^{-1} p + V(θ)
    Lagrangian: L = (1/2)θ̇^T M θ̇ - V(θ)

Demos:
------
1. Harmonic oscillator (analytical solution for validation)
2. Free energy potential with Fisher metric
3. Integrator comparison (energy conservation)
4. Phase space portraits
5. Overdamped vs underdamped regimes

Usage:
------
    python examples/dynamics_demo.py --demo all
    python examples/dynamics_demo.py --demo harmonic
    python examples/dynamics_demo.py --demo comparison

Author: Chris
Date: November 2025
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dynamics import (
    BeliefHamiltonian,
    BeliefLagrangian,
    Verlet,
    PEFRL,
    compare_integrators,
    overdamped_limit
)


def harmonic_oscillator_demo():
    """
    Demo 1: Harmonic oscillator (exact solution for validation).

    H = (1/2)(p² + q²)
    Solution: q(t) = A cos(ωt + φ), p(t) = -Aω sin(ωt + φ)
    """
    print("\n" + "="*70)
    print("DEMO 1: HARMONIC OSCILLATOR")
    print("="*70)

    # Define Hamiltonian
    def potential(q):
        return 0.5 * np.sum(q**2)

    def metric(q):
        return np.eye(len(q))

    H = BeliefHamiltonian(potential, metric)

    # Initial conditions
    q0 = np.array([1.0, 0.0])
    p0 = np.array([0.0, 1.0])

    print(f"Initial energy: {H.total_energy(q0, p0):.6f}")
    print(f"Initial position: {q0}")
    print(f"Initial momentum: {p0}")

    # Integrate
    integrator = Verlet(H)
    t, q, p, stats = integrator.integrate(
        q0, p0,
        t_span=(0.0, 20.0),
        dt=0.01,
        save_interval=10
    )

    print(f"\nAfter {stats.steps} steps:")
    print(f"Energy drift: {stats.energy_drift:+.6e}")
    print(f"Energy std: {stats.energy_std:.6e}")

    # Analytical solution
    omega = 1.0
    A = np.sqrt(q0[0]**2 + p0[0]**2)
    phi = np.arctan2(p0[0], q0[0])
    q_exact = A * np.cos(omega * t + phi)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory
    ax = axes[0, 0]
    ax.plot(q[:, 0], q[:, 1], 'b-', alpha=0.7, label='Numerical')
    ax.plot(q[0, 0], q[0, 1], 'go', markersize=10, label='Start')
    ax.plot(q[-1, 0], q[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('q₁')
    ax.set_ylabel('q₂')
    ax.set_title('Configuration Space Trajectory')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axis('equal')

    # Phase space
    ax = axes[0, 1]
    ax.plot(q[:, 0], p[:, 0], 'b-', alpha=0.7)
    ax.plot(q[0, 0], p[0, 0], 'go', markersize=10)
    ax.set_xlabel('q₁')
    ax.set_ylabel('p₁')
    ax.set_title('Phase Space (q₁, p₁)')
    ax.grid(alpha=0.3)

    # Time evolution
    ax = axes[1, 0]
    ax.plot(t, q[:, 0], 'b-', label='q₁ (numerical)', alpha=0.7)
    ax.plot(t, q_exact, 'r--', label='q₁ (exact)', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('q₁')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # Energy
    ax = axes[1, 1]
    energies = [H.total_energy(q[i], p[i]) for i in range(len(t))]
    ax.plot(t, energies, 'b-')
    ax.axhline(H.total_energy(q0, p0), color='r', linestyle='--', label='Initial')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("_results/dynamics_harmonic.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n✓ Saved {out_path}")
    plt.close()

    print("="*70)


def free_energy_demo():
    """
    Demo 2: Belief dynamics with free energy potential.

    V(θ) = (1/2)||θ - θ*||² (quadratic well)
    M(θ) = diag([1, 1, ...]) (flat metric for simplicity)
    """
    print("\n" + "="*70)
    print("DEMO 2: FREE ENERGY POTENTIAL")
    print("="*70)

    # Target belief parameters
    theta_star = np.array([1.0, -0.5, 0.3])

    # Potential: V = (1/2)||θ - θ*||²
    def potential(theta):
        return 0.5 * np.sum((theta - theta_star)**2)

    # Metric: Flat for simplicity
    def metric(theta):
        return np.eye(len(theta))

    H = BeliefHamiltonian(potential, metric)

    # Initial conditions (displaced from minimum)
    q0 = np.array([0.0, 0.0, 0.0])
    p0 = np.array([0.5, 0.3, -0.2])

    print(f"Target θ*: {theta_star}")
    print(f"Initial θ: {q0}")
    print(f"Initial p: {p0}")
    print(f"Initial energy: {H.total_energy(q0, p0):.6f}")

    # Integrate
    integrator = PEFRL(H)
    t, q, p, stats = integrator.integrate(
        q0, p0,
        t_span=(0.0, 50.0),
        dt=0.05,
        save_interval=5
    )

    print(f"\nAfter {stats.steps} steps:")
    print(f"Final θ: {q[-1]}")
    print(f"Distance to θ*: {np.linalg.norm(q[-1] - theta_star):.6f}")
    print(f"Energy drift: {stats.energy_drift:+.6e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Parameter evolution
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(t, q[:, i], label=f'θ_{i}', alpha=0.7)
        ax.axhline(theta_star[i], color=f'C{i}', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('θ')
    ax.set_title('Belief Parameters Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Momentum evolution
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(t, p[:, i], label=f'p_{i}', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('p')
    ax.set_title('Conjugate Momenta')
    ax.legend()
    ax.grid(alpha=0.3)

    # Phase space (θ₀, p₀)
    ax = axes[1, 0]
    ax.plot(q[:, 0], p[:, 0], 'b-', alpha=0.7)
    ax.plot(q[0, 0], p[0, 0], 'go', markersize=10, label='Start')
    ax.plot(q[-1, 0], p[-1, 0], 'ro', markersize=10, label='End')
    ax.axvline(theta_star[0], color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('θ₀')
    ax.set_ylabel('p₀')
    ax.set_title('Phase Space')
    ax.legend()
    ax.grid(alpha=0.3)

    # Energy
    ax = axes[1, 1]
    energies = [H.total_energy(q[i], p[i]) for i in range(len(t))]
    ax.plot(t, energies, 'b-')
    ax.axhline(energies[0], color='r', linestyle='--', label='Initial')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("_results/dynamics_free_energy.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n✓ Saved {out_path}")
    plt.close()

    print("="*70)


def integrator_comparison_demo():
    """
    Demo 3: Compare different integrators.
    """
    print("\n" + "="*70)
    print("DEMO 3: INTEGRATOR COMPARISON")
    print("="*70)

    # Simple Hamiltonian
    def potential(q):
        return 0.5 * np.sum(q**2)

    def metric(q):
        return np.eye(len(q))

    H = BeliefHamiltonian(potential, metric)

    # Initial conditions
    q0 = np.array([1.0, 0.5])
    p0 = np.array([0.0, 0.8])

    # Compare integrators
    results = compare_integrators(H, q0, p0, dt=0.1, t_end=100.0)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    drifts = [results[m].energy_drift for m in methods]
    stds = [results[m].energy_std for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, np.abs(drifts), width, label='|Energy Drift|', alpha=0.7)
    ax.bar(x + width/2, stds, width, label='Energy Std', alpha=0.7)

    ax.set_xlabel('Method')
    ax.set_ylabel('Energy Error')
    ax.set_title('Integrator Energy Conservation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    out_path = Path("_results/dynamics_integrator_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n✓ Saved {out_path}")
    plt.close()


def regime_comparison_demo():
    """
    Demo 4: Overdamped vs underdamped regimes.
    """
    print("\n" + "="*70)
    print("DEMO 4: OVERDAMPED VS UNDERDAMPED")
    print("="*70)

    # Potential well
    def potential(q):
        return 0.5 * np.sum(q**2)

    def metric(q):
        return np.eye(len(q))

    H = BeliefHamiltonian(potential, metric)

    q0 = np.array([2.0])
    p0 = np.array([0.0])

    # Underdamped (Hamiltonian)
    integrator = Verlet(H)
    t_un, q_un, p_un, _ = integrator.integrate(q0, p0, (0, 20), dt=0.01, save_interval=10)

    # Overdamped (gradient flow)
    q_ov = [q0[0]]
    t_ov = [0.0]
    q_curr = q0.copy()
    dt = 0.01
    friction = 2.0

    for i in range(2000):
        dq = overdamped_limit(H, q_curr, friction)
        q_curr = q_curr + dt * dq
        if i % 10 == 0:
            q_ov.append(q_curr[0])
            t_ov.append((i+1) * dt)

    q_ov = np.array(q_ov)
    t_ov = np.array(t_ov)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time evolution
    ax = axes[0]
    ax.plot(t_un, q_un[:, 0], 'b-', label='Underdamped (Hamiltonian)', linewidth=2)
    ax.plot(t_ov, q_ov, 'r--', label='Overdamped (Gradient Flow)', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('θ')
    ax.set_title('Relaxation to Equilibrium')
    ax.legend()
    ax.grid(alpha=0.3)

    # Phase space
    ax = axes[1]
    ax.plot(q_un[:, 0], p_un[:, 0], 'b-', label='Underdamped', linewidth=2)
    ax.plot(q_ov, np.zeros_like(q_ov), 'ro', markersize=3, label='Overdamped', alpha=0.5)
    ax.set_xlabel('θ')
    ax.set_ylabel('p')
    ax.set_title('Phase Space')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("_results/dynamics_regime_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n✓ Saved {out_path}")
    plt.close()

    print("\nKey differences:")
    print(f"  Underdamped: Oscillates around minimum (conserved energy)")
    print(f"  Overdamped: Exponential decay to minimum (dissipative)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Dynamics demo")
    parser.add_argument(
        "--demo",
        choices=["all", "harmonic", "free_energy", "comparison", "regimes"],
        default="all"
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("HAMILTONIAN & LAGRANGIAN DYNAMICS DEMO")
    print("="*70)
    print("\nKey Insight: Fisher metric provides kinetic terms!")
    print("  KL ≈ const + force·δθ + (1/2)δθ^T·MASS·δθ")
    print("  → Hamiltonian dynamics with G = Fisher metric")
    print("="*70)

    if args.demo in ["all", "harmonic"]:
        harmonic_oscillator_demo()

    if args.demo in ["all", "free_energy"]:
        free_energy_demo()

    if args.demo in ["all", "comparison"]:
        integrator_comparison_demo()

    if args.demo in ["all", "regimes"]:
        regime_comparison_demo()

    print("\n" + "="*70)
    print("✓ ALL DEMOS COMPLETE")
    print("="*70)
    print("Results saved to _results/dynamics_*.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()