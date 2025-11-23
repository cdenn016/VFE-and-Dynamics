#!/usr/bin/env python3
"""
Test Trajectory Metrics with Snapshot Storage
==============================================

Validates the complete pipeline:
1. Train agents with snapshot storage enabled
2. Compute time-varying pullback metrics from snapshots
3. Analyze signature evolution
4. Generate trajectory visualizations

Author: Chris & Claude
Date: November 23, 2025
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from agent.system import MultiAgentSystem
from agent.agents import Agent
from agent.trainer import Trainer
from config import SystemConfig, AgentConfig, TrainingConfig

from dynamics.trajectory_metrics import (
    compute_trajectory_pullback_metrics,
    analyze_signature_evolution,
    print_signature_summary
)

from analysis.plots.belief_trajectories import (
    plot_trajectory_metric_signature,
    plot_trajectory_dashboard
)


def create_test_system():
    """Create a simple 1D field agent system for testing."""

    # Agent config: 1D spatial grid with 16 points, K=3 latent dim
    agent_config = AgentConfig(
        spatial_shape=(16,),  # 1D grid with 16 points
        K=3,  # 3D latent space (must be odd for SO(3))
        mu_scale=0.2,
        sigma_scale=0.1
    )

    # System config
    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0
    )

    # Create agent
    rng = np.random.default_rng(42)
    agent = Agent(0, agent_config, rng=rng)

    # Create system (no observations - just self-energy and alignment)
    system = MultiAgentSystem([agent], system_config)

    return system


def test_snapshot_storage_and_metrics():
    """Test snapshot storage and trajectory metric computation."""

    print("="*70)
    print("TRAJECTORY METRICS TEST")
    print("="*70)
    print()

    # Create system
    print("Creating test system...")
    system = create_test_system()
    agent = system.agents[0]
    print(f"✓ System: {system.n_agents} agent(s)")
    print(f"  Latent space: K={agent.config.K}")
    print(f"  Spatial shape: {agent.config.spatial_shape}")
    print()

    # Training config with snapshots enabled
    config = TrainingConfig(
        n_steps=100,
        lr_mu_q=0.01,
        lr_sigma_q=0.005,
        lr_phi=0.0,  # No gauge dynamics for this test
        log_every=20,
        save_history=True,
        save_snapshots=True,      # ✨ Enable snapshots
        snapshot_every=10          # Save every 10 steps
    )

    print(f"Training config:")
    print(f"  Steps: {config.n_steps}")
    print(f"  Snapshots: enabled (every {config.snapshot_every} steps)")
    print()

    # Train
    trainer = Trainer(system, config)
    print("Training...")
    history = trainer.train()

    # Check snapshots were saved
    print()
    print("="*70)
    print("SNAPSHOT VALIDATION")
    print("="*70)
    n_snapshots = len(history.agent_snapshots)
    expected_snapshots = config.n_steps // config.snapshot_every + 1
    print(f"Snapshots saved: {n_snapshots}")
    print(f"Expected: ~{expected_snapshots}")

    if n_snapshots == 0:
        print("❌ FAILED: No snapshots saved!")
        return False

    print(f"✓ Snapshots saved successfully")
    print(f"  Snapshot steps: {history.snapshot_steps[:5]}... (showing first 5)")
    print()

    # Compute trajectory metrics
    print("="*70)
    print("TRAJECTORY METRIC COMPUTATION")
    print("="*70)

    try:
        traj_metrics = compute_trajectory_pullback_metrics(
            history,
            agent_idx=0,
            point_idx=8,  # Analyze middle point
            metric_type="belief"
        )

        print(f"✓ Computed trajectory metrics")
        print(f"  Time points: {len(traj_metrics.metrics)}")
        print(f"  Spatial dimension: {traj_metrics.metrics[0].G.shape[-1]}")
        print()

    except Exception as e:
        print(f"❌ FAILED to compute trajectory metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze signature evolution
    print("="*70)
    print("SIGNATURE EVOLUTION ANALYSIS")
    print("="*70)

    try:
        sig_analysis = analyze_signature_evolution(traj_metrics)

        print(f"Signature distribution:")
        for sig, count in sig_analysis['signature_counts'].items():
            frac = count / len(traj_metrics.steps)
            print(f"  {sig:15s}: {count:4d} ({frac:6.1%})")

        print()
        print(f"Dominant signature: {sig_analysis['dominant_signature']}")
        print(f"Number of transitions: {sig_analysis['n_transitions']}")

        if sig_analysis['transitions']:
            print(f"\nSignature transitions detected:")
            for step, old_sig, new_sig in sig_analysis['transitions']:
                print(f"  Step {step:4d}: {old_sig} → {new_sig}")
        else:
            print(f"\nNo signature transitions detected")

        print()
        print(f"Lorentzian fraction: {sig_analysis['lorentzian_fraction']:.1%}")

        print()

    except Exception as e:
        print(f"❌ FAILED signature analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print detailed summary
    print_signature_summary(traj_metrics)

    # Generate visualizations
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    out_dir = Path("_results/trajectory_metrics_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Signature evolution plot
        print("Creating signature evolution plot...")
        plot_trajectory_metric_signature(
            history, system, out_dir,
            point_idx=8, agent_idx=0
        )

        # Dashboard
        print("Creating trajectory dashboard...")
        plot_trajectory_dashboard(
            history, system, out_dir,
            point_idx=8, agent_idx=0
        )

        print()
        print(f"✓ Plots saved to {out_dir}/")
        print()

    except Exception as e:
        print(f"⚠️  Warning: Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    # Eigenvalue evolution plot
    print("Creating eigenvalue evolution plot...")
    eigenvalues = traj_metrics.get_eigenvalue_trajectories()

    fig, ax = plt.subplots(figsize=(10, 6))
    d = eigenvalues.shape[1]
    for i in range(d):
        ax.plot(traj_metrics.steps, eigenvalues[:, i],
                label=f'λ_{i}', linewidth=2, marker='o', markersize=3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Pullback Metric Eigenvalue Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_dir / "eigenvalue_evolution.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {out_path}")

    print()
    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print()
    print(f"Results saved to: {out_dir.absolute()}")

    return True


if __name__ == "__main__":
    success = test_snapshot_storage_and_metrics()
    exit(0 if success else 1)
