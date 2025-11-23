#!/usr/bin/env python3
"""
Quick 2D Validation Test
========================

Validates that existing Agent/Trainer infrastructure works with 2D spatial fields.
Just needs to run without errors - we'll do deep analysis in Lorentzian search.

Author: Chris & Claude
Date: November 23, 2025
"""

import numpy as np
from pathlib import Path

from agent.system import MultiAgentSystem
from agent.agents import Agent
from agent.trainer import Trainer
from config import SystemConfig, AgentConfig, TrainingConfig

from dynamics.trajectory_metrics import (
    compute_trajectory_pullback_metrics,
    analyze_signature_evolution
)


def test_2d_basic():
    """Validate 2D agent creation and training."""

    print("="*70)
    print("2D VALIDATION TEST")
    print("="*70)
    print()

    # Create 2D agent
    print("Creating 2D agent...")
    agent_config = AgentConfig(
        spatial_shape=(8, 8),  # 2D grid: 8×8
        K=3,  # K must be odd for SO(3)
        mu_scale=0.2,
        sigma_scale=0.1
    )

    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0
    )

    rng = np.random.default_rng(42)
    agent = Agent(0, agent_config, rng=rng)
    system = MultiAgentSystem([agent], system_config)

    print(f"✓ Agent created")
    print(f"  Spatial shape: {agent.config.spatial_shape}")
    print(f"  Total spatial points: {np.prod(agent.config.spatial_shape)}")
    print(f"  Latent dim: K={agent.config.K}")
    print(f"  mu_q shape: {agent.mu_q.shape}")
    print(f"  Sigma_q shape: {agent.Sigma_q.shape}")
    print()

    # Train briefly
    print("Training for 50 steps...")
    config = TrainingConfig(
        n_steps=50,
        lr_mu_q=0.01,
        lr_sigma_q=0.005,
        lr_phi=0.0,
        log_every=25,
        save_snapshots=True,
        snapshot_every=10
    )

    trainer = Trainer(system, config)
    history = trainer.train()

    print()
    print(f"✓ Training complete")
    print(f"  Snapshots saved: {len(history.agent_snapshots)}")
    print()

    # Compute metrics at 2D point
    print("Computing trajectory metrics...")
    point_2d = (4, 4)  # Center of 8×8 grid

    traj_metrics = compute_trajectory_pullback_metrics(
        history,
        agent_idx=0,
        point_idx=point_2d,
        metric_type="belief"
    )

    sig_analysis = analyze_signature_evolution(traj_metrics)

    print(f"✓ Metrics computed at point {point_2d}")
    print(f"  Snapshots analyzed: {len(traj_metrics.metrics)}")
    print(f"  Dominant signature: {sig_analysis['dominant_signature']}")
    print(f"  Lorentzian fraction: {sig_analysis['lorentzian_fraction']:.1%}")
    print()

    print("="*70)
    print("✓ 2D VALIDATION PASSED")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_2d_basic()
    exit(0 if success else 1)
