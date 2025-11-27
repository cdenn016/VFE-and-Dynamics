#!/usr/bin/env python3
"""
Critical Points Scanner - Click to Run
======================================

Configure the settings below, then run this file directly.

Author: Claude
Date: November 2025
"""

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Mode: "scan", "bifurcation", or "demo"
MODE = "demo"

# System setup
N_AGENTS = 6          # Number of agents
K_LATENT = 3          # Latent dimension
SEED = 42             # Random seed

# Critical point scan settings (for MODE = "scan")
SCAN_AGENT_IDX = 0           # Which agent to analyze
SCAN_PARAM = "mu_q"          # Parameter to scan: "mu_q" or "mu_p"
SCAN_DIMS = (0, 1)           # Which dimensions to scan
GRID_RANGE = (-2.0, 2.0)     # Min/max range for grid
GRID_RESOLUTION = 15         # Grid points per dimension
GRADIENT_THRESHOLD = 0.2     # Threshold for candidate critical points

# Bifurcation scan settings (for MODE = "bifurcation")
BIF_PARAM = "lambda_prior_align"  # Control parameter to vary
BIF_RANGE = (0.5, 2.5)            # Parameter range
BIF_POINTS = 20                    # Number of parameter values
REFINE_BIFURCATIONS = True         # Refine locations via bisection

# Output settings
OUTPUT_DIR = None              # Set to path like "_results/critical_points" to save
SHOW_PLOTS = True              # Display plots interactively

# =============================================================================
# END CONFIGURATION - Don't edit below unless you know what you're doing
# =============================================================================

import numpy as np
from pathlib import Path
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_system(n_agents: int, K: int, seed: int):
    """Create a test multi-agent system."""
    np.random.seed(seed)

    from geometry.geometry_base import BaseManifold, TopologyType
    from config import AgentConfig, SystemConfig, MaskConfig
    from agent.agents import Agent
    from agent.masking import SupportRegionSmooth
    from agent.system import MultiAgentSystem
    from math_utils.generators import generate_so3_generators

    manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)

    system_cfg = SystemConfig(
        lambda_self=3.0,
        lambda_belief_align=2.0,
        lambda_prior_align=1.5,
        lambda_obs=0.0,
        lambda_phi=0.0,
        kappa_beta=1.0,
        kappa_gamma=1.0,
        identical_priors="off"
    )

    agent_cfg = AgentConfig(K=K, observation_noise=0.1)
    agents = []
    generators = generate_so3_generators(K)
    mask_cfg = MaskConfig()

    for i in range(n_agents):
        rng_i = np.random.default_rng(seed + i)

        support = SupportRegionSmooth(
            mask_binary=np.array(True),
            base_shape=(),
            config=mask_cfg
        )
        support.base_manifold = manifold

        agent = Agent(
            agent_id=i,
            config=agent_cfg,
            rng=rng_i,
            base_manifold=manifold
        )

        agent.support = support
        agent.geometry.support = support
        agent.geometry.n_active = support.n_active
        agent._initialize_gauge()
        agent.generators = generators
        agent.mu_q = rng_i.standard_normal(K) * 0.5
        agent.Sigma_q = np.eye(K)
        agent.mu_p = rng_i.standard_normal(K) * 0.3
        agent.Sigma_p = np.eye(K)

        agents.append(agent)

    return MultiAgentSystem(agents, system_cfg, manifold)


def run_demo():
    """Quick demonstration."""
    from critical_points import quick_scan, plot_critical_point_summary
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CRITICAL POINTS SCANNER - DEMO")
    print("=" * 60)
    print()

    print("Creating test system...")
    system = create_test_system(N_AGENTS, K_LATENT, SEED)
    print(f"  {len(system.agents)} agents, K={K_LATENT}")
    print()

    print("Scanning for critical points...")
    print("-" * 40)
    scan = quick_scan(system, verbose=True)
    print()

    if SHOW_PLOTS:
        print("Displaying plots...")
        fig = plot_critical_point_summary(scan, title="Critical Points Demo")
        plt.show()

    print("Done!")
    return scan


def run_scan():
    """Full critical point scan."""
    from critical_points import (
        scan_for_critical_points,
        classify_all_critical_points,
        plot_critical_point_summary,
        plot_energy_landscape_2d,
        plot_gradient_norm_field
    )
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CRITICAL POINTS SCAN")
    print("=" * 60)
    print(f"Agents: {N_AGENTS}, K: {K_LATENT}")
    print(f"Grid: {GRID_RESOLUTION}x{GRID_RESOLUTION}, range: {GRID_RANGE}")
    print(f"Scanning: {SCAN_PARAM}[{SCAN_DIMS[0]}, {SCAN_DIMS[1]}]")
    print()

    print("Creating system...")
    system = create_test_system(N_AGENTS, K_LATENT, SEED)
    print()

    print("Scanning for critical points...")
    scan = scan_for_critical_points(
        system,
        agent_idx=SCAN_AGENT_IDX,
        param=SCAN_PARAM,
        dims=SCAN_DIMS,
        grid_range=GRID_RANGE,
        grid_resolution=GRID_RESOLUTION,
        gradient_threshold=GRADIENT_THRESHOLD,
        refine=True,
        verbose=True
    )
    print()

    if len(scan.critical_points) > 0:
        print("Classifying via Hessian analysis...")
        classify_all_critical_points(system, scan, agent_idx=SCAN_AGENT_IDX)
        print()

    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Critical points: {len(scan.critical_points)}")
    print(f"  Stable (minima):   {scan.n_minima}")
    print(f"  Unstable (maxima): {scan.n_maxima}")
    print(f"  Saddles:           {scan.n_saddles}")
    print()

    for i, cp in enumerate(scan.critical_points):
        print(f"  [{i}] {cp}")

    global_min = scan.get_global_minimum()
    if global_min:
        print(f"\nGlobal minimum: F = {global_min.energy:.4f}")

    # Save
    if OUTPUT_DIR:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "scan_results.pkl", "wb") as f:
            pickle.dump(scan, f)

        fig = plot_critical_point_summary(scan)
        fig.savefig(output_path / "summary.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        if scan.energy_landscape is not None:
            x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], GRID_RESOLUTION)
            X, Y = np.meshgrid(x, x)

            fig = plot_energy_landscape_2d(X, Y, scan.energy_landscape, scan.critical_points)
            fig.savefig(output_path / "energy_landscape.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig = plot_gradient_norm_field(X, Y, scan.gradient_field, scan.critical_points)
            fig.savefig(output_path / "gradient_field.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"\nSaved to {output_path}/")

    # Plot
    if SHOW_PLOTS:
        fig = plot_critical_point_summary(scan)
        plt.show()

    return scan


def run_bifurcation():
    """Bifurcation analysis."""
    from critical_points import (
        scan_for_bifurcations,
        refine_bifurcation_point,
        plot_bifurcation_diagram
    )
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("BIFURCATION SCAN")
    print("=" * 60)
    print(f"Parameter: {BIF_PARAM}")
    print(f"Range: {BIF_RANGE}")
    print(f"Points: {BIF_POINTS}")
    print()

    print("Creating system...")
    system = create_test_system(N_AGENTS, K_LATENT, SEED)
    print()

    print(f"Scanning {BIF_PARAM}...")
    diagram = scan_for_bifurcations(
        system,
        param_name=BIF_PARAM,
        param_range=BIF_RANGE,
        n_points=BIF_POINTS,
        agent_idx=SCAN_AGENT_IDX,
        verbose=True
    )
    print()

    if REFINE_BIFURCATIONS and len(diagram.bifurcations) > 0:
        print("Refining bifurcation locations...")
        for bif in diagram.bifurcations:
            refine_bifurcation_point(system, bif, verbose=True)
        print()

    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Bifurcations found: {diagram.n_bifurcations}")
    for bif in diagram.bifurcations:
        print(f"  {bif}")

    # Save
    if OUTPUT_DIR:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "bifurcation_diagram.pkl", "wb") as f:
            pickle.dump(diagram, f)

        fig = plot_bifurcation_diagram(diagram, observable='energy')
        fig.savefig(output_path / "bifurcation_diagram.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\nSaved to {output_path}/")

    # Plot
    if SHOW_PLOTS:
        fig = plot_bifurcation_diagram(diagram, observable='energy')
        plt.show()

    return diagram


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if MODE == "demo":
        run_demo()
    elif MODE == "scan":
        run_scan()
    elif MODE == "bifurcation":
        run_bifurcation()
    else:
        print(f"Unknown MODE: {MODE}")
        print("Set MODE to 'demo', 'scan', or 'bifurcation'")
