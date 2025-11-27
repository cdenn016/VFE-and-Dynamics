#!/usr/bin/env python3
"""
Critical Points Scanner - Entry Point
======================================

Scan the VFE landscape for critical points and analyze their stability.

Usage:
    python -m critical_points.run_scanner [options]

Examples:
    # Quick scan with defaults
    python -m critical_points.run_scanner

    # Detailed scan with custom parameters
    python -m critical_points.run_scanner --agents 8 --resolution 25 --range -3 3

    # Bifurcation scan
    python -m critical_points.run_scanner --mode bifurcation --param lambda_prior_align

    # Save results
    python -m critical_points.run_scanner --output _results/critical_points

Author: Claude
Date: November 2025
"""

import argparse
import numpy as np
from pathlib import Path
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_system(n_agents: int = 6, K: int = 3, seed: int = 42):
    """
    Create a test multi-agent system for critical point analysis.

    Args:
        n_agents: Number of agents
        K: Latent dimension
        seed: Random seed

    Returns:
        system: Configured MultiAgentSystem
    """
    np.random.seed(seed)

    from geometry.geometry_base import BaseManifold, TopologyType
    from config import AgentConfig, SystemConfig, MaskConfig
    from agent.agents import Agent
    from agent.masking import SupportRegionSmooth
    from agent.system import MultiAgentSystem
    from math_utils.generators import generate_so3_generators

    # Create point manifold (0D)
    manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)

    # System configuration
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

    # Agent configuration
    agent_cfg = AgentConfig(
        K=K,
        observation_noise=0.1
    )

    # Create agents
    agents = []
    generators = generate_so3_generators(K)
    mask_cfg = MaskConfig()

    for i in range(n_agents):
        rng_i = np.random.default_rng(seed + i)

        # Create support region
        support = SupportRegionSmooth(
            mask_binary=np.array(True),
            base_shape=(),
            config=mask_cfg
        )
        support.base_manifold = manifold

        # Create agent
        agent = Agent(
            agent_id=i,
            config=agent_cfg,
            rng=rng_i,
            base_manifold=manifold
        )

        # Attach support
        agent.support = support
        agent.geometry.support = support
        agent.geometry.n_active = support.n_active

        # Initialize gauge and generators
        agent._initialize_gauge()
        agent.generators = generators

        # Initialize belief/prior fields with some spread
        agent.mu_q = rng_i.standard_normal(K) * 0.5
        agent.Sigma_q = np.eye(K)
        agent.mu_p = rng_i.standard_normal(K) * 0.3
        agent.Sigma_p = np.eye(K)

        agents.append(agent)

    # Create system
    system = MultiAgentSystem(agents, system_cfg, manifold)

    return system


def run_critical_point_scan(args):
    """Run critical point scan mode."""
    from critical_points import (
        scan_for_critical_points,
        classify_all_critical_points,
        plot_critical_point_summary,
        plot_energy_landscape_2d,
        plot_gradient_norm_field
    )
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("CRITICAL POINTS SCANNER")
    print("=" * 70)
    print(f"Agents: {args.agents}")
    print(f"Latent dim: {args.K}")
    print(f"Grid resolution: {args.resolution}")
    print(f"Grid range: [{args.range[0]}, {args.range[1]}]")
    print(f"Parameter: {args.scan_param}")
    print(f"Dimensions: ({args.dims[0]}, {args.dims[1]})")
    print()

    # Create system
    print("Creating test system...")
    system = create_test_system(n_agents=args.agents, K=args.K, seed=args.seed)
    print(f"  Created system with {len(system.agents)} agents")
    print()

    # Run scan
    print("Scanning for critical points...")
    scan = scan_for_critical_points(
        system,
        agent_idx=args.agent_idx,
        param=args.scan_param,
        dims=tuple(args.dims),
        grid_range=tuple(args.range),
        grid_resolution=args.resolution,
        gradient_threshold=args.threshold,
        refine=True,
        verbose=True
    )
    print()

    # Classify critical points
    if len(scan.critical_points) > 0:
        print("Classifying critical points via Hessian analysis...")
        classify_all_critical_points(
            system, scan,
            agent_idx=args.agent_idx,
            include_priors=True
        )
        print()

    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total critical points found: {len(scan.critical_points)}")
    print(f"  Stable (minima): {scan.n_minima}")
    print(f"  Unstable (maxima): {scan.n_maxima}")
    print(f"  Saddle points: {scan.n_saddles}")
    print()

    if len(scan.critical_points) > 0:
        print("Critical Points:")
        for i, cp in enumerate(scan.critical_points):
            print(f"  [{i}] {cp}")
            if cp.hessian_eigenvalues is not None:
                eigs = cp.hessian_eigenvalues
                print(f"      Eigenvalues: [{', '.join(f'{e:.3f}' for e in eigs[:5])}{'...' if len(eigs) > 5 else ''}]")
        print()

        global_min = scan.get_global_minimum()
        if global_min:
            print(f"Global minimum: F = {global_min.energy:.4f}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scan data
        with open(output_dir / "scan_results.pkl", "wb") as f:
            pickle.dump(scan, f)
        print(f"\nResults saved to {output_dir / 'scan_results.pkl'}")

        # Generate plots
        print("Generating plots...")

        # Summary plot
        fig = plot_critical_point_summary(scan, title="Critical Points Analysis")
        fig.savefig(output_dir / "summary.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Energy landscape
        if scan.energy_landscape is not None:
            x = np.linspace(args.range[0], args.range[1], args.resolution)
            y = np.linspace(args.range[0], args.range[1], args.resolution)
            X, Y = np.meshgrid(x, y)

            fig = plot_energy_landscape_2d(
                X, Y, scan.energy_landscape,
                scan.critical_points,
                agent_idx=args.agent_idx,
                param=args.scan_param,
                dims=tuple(args.dims),
                title="Energy Landscape"
            )
            fig.savefig(output_dir / "energy_landscape.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig = plot_gradient_norm_field(
                X, Y, scan.gradient_field,
                scan.critical_points,
                agent_idx=args.agent_idx,
                param=args.scan_param,
                dims=tuple(args.dims),
                title="Gradient Norm Field"
            )
            fig.savefig(output_dir / "gradient_field.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"Plots saved to {output_dir}")

    # Show plots if not saving
    if not args.output and not args.no_plot:
        print("\nDisplaying plots...")
        fig = plot_critical_point_summary(scan, title="Critical Points Analysis")
        plt.show()

    return scan


def run_bifurcation_scan(args):
    """Run bifurcation scan mode."""
    from critical_points import (
        scan_for_bifurcations,
        refine_bifurcation_point,
        plot_bifurcation_diagram,
        plot_eigenvalue_spectrum,
        compute_eigenvalue_spectrum_along_branch
    )
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("BIFURCATION SCANNER")
    print("=" * 70)
    print(f"Parameter: {args.param}")
    print(f"Range: [{args.param_range[0]}, {args.param_range[1]}]")
    print(f"Points: {args.points}")
    print()

    # Create system
    print("Creating test system...")
    system = create_test_system(n_agents=args.agents, K=args.K, seed=args.seed)
    print(f"  Created system with {len(system.agents)} agents")
    print()

    # Run bifurcation scan
    print(f"Scanning {args.param} for bifurcations...")
    diagram = scan_for_bifurcations(
        system,
        param_name=args.param,
        param_range=tuple(args.param_range),
        n_points=args.points,
        agent_idx=args.agent_idx,
        verbose=True
    )
    print()

    # Refine bifurcations
    if len(diagram.bifurcations) > 0 and args.refine:
        print("Refining bifurcation locations...")
        for bif in diagram.bifurcations:
            refine_bifurcation_point(system, bif, tolerance=0.001, verbose=True)
        print()

    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Bifurcations detected: {diagram.n_bifurcations}")
    print()

    if diagram.n_bifurcations > 0:
        print("Bifurcation Events:")
        for bif in diagram.bifurcations:
            print(f"  {bif}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save diagram data
        with open(output_dir / "bifurcation_diagram.pkl", "wb") as f:
            pickle.dump(diagram, f)
        print(f"\nResults saved to {output_dir / 'bifurcation_diagram.pkl'}")

        # Generate plots
        print("Generating plots...")

        fig = plot_bifurcation_diagram(
            diagram,
            observable='energy',
            title=f"Bifurcation Diagram: {args.param}"
        )
        fig.savefig(output_dir / "bifurcation_diagram.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Plots saved to {output_dir}")

    # Show plots if not saving
    if not args.output and not args.no_plot:
        print("\nDisplaying plots...")
        fig = plot_bifurcation_diagram(
            diagram,
            observable='energy',
            title=f"Bifurcation Diagram: {args.param}"
        )
        plt.show()

    return diagram


def run_quick_demo(args):
    """Run a quick demonstration of the module."""
    from critical_points import quick_scan, quick_bifurcation_scan
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("CRITICAL POINTS SCANNER - QUICK DEMO")
    print("=" * 70)
    print()

    # Create system
    print("Creating test system...")
    system = create_test_system(n_agents=6, K=3, seed=42)
    print(f"  Created system with {len(system.agents)} agents")
    print()

    # Run quick scan
    print("Running quick critical point scan...")
    print("-" * 40)
    scan = quick_scan(system, verbose=True)
    print()

    # Show results
    if not args.no_plot:
        from critical_points import plot_critical_point_summary
        print("Displaying summary plot...")
        fig = plot_critical_point_summary(scan, title="Quick Demo - Critical Points")
        plt.show()

    print()
    print("Demo complete!")
    print()
    print("To run a full scan:")
    print("  python -m critical_points.run_scanner --resolution 25 --output _results/cp_scan")
    print()
    print("To run a bifurcation scan:")
    print("  python -m critical_points.run_scanner --mode bifurcation --param lambda_prior_align")


def main():
    parser = argparse.ArgumentParser(
        description="Scan the VFE landscape for critical points and bifurcations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo
  python -m critical_points.run_scanner --mode demo

  # Critical point scan
  python -m critical_points.run_scanner --agents 8 --resolution 20

  # Bifurcation scan
  python -m critical_points.run_scanner --mode bifurcation --param lambda_prior_align

  # Save results
  python -m critical_points.run_scanner --output _results/critical_points
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="scan",
        choices=["scan", "bifurcation", "demo"],
        help="Mode: 'scan' for critical points, 'bifurcation' for bifurcation analysis, 'demo' for quick demo"
    )

    # System parameters
    parser.add_argument(
        "--agents",
        type=int,
        default=6,
        help="Number of agents (default: 6)"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Latent dimension (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Scan parameters
    parser.add_argument(
        "--agent-idx",
        type=int,
        default=0,
        help="Which agent to analyze (default: 0)"
    )
    parser.add_argument(
        "--scan-param",
        type=str,
        default="mu_q",
        choices=["mu_q", "mu_p"],
        help="Parameter to scan (default: mu_q)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Dimensions to scan (default: 0 1)"
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=[-2.0, 2.0],
        help="Grid range (default: -2.0 2.0)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=15,
        help="Grid resolution (default: 15)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Gradient threshold for candidates (default: 0.2)"
    )

    # Bifurcation parameters
    parser.add_argument(
        "--param",
        type=str,
        default="lambda_prior_align",
        help="Control parameter for bifurcation scan (default: lambda_prior_align)"
    )
    parser.add_argument(
        "--param-range",
        type=float,
        nargs=2,
        default=[0.5, 2.5],
        help="Parameter range for bifurcation scan (default: 0.5 2.5)"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="Number of parameter points (default: 20)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        dest="no_refine",
        help="Skip bifurcation refinement"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: None, display only)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't display plots"
    )

    args = parser.parse_args()
    args.refine = not args.no_refine

    # Run appropriate mode
    if args.mode == "demo":
        run_quick_demo(args)
    elif args.mode == "scan":
        run_critical_point_scan(args)
    elif args.mode == "bifurcation":
        run_bifurcation_scan(args)


if __name__ == "__main__":
    main()
