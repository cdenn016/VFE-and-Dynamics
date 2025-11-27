"""
Critical Points Scanner Module
==============================

Tools for detecting, classifying, and analyzing critical points in the
Variational Free Energy (VFE) landscape.

Critical points are where the gradient vanishes (nabla F = 0):
- Local minima: Stable equilibria (attractors)
- Local maxima: Unstable equilibria (repellers)
- Saddle points: Transition states between basins

This module provides:
1. **Detection**: Find critical points via gradient descent and grid scanning
2. **Classification**: Analyze Hessian eigenvalues for stability type
3. **Bifurcation Analysis**: Track how critical points change with parameters
4. **Visualization**: Plot energy landscapes, basins, and bifurcation diagrams

Example Usage
-------------

Basic critical point detection:

    from critical_points import (
        scan_for_critical_points,
        classify_all_critical_points,
        plot_critical_point_summary
    )

    # Scan parameter space for critical points
    scan = scan_for_critical_points(
        system,
        agent_idx=0,
        param='mu_q',
        dims=(0, 1),
        grid_range=(-2, 2),
        grid_resolution=20,
        verbose=True
    )

    # Classify stability via Hessian analysis
    classify_all_critical_points(system, scan)

    # Visualize results
    fig = plot_critical_point_summary(scan)
    plt.show()


Bifurcation analysis:

    from critical_points import scan_for_bifurcations, plot_bifurcation_diagram

    # Scan for bifurcations as parameter varies
    diagram = scan_for_bifurcations(
        system,
        param_name='lambda_prior_align',
        param_range=(0.5, 2.0),
        n_points=30,
        verbose=True
    )

    # Plot bifurcation diagram
    fig = plot_bifurcation_diagram(diagram, observable='energy')
    plt.show()


Author: Claude
Date: November 2025
"""

# =============================================================================
# Core Detection
# =============================================================================

from critical_points.detector import (
    # Data classes
    CriticalPoint,
    CriticalPointType,
    CriticalPointScan,

    # Detection functions
    compute_gradient_norm,
    compute_per_agent_gradient_norms,
    detect_critical_point_at_state,
    find_critical_point_gradient_descent,
    find_critical_points_random_restarts,

    # Landscape computation
    compute_gradient_norm_field,
    compute_energy_landscape,

    # Scanning
    scan_for_critical_points,
    track_critical_points_under_parameter_change,

    # State management
    extract_system_state,
    set_system_state,
)

# =============================================================================
# Stability Analysis
# =============================================================================

from critical_points.stability import (
    # Data classes
    HessianAnalysis,

    # Hessian computation
    compute_hessian_numerical,
    compute_full_hessian,
    analyze_hessian,

    # Classification
    classify_critical_point,
    classify_all_critical_points,

    # Basin analysis
    compute_stability_basin,

    # Direction analysis
    compute_escape_directions,
    compute_approach_directions,
    estimate_local_timescales,
)

# =============================================================================
# Bifurcation Detection
# =============================================================================

from critical_points.bifurcation import (
    # Data classes
    BifurcationType,
    BifurcationEvent,
    BifurcationDiagram,

    # Detection
    detect_eigenvalue_zero_crossing,
    classify_bifurcation,
    scan_for_bifurcations,
    refine_bifurcation_point,

    # Analysis
    compute_eigenvalue_spectrum_along_branch,
    detect_bifurcations_from_eigenvalues,
    compute_codimension_one_bifurcation_normal_form,
)

# =============================================================================
# Visualization
# =============================================================================

from critical_points.visualization import (
    # 2D/3D landscape plots
    plot_energy_landscape_2d,
    plot_energy_landscape_3d,
    plot_gradient_norm_field,

    # Basin visualization
    plot_stability_basin,

    # Bifurcation diagrams
    plot_bifurcation_diagram,
    plot_eigenvalue_spectrum,

    # Summary plots
    plot_critical_point_summary,

    # Animation
    create_animation_frames,

    # Style constants
    CP_COLORS,
    CP_MARKERS,
)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_scan(system, verbose: bool = True):
    """
    Quickly scan for critical points with default parameters.

    This is a convenience function for rapid exploration.

    Args:
        system: MultiAgentSystem
        verbose: Print progress

    Returns:
        CriticalPointScan with detected and classified critical points
    """
    scan = scan_for_critical_points(
        system,
        agent_idx=0,
        param='mu_q',
        dims=(0, 1),
        grid_range=(-2.0, 2.0),
        grid_resolution=15,
        gradient_threshold=0.2,
        refine=True,
        verbose=verbose
    )

    classify_all_critical_points(system, scan, agent_idx=0)

    if verbose:
        print(f"\nSummary:")
        print(f"  Total critical points: {len(scan.critical_points)}")
        print(f"  Minima (stable): {scan.n_minima}")
        print(f"  Maxima (unstable): {scan.n_maxima}")
        print(f"  Saddles: {scan.n_saddles}")

        global_min = scan.get_global_minimum()
        if global_min:
            print(f"  Global minimum energy: {global_min.energy:.4f}")

    return scan


def quick_bifurcation_scan(
    system,
    param_name: str = 'lambda_prior_align',
    param_range: tuple = (0.5, 2.0),
    verbose: bool = True
):
    """
    Quickly scan for bifurcations with default parameters.

    Args:
        system: MultiAgentSystem
        param_name: Control parameter to scan
        param_range: (min, max) parameter range
        verbose: Print progress

    Returns:
        BifurcationDiagram with detected bifurcations
    """
    diagram = scan_for_bifurcations(
        system,
        param_name=param_name,
        param_range=param_range,
        n_points=20,
        agent_idx=0,
        verbose=verbose
    )

    if verbose:
        print(f"\nSummary:")
        print(f"  Branches tracked: {diagram.n_branches}")
        print(f"  Bifurcations detected: {diagram.n_bifurcations}")

        for bif in diagram.bifurcations:
            print(f"    {bif}")

    return diagram


# =============================================================================
# Module Info
# =============================================================================

__version__ = "1.0.0"
__author__ = "Claude"

__all__ = [
    # Core data classes
    'CriticalPoint',
    'CriticalPointType',
    'CriticalPointScan',
    'HessianAnalysis',
    'BifurcationType',
    'BifurcationEvent',
    'BifurcationDiagram',

    # Detection
    'compute_gradient_norm',
    'compute_per_agent_gradient_norms',
    'detect_critical_point_at_state',
    'find_critical_point_gradient_descent',
    'find_critical_points_random_restarts',
    'scan_for_critical_points',
    'track_critical_points_under_parameter_change',

    # Landscapes
    'compute_gradient_norm_field',
    'compute_energy_landscape',

    # Stability
    'compute_hessian_numerical',
    'compute_full_hessian',
    'analyze_hessian',
    'classify_critical_point',
    'classify_all_critical_points',
    'compute_stability_basin',
    'compute_escape_directions',
    'compute_approach_directions',
    'estimate_local_timescales',

    # Bifurcations
    'detect_eigenvalue_zero_crossing',
    'classify_bifurcation',
    'scan_for_bifurcations',
    'refine_bifurcation_point',
    'compute_eigenvalue_spectrum_along_branch',
    'detect_bifurcations_from_eigenvalues',

    # Visualization
    'plot_energy_landscape_2d',
    'plot_energy_landscape_3d',
    'plot_gradient_norm_field',
    'plot_stability_basin',
    'plot_bifurcation_diagram',
    'plot_eigenvalue_spectrum',
    'plot_critical_point_summary',
    'create_animation_frames',

    # Convenience
    'quick_scan',
    'quick_bifurcation_scan',

    # State management
    'extract_system_state',
    'set_system_state',

    # Constants
    'CP_COLORS',
    'CP_MARKERS',
]
