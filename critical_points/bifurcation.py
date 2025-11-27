"""
Bifurcation Detection and Analysis
==================================

Detects and classifies bifurcations - qualitative changes in system behavior
as control parameters vary.

Common bifurcation types:
1. Saddle-node: Two critical points collide and annihilate
2. Pitchfork: One critical point splits into three
3. Transcritical: Two critical points exchange stability
4. Hopf: Stable point becomes unstable with oscillations

Detection strategies:
- Track eigenvalue signs through parameter space
- Monitor number of critical points
- Detect when gradient norm minimum approaches zero

Author: Claude
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .detector import (
    CriticalPoint, CriticalPointType, CriticalPointScan,
    compute_gradient_norm, extract_system_state, set_system_state,
    find_critical_point_gradient_descent, track_critical_points_under_parameter_change
)
from .stability import (
    HessianAnalysis, compute_hessian_numerical, analyze_hessian,
    compute_full_hessian, classify_critical_point
)
from gradients.free_energy_clean import compute_total_free_energy


class BifurcationType(Enum):
    """Types of bifurcations."""
    SADDLE_NODE = "saddle_node"          # Two points collide and vanish
    PITCHFORK = "pitchfork"              # One point splits into three
    TRANSCRITICAL = "transcritical"       # Points cross and exchange stability
    HOPF = "hopf"                         # Stable -> oscillatory
    FOLD = "fold"                         # Same as saddle-node
    PERIOD_DOUBLING = "period_doubling"  # For discrete dynamics
    UNKNOWN = "unknown"


@dataclass
class BifurcationEvent:
    """
    Represents a detected bifurcation.

    Attributes:
        param_name: Name of the control parameter
        param_value: Value at which bifurcation occurs
        type: Classification of bifurcation type
        critical_points_before: Critical points just before bifurcation
        critical_points_after: Critical points just after bifurcation
        eigenvalue_crossing: Eigenvalue(s) that cross zero
        metadata: Additional information
    """
    param_name: str
    param_value: float
    type: BifurcationType
    critical_points_before: List[CriticalPoint] = field(default_factory=list)
    critical_points_after: List[CriticalPoint] = field(default_factory=list)
    eigenvalue_crossing: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __str__(self):
        n_before = len(self.critical_points_before)
        n_after = len(self.critical_points_after)
        return (f"Bifurcation({self.type.value}) at {self.param_name}={self.param_value:.4f}: "
                f"{n_before} -> {n_after} critical points")


@dataclass
class BifurcationDiagram:
    """
    Complete bifurcation diagram for a parameter range.

    Attributes:
        param_name: Name of scanned parameter
        param_values: Array of parameter values
        branches: List of branches, each a list of (param, critical_point) pairs
        bifurcations: List of detected bifurcation events
        metadata: Additional information
    """
    param_name: str
    param_values: np.ndarray
    branches: List[List[Tuple[float, CriticalPoint]]]
    bifurcations: List[BifurcationEvent]
    metadata: Dict = field(default_factory=dict)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    @property
    def n_bifurcations(self) -> int:
        return len(self.bifurcations)


def detect_eigenvalue_zero_crossing(
    eigenvalues_before: np.ndarray,
    eigenvalues_after: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, Optional[int]]:
    """
    Detect if any eigenvalue crossed zero between two parameter values.

    Args:
        eigenvalues_before: Eigenvalues at parameter value λ
        eigenvalues_after: Eigenvalues at parameter value λ + δλ
        tolerance: Threshold for zero

    Returns:
        (crossed, idx): Whether crossing occurred and which eigenvalue
    """
    # Check for sign changes
    for i in range(min(len(eigenvalues_before), len(eigenvalues_after))):
        ev_before = eigenvalues_before[i]
        ev_after = eigenvalues_after[i]

        # Sign change indicates zero crossing
        if ev_before * ev_after < 0:
            return True, i

        # Close to zero on both sides
        if abs(ev_before) < tolerance and abs(ev_after) < tolerance:
            return True, i

    return False, None


def classify_bifurcation(
    critical_points_before: List[CriticalPoint],
    critical_points_after: List[CriticalPoint],
    eigenvalue_crossing_idx: Optional[int] = None
) -> BifurcationType:
    """
    Classify the type of bifurcation based on critical point changes.

    Args:
        critical_points_before: Critical points before bifurcation
        critical_points_after: Critical points after bifurcation
        eigenvalue_crossing_idx: Which eigenvalue crossed zero (if known)

    Returns:
        BifurcationType classification
    """
    n_before = len(critical_points_before)
    n_after = len(critical_points_after)

    # Count by stability
    n_stable_before = sum(1 for cp in critical_points_before if cp.is_minimum)
    n_stable_after = sum(1 for cp in critical_points_after if cp.is_minimum)
    n_saddle_before = sum(1 for cp in critical_points_before if cp.is_saddle)
    n_saddle_after = sum(1 for cp in critical_points_after if cp.is_saddle)

    # Saddle-node: stable + saddle -> nothing (or vice versa)
    if n_before == 2 and n_after == 0:
        if n_stable_before == 1 and n_saddle_before == 1:
            return BifurcationType.SADDLE_NODE
    if n_before == 0 and n_after == 2:
        if n_stable_after == 1 and n_saddle_after == 1:
            return BifurcationType.SADDLE_NODE

    # Pitchfork: 1 -> 3 or 3 -> 1
    if n_before == 1 and n_after == 3:
        return BifurcationType.PITCHFORK
    if n_before == 3 and n_after == 1:
        return BifurcationType.PITCHFORK

    # Transcritical: 2 -> 2 with stability exchange
    if n_before == 2 and n_after == 2:
        if n_stable_before != n_stable_after:
            return BifurcationType.TRANSCRITICAL

    # If we have eigenvalue info, check for Hopf
    # (Hopf: pair of complex conjugate eigenvalues cross imaginary axis)
    # For now, we only detect real eigenvalue crossings

    return BifurcationType.UNKNOWN


def scan_for_bifurcations(
    system,
    param_name: str,
    param_range: Tuple[float, float],
    n_points: int = 50,
    agent_idx: int = 0,
    verbose: bool = False
) -> BifurcationDiagram:
    """
    Scan parameter space to detect bifurcations.

    Strategy:
    1. For each parameter value, find all critical points
    2. Track how critical points move with parameter
    3. Detect when number or stability of points changes

    Args:
        system: MultiAgentSystem
        param_name: Parameter to scan (e.g., 'lambda_prior_align')
        param_range: (min, max) parameter values
        n_points: Number of parameter values to sample
        agent_idx: Which agent to analyze for classification
        verbose: Print progress

    Returns:
        BifurcationDiagram with branches and bifurcation events
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)

    # Save original parameter value
    original_value = getattr(system.config, param_name)
    initial_state = extract_system_state(system)

    # Track branches and bifurcations
    branches = []
    bifurcations = []
    prev_critical_points = []

    if verbose:
        print(f"Scanning {param_name} from {param_range[0]} to {param_range[1]}")
        print(f"{'='*60}")

    for i, param_val in enumerate(param_values):
        # Set parameter
        setattr(system.config, param_name, param_val)

        # Reset to initial state and find critical points
        set_system_state(system, initial_state)

        # Find critical points with multiple restarts
        from .detector import find_critical_points_random_restarts
        critical_points = find_critical_points_random_restarts(
            system,
            n_restarts=5,
            perturbation_scale=0.3,
            max_steps=200,
            learning_rate=0.02,
            verbose=False
        )

        # Classify each critical point
        for cp in critical_points:
            try:
                classify_critical_point(system, cp, agent_idx=agent_idx)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not classify CP at {param_val:.4f}: {e}")

        if verbose:
            n_stable = sum(1 for cp in critical_points if cp.is_minimum)
            n_saddle = sum(1 for cp in critical_points if cp.is_saddle)
            print(f"{param_name}={param_val:.4f}: {len(critical_points)} CPs "
                  f"(stable={n_stable}, saddle={n_saddle})")

        # Check for bifurcation
        if i > 0 and len(prev_critical_points) != len(critical_points):
            bif_type = classify_bifurcation(prev_critical_points, critical_points)

            bifurcation = BifurcationEvent(
                param_name=param_name,
                param_value=0.5 * (param_values[i - 1] + param_val),  # Midpoint estimate
                type=bif_type,
                critical_points_before=prev_critical_points.copy(),
                critical_points_after=critical_points.copy(),
                metadata={'param_before': param_values[i - 1], 'param_after': param_val}
            )
            bifurcations.append(bifurcation)

            if verbose:
                print(f"  *** {bifurcation}")

        # Store for branch tracking
        for cp in critical_points:
            branches.append([(param_val, cp)])  # Simple: each point is its own "branch"

        prev_critical_points = critical_points

    # Restore original parameter
    setattr(system.config, param_name, original_value)
    set_system_state(system, initial_state)

    return BifurcationDiagram(
        param_name=param_name,
        param_values=param_values,
        branches=branches,
        bifurcations=bifurcations,
        metadata={'param_range': param_range, 'n_points': n_points}
    )


def refine_bifurcation_point(
    system,
    bifurcation: BifurcationEvent,
    tolerance: float = 0.001,
    max_iterations: int = 20,
    verbose: bool = False
) -> BifurcationEvent:
    """
    Refine the location of a bifurcation using bisection.

    Args:
        system: MultiAgentSystem
        bifurcation: Initial bifurcation estimate
        tolerance: Desired accuracy in parameter value
        max_iterations: Maximum bisection iterations
        verbose: Print progress

    Returns:
        Refined BifurcationEvent with more accurate parameter value
    """
    param_name = bifurcation.param_name
    param_low = bifurcation.metadata.get('param_before', bifurcation.param_value - 0.1)
    param_high = bifurcation.metadata.get('param_after', bifurcation.param_value + 0.1)

    n_before = len(bifurcation.critical_points_before)

    # Save state
    original_value = getattr(system.config, param_name)
    initial_state = extract_system_state(system)

    if verbose:
        print(f"Refining bifurcation in [{param_low:.4f}, {param_high:.4f}]")

    for iteration in range(max_iterations):
        param_mid = 0.5 * (param_low + param_high)

        # Find critical points at midpoint
        setattr(system.config, param_name, param_mid)
        set_system_state(system, initial_state)

        from .detector import find_critical_points_random_restarts
        critical_points = find_critical_points_random_restarts(
            system,
            n_restarts=5,
            perturbation_scale=0.3,
            max_steps=200,
            verbose=False
        )

        n_mid = len(critical_points)

        if verbose:
            print(f"  Iteration {iteration}: [{param_low:.4f}, {param_high:.4f}] "
                  f"-> mid={param_mid:.4f}, n_cp={n_mid}")

        # Bisection logic
        if n_mid == n_before:
            param_low = param_mid
        else:
            param_high = param_mid

        # Check convergence
        if param_high - param_low < tolerance:
            break

    # Update bifurcation with refined value
    bifurcation.param_value = 0.5 * (param_low + param_high)
    bifurcation.metadata['refined'] = True
    bifurcation.metadata['refinement_iterations'] = iteration + 1
    bifurcation.metadata['param_range_final'] = (param_low, param_high)

    # Restore
    setattr(system.config, param_name, original_value)
    set_system_state(system, initial_state)

    if verbose:
        print(f"  Refined bifurcation: {bifurcation.param_name} = {bifurcation.param_value:.6f}")

    return bifurcation


def compute_eigenvalue_spectrum_along_branch(
    system,
    branch: List[Tuple[float, CriticalPoint]],
    agent_idx: int = 0,
    include_priors: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hessian eigenvalues along a branch of critical points.

    This reveals how stability changes along the branch and identifies
    potential bifurcation points where eigenvalues cross zero.

    Args:
        system: MultiAgentSystem
        branch: List of (param_value, CriticalPoint) pairs
        agent_idx: Which agent for Hessian computation
        include_priors: Include prior parameters

    Returns:
        (param_values, eigenvalue_matrix): Parameter values and eigenvalues
    """
    param_values = []
    all_eigenvalues = []

    for param_val, cp in branch:
        param_values.append(param_val)

        # Set system to critical point
        set_system_state(system, cp.location)

        # Compute Hessian
        hessian, _ = compute_full_hessian(
            system, agent_idx,
            include_priors=include_priors,
            include_gauge=False
        )

        # Analyze
        analysis = analyze_hessian(hessian)
        all_eigenvalues.append(analysis.eigenvalues)

    # Pad eigenvalue arrays to same length
    max_len = max(len(ev) for ev in all_eigenvalues)
    eigenvalue_matrix = np.full((len(branch), max_len), np.nan)

    for i, ev in enumerate(all_eigenvalues):
        eigenvalue_matrix[i, :len(ev)] = ev

    return np.array(param_values), eigenvalue_matrix


def detect_bifurcations_from_eigenvalues(
    param_values: np.ndarray,
    eigenvalue_matrix: np.ndarray,
    zero_threshold: float = 0.01
) -> List[Dict]:
    """
    Detect bifurcations by finding eigenvalue zero crossings.

    Args:
        param_values: Array of parameter values
        eigenvalue_matrix: (n_params, n_eigenvalues) matrix
        zero_threshold: Threshold for detecting zero crossing

    Returns:
        List of dicts with bifurcation information
    """
    bifurcations = []
    n_params, n_eigs = eigenvalue_matrix.shape

    for eig_idx in range(n_eigs):
        eigenvalues = eigenvalue_matrix[:, eig_idx]

        # Skip if all NaN
        if np.all(np.isnan(eigenvalues)):
            continue

        # Find sign changes
        for i in range(1, n_params):
            ev_before = eigenvalues[i - 1]
            ev_after = eigenvalues[i]

            if np.isnan(ev_before) or np.isnan(ev_after):
                continue

            # Check for zero crossing
            if ev_before * ev_after < 0:
                # Interpolate crossing location
                frac = abs(ev_before) / (abs(ev_before) + abs(ev_after))
                param_crossing = param_values[i - 1] + frac * (param_values[i] - param_values[i - 1])

                bifurcations.append({
                    'param_value': param_crossing,
                    'eigenvalue_idx': eig_idx,
                    'eigenvalue_before': ev_before,
                    'eigenvalue_after': ev_after,
                    'param_idx': i
                })

    return bifurcations


def compute_codimension_one_bifurcation_normal_form(
    system,
    bifurcation: BifurcationEvent,
    agent_idx: int = 0
) -> Dict:
    """
    Compute normal form coefficients for a codimension-1 bifurcation.

    For saddle-node bifurcation:
        dx/dt = r - x^2  (normal form)
        r_c = 0 is the bifurcation point

    We compute:
        - Linear coefficient a (should be small at bifurcation)
        - Quadratic coefficient b (determines unfolding)

    Args:
        system: MultiAgentSystem
        bifurcation: Bifurcation to analyze
        agent_idx: Which agent

    Returns:
        Dict with normal form coefficients
    """
    param_name = bifurcation.param_name
    param_value = bifurcation.param_value

    # Save state
    original_value = getattr(system.config, param_name)
    initial_state = extract_system_state(system)

    # Set to bifurcation parameter
    setattr(system.config, param_name, param_value)

    # Find critical point at bifurcation
    set_system_state(system, initial_state)
    from .detector import find_critical_point_gradient_descent

    cp = find_critical_point_gradient_descent(
        system,
        max_steps=500,
        learning_rate=0.01,
        verbose=False
    )

    normal_form = {'type': bifurcation.type.value}

    if cp is not None:
        # Compute Hessian at critical point
        set_system_state(system, cp.location)
        hessian, _ = compute_full_hessian(system, agent_idx)
        analysis = analyze_hessian(hessian)

        # Find the eigenvalue closest to zero
        min_abs_idx = np.argmin(np.abs(analysis.eigenvalues))
        critical_eigenvalue = analysis.eigenvalues[min_abs_idx]
        critical_direction = analysis.eigenvectors[:, min_abs_idx]

        normal_form['critical_eigenvalue'] = critical_eigenvalue
        normal_form['critical_direction'] = critical_direction
        normal_form['all_eigenvalues'] = analysis.eigenvalues

        # For saddle-node: need cubic term along critical direction
        # This would require computing third derivatives - simplified here

    # Restore
    setattr(system.config, param_name, original_value)
    set_system_state(system, initial_state)

    return normal_form
