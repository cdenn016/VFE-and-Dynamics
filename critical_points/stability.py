"""
Stability Analysis for Critical Points
======================================

Analyzes the stability of critical points via Hessian eigenvalue analysis.

For a critical point (∇F = 0), the Hessian H = ∇²F determines local behavior:
- All eigenvalues > 0: Stable (local minimum) - attractor
- All eigenvalues < 0: Unstable (local maximum) - repeller
- Mixed signs: Saddle point - transition state
- Zero eigenvalues: Degenerate - potential bifurcation

The stability index (Morse index) = number of negative eigenvalues.

Author: Claude
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from .detector import (
        CriticalPoint, CriticalPointType, CriticalPointScan,
        compute_gradient_norm, extract_system_state, set_system_state
    )
except ImportError:
    from detector import (
        CriticalPoint, CriticalPointType, CriticalPointScan,
        compute_gradient_norm, extract_system_state, set_system_state
    )
from gradients.free_energy_clean import compute_total_free_energy


@dataclass
class HessianAnalysis:
    """
    Results of Hessian analysis at a critical point.

    Attributes:
        eigenvalues: Sorted eigenvalues of the Hessian
        eigenvectors: Corresponding eigenvectors
        n_positive: Count of positive eigenvalues
        n_negative: Count of negative eigenvalues (stability index)
        n_zero: Count of zero eigenvalues (degeneracy)
        condition_number: Ratio of largest to smallest |eigenvalue|
        is_degenerate: Has zero eigenvalues
        stability_type: Classification based on eigenvalue signs
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_positive: int
    n_negative: int
    n_zero: int
    condition_number: float
    is_degenerate: bool
    stability_type: CriticalPointType

    @property
    def stability_index(self) -> int:
        """Morse index = number of negative eigenvalues."""
        return self.n_negative

    def __str__(self):
        sign_str = f"(−{self.n_negative}, 0×{self.n_zero}, +{self.n_positive})"
        return (f"HessianAnalysis: {self.stability_type.value}, "
                f"eigenvalue signs: {sign_str}, "
                f"κ = {self.condition_number:.2e}")


def compute_hessian_numerical(
    system,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Optional[List[int]] = None,
    epsilon: float = 1e-4
) -> np.ndarray:
    """
    Compute the Hessian ∇²F numerically via finite differences.

    Uses central differences:
        H_ij ≈ [F(θ + ε_i + ε_j) - F(θ + ε_i - ε_j)
                - F(θ - ε_i + ε_j) + F(θ - ε_i - ε_j)] / (4ε²)

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent's parameters
        param: Parameter name ('mu_q', 'mu_p', 'Sigma_q', 'Sigma_p')
        dims: Which dimensions to include (None = all)
        epsilon: Finite difference step size

    Returns:
        hessian: (n_dims, n_dims) Hessian matrix
    """
    agent = system.agents[agent_idx]

    # Get parameter and determine dimensions
    if param == 'mu_q':
        theta = agent.mu_q
    elif param == 'mu_p':
        theta = agent.mu_p
    else:
        raise ValueError(f"Hessian for {param} not yet implemented")

    if dims is None:
        dims = list(range(len(theta.flatten())))

    n_dims = len(dims)
    hessian = np.zeros((n_dims, n_dims))

    # Save original state
    original = theta.copy()

    for i, dim_i in enumerate(dims):
        for j, dim_j in enumerate(dims):
            if j < i:
                # Hessian is symmetric
                hessian[i, j] = hessian[j, i]
                continue

            # F(θ + ε_i + ε_j)
            theta_pp = original.copy()
            theta_pp.flat[dim_i] += epsilon
            theta_pp.flat[dim_j] += epsilon
            if param == 'mu_q':
                agent.mu_q = theta_pp
            else:
                agent.mu_p = theta_pp
            F_pp = compute_total_free_energy(system).total

            # F(θ + ε_i - ε_j)
            theta_pm = original.copy()
            theta_pm.flat[dim_i] += epsilon
            theta_pm.flat[dim_j] -= epsilon
            if param == 'mu_q':
                agent.mu_q = theta_pm
            else:
                agent.mu_p = theta_pm
            F_pm = compute_total_free_energy(system).total

            # F(θ - ε_i + ε_j)
            theta_mp = original.copy()
            theta_mp.flat[dim_i] -= epsilon
            theta_mp.flat[dim_j] += epsilon
            if param == 'mu_q':
                agent.mu_q = theta_mp
            else:
                agent.mu_p = theta_mp
            F_mp = compute_total_free_energy(system).total

            # F(θ - ε_i - ε_j)
            theta_mm = original.copy()
            theta_mm.flat[dim_i] -= epsilon
            theta_mm.flat[dim_j] -= epsilon
            if param == 'mu_q':
                agent.mu_q = theta_mm
            else:
                agent.mu_p = theta_mm
            F_mm = compute_total_free_energy(system).total

            # Central difference formula
            hessian[i, j] = (F_pp - F_pm - F_mp + F_mm) / (4 * epsilon ** 2)

    # Restore original parameter
    if param == 'mu_q':
        agent.mu_q = original
    else:
        agent.mu_p = original

    # Ensure symmetry
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


def compute_full_hessian(
    system,
    agent_idx: int = 0,
    include_priors: bool = True,
    include_gauge: bool = False,
    epsilon: float = 1e-4
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """
    Compute the full Hessian over all belief parameters.

    Parameters are concatenated as: [mu_q, vec(Sigma_q), mu_p, vec(Sigma_p), phi]

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent
        include_priors: Include prior parameters
        include_gauge: Include gauge field
        epsilon: Finite difference step

    Returns:
        (hessian, param_slices): Full Hessian and dict mapping param names to slices
    """
    agent = system.agents[agent_idx]
    K = agent.K

    # Build parameter vector and track slices
    param_slices = {}
    params = []
    current_idx = 0

    # mu_q
    params.append(agent.mu_q.flatten())
    param_slices['mu_q'] = slice(current_idx, current_idx + K)
    current_idx += K

    # Sigma_q (lower triangular to respect symmetry)
    L_q_indices = np.tril_indices(K)
    Sigma_q_params = agent.Sigma_q[L_q_indices]
    params.append(Sigma_q_params)
    n_sigma = len(Sigma_q_params)
    param_slices['Sigma_q'] = slice(current_idx, current_idx + n_sigma)
    current_idx += n_sigma

    if include_priors:
        # mu_p
        params.append(agent.mu_p.flatten())
        param_slices['mu_p'] = slice(current_idx, current_idx + K)
        current_idx += K

        # Sigma_p (lower triangular)
        Sigma_p_params = agent.Sigma_p[L_q_indices]
        params.append(Sigma_p_params)
        param_slices['Sigma_p'] = slice(current_idx, current_idx + n_sigma)
        current_idx += n_sigma

    if include_gauge and hasattr(agent, 'gauge'):
        params.append(agent.gauge.phi.flatten())
        param_slices['phi'] = slice(current_idx, current_idx + 3)
        current_idx += 3

    theta = np.concatenate(params)
    n_params = len(theta)

    # Compute Hessian via finite differences
    hessian = np.zeros((n_params, n_params))

    def set_params(theta_new):
        """Helper to set parameters from vector."""
        agent.mu_q = theta_new[param_slices['mu_q']].reshape(agent.mu_q.shape)

        Sigma_q_flat = theta_new[param_slices['Sigma_q']]
        Sigma_q_new = np.zeros((K, K))
        Sigma_q_new[L_q_indices] = Sigma_q_flat
        Sigma_q_new = Sigma_q_new + Sigma_q_new.T - np.diag(np.diag(Sigma_q_new))
        agent.Sigma_q = Sigma_q_new

        if include_priors:
            agent.mu_p = theta_new[param_slices['mu_p']].reshape(agent.mu_p.shape)

            Sigma_p_flat = theta_new[param_slices['Sigma_p']]
            Sigma_p_new = np.zeros((K, K))
            Sigma_p_new[L_q_indices] = Sigma_p_flat
            Sigma_p_new = Sigma_p_new + Sigma_p_new.T - np.diag(np.diag(Sigma_p_new))
            agent.Sigma_p = Sigma_p_new

        if include_gauge and 'phi' in param_slices and hasattr(agent, 'gauge'):
            agent.gauge.phi = theta_new[param_slices['phi']].reshape(agent.gauge.phi.shape)

    def get_energy(theta_new):
        """Compute energy at parameter vector."""
        set_params(theta_new)
        return compute_total_free_energy(system).total

    # Compute Hessian entries
    for i in range(n_params):
        for j in range(i, n_params):
            # F(θ + ε_i + ε_j)
            theta_pp = theta.copy()
            theta_pp[i] += epsilon
            theta_pp[j] += epsilon
            F_pp = get_energy(theta_pp)

            # F(θ + ε_i - ε_j)
            theta_pm = theta.copy()
            theta_pm[i] += epsilon
            theta_pm[j] -= epsilon
            F_pm = get_energy(theta_pm)

            # F(θ - ε_i + ε_j)
            theta_mp = theta.copy()
            theta_mp[i] -= epsilon
            theta_mp[j] += epsilon
            F_mp = get_energy(theta_mp)

            # F(θ - ε_i - ε_j)
            theta_mm = theta.copy()
            theta_mm[i] -= epsilon
            theta_mm[j] -= epsilon
            F_mm = get_energy(theta_mm)

            hessian[i, j] = (F_pp - F_pm - F_mp + F_mm) / (4 * epsilon ** 2)
            hessian[j, i] = hessian[i, j]

    # Restore original parameters
    set_params(theta)

    return hessian, param_slices


def analyze_hessian(
    hessian: np.ndarray,
    zero_threshold: float = 1e-6
) -> HessianAnalysis:
    """
    Analyze Hessian eigenvalues to classify stability.

    Args:
        hessian: (n, n) symmetric Hessian matrix
        zero_threshold: Threshold for zero eigenvalues

    Returns:
        HessianAnalysis with full eigenvalue decomposition
    """
    # Ensure symmetry
    hessian = 0.5 * (hessian + hessian.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Count signs
    n_positive = np.sum(eigenvalues > zero_threshold)
    n_negative = np.sum(eigenvalues < -zero_threshold)
    n_zero = np.sum(np.abs(eigenvalues) <= zero_threshold)

    # Condition number
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) > zero_threshold]
    if len(nonzero_eigs) > 0:
        condition_number = np.max(np.abs(nonzero_eigs)) / np.min(np.abs(nonzero_eigs))
    else:
        condition_number = np.inf

    # Classify stability
    is_degenerate = n_zero > 0

    if is_degenerate:
        stability_type = CriticalPointType.DEGENERATE
    elif n_negative == 0:
        stability_type = CriticalPointType.STABLE
    elif n_positive == 0:
        stability_type = CriticalPointType.UNSTABLE
    else:
        stability_type = CriticalPointType.SADDLE

    return HessianAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        condition_number=condition_number,
        is_degenerate=is_degenerate,
        stability_type=stability_type
    )


def classify_critical_point(
    system,
    critical_point: CriticalPoint,
    agent_idx: int = 0,
    include_priors: bool = True,
    include_gauge: bool = False,
    epsilon: float = 1e-4
) -> CriticalPoint:
    """
    Classify a critical point by computing its Hessian.

    Updates the CriticalPoint object with stability information.

    Args:
        system: MultiAgentSystem
        critical_point: CriticalPoint to classify
        agent_idx: Which agent to analyze
        include_priors: Include prior parameters in Hessian
        include_gauge: Include gauge field in Hessian
        epsilon: Finite difference step

    Returns:
        Updated CriticalPoint with type and eigenvalues
    """
    # Set system to critical point state
    set_system_state(system, critical_point.location)

    # Compute Hessian
    hessian, param_slices = compute_full_hessian(
        system, agent_idx,
        include_priors=include_priors,
        include_gauge=include_gauge,
        epsilon=epsilon
    )

    # Analyze Hessian
    analysis = analyze_hessian(hessian)

    # Update critical point
    critical_point.type = analysis.stability_type
    critical_point.hessian_eigenvalues = analysis.eigenvalues
    critical_point.stability_index = analysis.n_negative
    critical_point.metadata['hessian_analysis'] = {
        'n_positive': analysis.n_positive,
        'n_negative': analysis.n_negative,
        'n_zero': analysis.n_zero,
        'condition_number': analysis.condition_number,
        'param_slices': {k: (v.start, v.stop) for k, v in param_slices.items()}
    }

    return critical_point


def classify_all_critical_points(
    system,
    scan_result: CriticalPointScan,
    **kwargs
) -> CriticalPointScan:
    """
    Classify all critical points in a scan result.

    Args:
        system: MultiAgentSystem
        scan_result: CriticalPointScan to classify
        **kwargs: Arguments passed to classify_critical_point

    Returns:
        Updated CriticalPointScan with classified points
    """
    for cp in scan_result.critical_points:
        classify_critical_point(system, cp, **kwargs)

    return scan_result


def compute_stability_basin(
    system,
    critical_point: CriticalPoint,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    grid_resolution: int = 20,
    convergence_threshold: float = 0.1,
    max_steps: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the basin of attraction of a stable critical point.

    For each grid point, run gradient descent and check if it converges
    to the given critical point.

    Args:
        system: MultiAgentSystem
        critical_point: Target stable critical point
        agent_idx: Which agent
        param: Parameter to vary
        dims: Which dimensions to scan
        grid_range: (min, max) range
        grid_resolution: Grid points per dimension
        convergence_threshold: Distance threshold for convergence
        max_steps: Max gradient descent steps per point

    Returns:
        (X, Y, basin_mask): Meshgrid and boolean mask of basin
    """
    from .detector import find_critical_point_gradient_descent

    if critical_point.type != CriticalPointType.STABLE:
        print("Warning: Computing basin for non-stable critical point")

    # Save initial state
    initial_state = extract_system_state(system)
    target_state = critical_point.location[agent_idx]

    # Create grid
    x = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    y = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    X, Y = np.meshgrid(x, y)
    basin_mask = np.zeros_like(X, dtype=bool)

    agent = system.agents[agent_idx]

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Set starting point
            set_system_state(system, initial_state)

            if param == 'mu_q':
                agent.mu_q[dims[0]] = X[i, j]
                agent.mu_q[dims[1]] = Y[i, j]
            elif param == 'mu_p':
                agent.mu_p[dims[0]] = X[i, j]
                agent.mu_p[dims[1]] = Y[i, j]

            # Run gradient descent
            cp_found = find_critical_point_gradient_descent(
                system,
                max_steps=max_steps,
                learning_rate=0.01,
                gradient_threshold=1e-4,
                verbose=False
            )

            if cp_found is not None:
                # Check if converged to target
                if param == 'mu_q':
                    final_val = agent.mu_q.copy()
                    target_val = target_state['mu_q']
                else:
                    final_val = agent.mu_p.copy()
                    target_val = target_state['mu_p']

                distance = np.linalg.norm(final_val - target_val)
                basin_mask[i, j] = distance < convergence_threshold

    # Restore initial state
    set_system_state(system, initial_state)

    return X, Y, basin_mask


def compute_escape_directions(
    hessian_analysis: HessianAnalysis
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify escape directions from a saddle point.

    For a saddle point, the eigenvectors corresponding to negative eigenvalues
    point in directions where the energy decreases (escape routes).

    Args:
        hessian_analysis: HessianAnalysis from analyze_hessian

    Returns:
        (escape_directions, eigenvalues): Eigenvectors and their eigenvalues
                                          for negative eigenvalue directions
    """
    eigenvalues = hessian_analysis.eigenvalues
    eigenvectors = hessian_analysis.eigenvectors

    # Find negative eigenvalue indices
    neg_mask = eigenvalues < 0
    escape_directions = eigenvectors[:, neg_mask]
    escape_eigenvalues = eigenvalues[neg_mask]

    return escape_directions, escape_eigenvalues


def compute_approach_directions(
    hessian_analysis: HessianAnalysis
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify stable approach directions to a critical point.

    For a saddle point, the eigenvectors corresponding to positive eigenvalues
    point in directions where the point is locally attractive.

    Args:
        hessian_analysis: HessianAnalysis from analyze_hessian

    Returns:
        (approach_directions, eigenvalues): Eigenvectors and their eigenvalues
                                            for positive eigenvalue directions
    """
    eigenvalues = hessian_analysis.eigenvalues
    eigenvectors = hessian_analysis.eigenvectors

    # Find positive eigenvalue indices
    pos_mask = eigenvalues > 0
    approach_directions = eigenvectors[:, pos_mask]
    approach_eigenvalues = eigenvalues[pos_mask]

    return approach_directions, approach_eigenvalues


def estimate_local_timescales(
    hessian_analysis: HessianAnalysis,
    friction: float = 1.0
) -> np.ndarray:
    """
    Estimate relaxation timescales from Hessian eigenvalues.

    For overdamped dynamics (gradient flow):
        τ_i ≈ 1 / λ_i  (where λ_i is the i-th eigenvalue)

    For underdamped dynamics with friction γ:
        τ_i ≈ γ / λ_i  (critically damped)

    Args:
        hessian_analysis: HessianAnalysis from analyze_hessian
        friction: Friction coefficient (for Hamiltonian dynamics)

    Returns:
        timescales: Array of relaxation timescales
    """
    eigenvalues = hessian_analysis.eigenvalues

    # Only compute for positive eigenvalues (stable directions)
    pos_eigs = eigenvalues[eigenvalues > 0]

    if len(pos_eigs) == 0:
        return np.array([np.inf])

    timescales = friction / pos_eigs

    return timescales
