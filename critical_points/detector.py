"""
Critical Points Detector
========================

Detects critical points in the VFE landscape where gradients vanish (∇F ≈ 0).

Critical points are fundamental to understanding:
1. Equilibrium states (stable fixed points)
2. Unstable fixed points (repellers)
3. Saddle points (transition states)
4. Bifurcations (how equilibria change with parameters)

Author: Claude
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from gradients.gradient_engine import compute_natural_gradients, AgentGradients
from gradients.free_energy_clean import compute_total_free_energy


class CriticalPointType(Enum):
    """Classification of critical point types based on Hessian eigenvalues."""
    STABLE = "stable"              # All eigenvalues positive (minimum)
    UNSTABLE = "unstable"          # All eigenvalues negative (maximum)
    SADDLE = "saddle"              # Mixed signs (saddle point)
    DEGENERATE = "degenerate"      # Has zero eigenvalues
    CENTER = "center"              # Pure imaginary eigenvalues (conservative system)
    UNKNOWN = "unknown"


@dataclass
class CriticalPoint:
    """
    Represents a detected critical point in the VFE landscape.

    Attributes:
        location: Dictionary mapping agent_id -> (mu_q, Sigma_q, mu_p, Sigma_p, phi)
        gradient_norm: L2 norm of the total gradient (should be ≈ 0)
        energy: Free energy value F at this point
        type: Classification (stable, unstable, saddle, etc.)
        hessian_eigenvalues: Eigenvalues of the Hessian (if computed)
        stability_index: Number of negative eigenvalues (Morse index)
        detection_method: How this point was found
        metadata: Additional information
    """
    location: Dict[int, Dict[str, np.ndarray]]
    gradient_norm: float
    energy: float
    type: CriticalPointType = CriticalPointType.UNKNOWN
    hessian_eigenvalues: Optional[np.ndarray] = None
    stability_index: int = 0  # Number of negative eigenvalues
    detection_method: str = "gradient_descent"
    metadata: Dict = field(default_factory=dict)

    def __str__(self):
        return (f"CriticalPoint(type={self.type.value}, "
                f"energy={self.energy:.4f}, "
                f"||∇F||={self.gradient_norm:.2e}, "
                f"stability_index={self.stability_index})")

    @property
    def is_minimum(self) -> bool:
        """Check if this is a local minimum (all positive eigenvalues)."""
        return self.type == CriticalPointType.STABLE

    @property
    def is_maximum(self) -> bool:
        """Check if this is a local maximum (all negative eigenvalues)."""
        return self.type == CriticalPointType.UNSTABLE

    @property
    def is_saddle(self) -> bool:
        """Check if this is a saddle point (mixed signs)."""
        return self.type == CriticalPointType.SADDLE


@dataclass
class CriticalPointScan:
    """
    Results of a critical points scan.

    Attributes:
        critical_points: List of detected critical points
        scan_parameters: Parameters used for the scan
        gradient_field: Optional gradient norm field over parameter space
        energy_landscape: Optional energy values over parameter space
    """
    critical_points: List[CriticalPoint]
    scan_parameters: Dict
    gradient_field: Optional[np.ndarray] = None
    energy_landscape: Optional[np.ndarray] = None

    @property
    def n_minima(self) -> int:
        return sum(1 for cp in self.critical_points if cp.is_minimum)

    @property
    def n_maxima(self) -> int:
        return sum(1 for cp in self.critical_points if cp.is_maximum)

    @property
    def n_saddles(self) -> int:
        return sum(1 for cp in self.critical_points if cp.is_saddle)

    def get_global_minimum(self) -> Optional[CriticalPoint]:
        """Return the critical point with lowest energy."""
        minima = [cp for cp in self.critical_points if cp.is_minimum]
        if not minima:
            return None
        return min(minima, key=lambda cp: cp.energy)


def compute_gradient_norm(system) -> Tuple[float, List[AgentGradients]]:
    """
    Compute the total gradient norm across all agents.

    Returns:
        (total_norm, gradients): Total L2 norm and list of gradients per agent
    """
    gradients = compute_natural_gradients(system, n_jobs=1)

    total_norm_sq = 0.0

    for grads in gradients:
        # Sum of squared norms for each parameter
        if grads.delta_mu_q is not None:
            total_norm_sq += np.sum(grads.delta_mu_q ** 2)
        if grads.delta_Sigma_q is not None:
            total_norm_sq += np.sum(grads.delta_Sigma_q ** 2)
        if grads.delta_mu_p is not None:
            total_norm_sq += np.sum(grads.delta_mu_p ** 2)
        if grads.delta_Sigma_p is not None:
            total_norm_sq += np.sum(grads.delta_Sigma_p ** 2)
        if grads.delta_phi is not None:
            total_norm_sq += np.sum(grads.delta_phi ** 2)

    return np.sqrt(total_norm_sq), gradients


def compute_per_agent_gradient_norms(gradients: List[AgentGradients]) -> Dict[int, float]:
    """
    Compute gradient norm for each agent separately.

    Args:
        gradients: List of AgentGradients from compute_natural_gradients

    Returns:
        Dict mapping agent_id -> gradient norm
    """
    norms = {}

    for i, grads in enumerate(gradients):
        norm_sq = 0.0

        if grads.delta_mu_q is not None:
            norm_sq += np.sum(grads.delta_mu_q ** 2)
        if grads.delta_Sigma_q is not None:
            norm_sq += np.sum(grads.delta_Sigma_q ** 2)
        if grads.delta_mu_p is not None:
            norm_sq += np.sum(grads.delta_mu_p ** 2)
        if grads.delta_Sigma_p is not None:
            norm_sq += np.sum(grads.delta_Sigma_p ** 2)
        if grads.delta_phi is not None:
            norm_sq += np.sum(grads.delta_phi ** 2)

        norms[i] = np.sqrt(norm_sq)

    return norms


def extract_system_state(system) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract current state of all agents.

    Returns:
        Dict mapping agent_id -> dict with 'mu_q', 'Sigma_q', 'mu_p', 'Sigma_p', 'phi'
    """
    state = {}

    for i, agent in enumerate(system.agents):
        state[i] = {
            'mu_q': agent.mu_q.copy(),
            'Sigma_q': agent.Sigma_q.copy(),
            'mu_p': agent.mu_p.copy(),
            'Sigma_p': agent.Sigma_p.copy(),
            'phi': agent.gauge.phi.copy() if hasattr(agent, 'gauge') else np.zeros(3)
        }

    return state


def set_system_state(system, state: Dict[int, Dict[str, np.ndarray]]):
    """
    Set system state from a saved state dictionary.

    Args:
        system: MultiAgentSystem to modify
        state: State dictionary from extract_system_state
    """
    for i, agent_state in state.items():
        if i < len(system.agents):
            agent = system.agents[i]
            agent.mu_q = agent_state['mu_q'].copy()
            agent.Sigma_q = agent_state['Sigma_q'].copy()
            agent.mu_p = agent_state['mu_p'].copy()
            agent.Sigma_p = agent_state['Sigma_p'].copy()
            if hasattr(agent, 'gauge') and 'phi' in agent_state:
                agent.gauge.phi = agent_state['phi'].copy()


def detect_critical_point_at_state(
    system,
    gradient_threshold: float = 1e-4
) -> Optional[CriticalPoint]:
    """
    Check if the current system state is a critical point.

    A critical point has ||∇F|| < threshold.

    Args:
        system: MultiAgentSystem at current state
        gradient_threshold: Maximum gradient norm to consider critical

    Returns:
        CriticalPoint if gradient is small enough, None otherwise
    """
    # Compute gradient norm
    grad_norm, gradients = compute_gradient_norm(system)

    if grad_norm > gradient_threshold:
        return None

    # Extract state
    location = extract_system_state(system)

    # Compute energy
    energy_breakdown = compute_total_free_energy(system)
    energy = energy_breakdown.total

    return CriticalPoint(
        location=location,
        gradient_norm=grad_norm,
        energy=energy,
        type=CriticalPointType.UNKNOWN,  # Will be classified by stability analysis
        detection_method="current_state"
    )


def find_critical_point_gradient_descent(
    system,
    max_steps: int = 1000,
    learning_rate: float = 0.01,
    gradient_threshold: float = 1e-5,
    energy_tolerance: float = 1e-8,
    verbose: bool = False
) -> Optional[CriticalPoint]:
    """
    Find a critical point by gradient descent from current state.

    This finds a LOCAL MINIMUM by following the negative gradient.

    Args:
        system: MultiAgentSystem
        max_steps: Maximum gradient descent steps
        learning_rate: Step size
        gradient_threshold: Convergence criterion for gradient norm
        energy_tolerance: Convergence criterion for energy change
        verbose: Print progress

    Returns:
        CriticalPoint if converged, None otherwise
    """
    prev_energy = float('inf')

    for step in range(max_steps):
        # Compute gradients
        grad_norm, gradients = compute_gradient_norm(system)

        # Compute energy
        energy_breakdown = compute_total_free_energy(system)
        energy = energy_breakdown.total

        if verbose and step % 100 == 0:
            print(f"Step {step}: ||∇F|| = {grad_norm:.2e}, F = {energy:.4f}")

        # Check convergence
        if grad_norm < gradient_threshold:
            if verbose:
                print(f"Converged at step {step}: ||∇F|| = {grad_norm:.2e}")

            return CriticalPoint(
                location=extract_system_state(system),
                gradient_norm=grad_norm,
                energy=energy,
                type=CriticalPointType.STABLE,  # Gradient descent finds minima
                detection_method="gradient_descent",
                metadata={'steps': step}
            )

        # Check energy plateau
        if abs(energy - prev_energy) < energy_tolerance:
            if verbose:
                print(f"Energy plateau at step {step}: ΔF = {abs(energy - prev_energy):.2e}")

            return CriticalPoint(
                location=extract_system_state(system),
                gradient_norm=grad_norm,
                energy=energy,
                type=CriticalPointType.STABLE,
                detection_method="gradient_descent",
                metadata={'steps': step, 'converged_by': 'energy_plateau'}
            )

        prev_energy = energy

        # Update parameters (gradient descent)
        for i, (agent, grads) in enumerate(zip(system.agents, gradients)):
            if grads.delta_mu_q is not None:
                agent.mu_q -= learning_rate * grads.delta_mu_q
            if grads.delta_Sigma_q is not None:
                # Project to ensure positive definiteness
                agent.Sigma_q -= learning_rate * grads.delta_Sigma_q
                agent.Sigma_q = 0.5 * (agent.Sigma_q + agent.Sigma_q.T)
                eigvals = np.linalg.eigvalsh(agent.Sigma_q)
                if eigvals.min() < 1e-6:
                    agent.Sigma_q += (1e-6 - eigvals.min()) * np.eye(agent.K)

            # Prior updates (if enabled)
            if grads.delta_mu_p is not None:
                agent.mu_p -= learning_rate * grads.delta_mu_p
            if grads.delta_Sigma_p is not None:
                agent.Sigma_p -= learning_rate * grads.delta_Sigma_p
                agent.Sigma_p = 0.5 * (agent.Sigma_p + agent.Sigma_p.T)
                eigvals = np.linalg.eigvalsh(agent.Sigma_p)
                if eigvals.min() < 1e-6:
                    agent.Sigma_p += (1e-6 - eigvals.min()) * np.eye(agent.K)

            # Gauge field
            if grads.delta_phi is not None and hasattr(agent, 'gauge'):
                agent.gauge.phi -= learning_rate * grads.delta_phi

    if verbose:
        print(f"Did not converge after {max_steps} steps")

    return None


def find_critical_points_random_restarts(
    system,
    n_restarts: int = 10,
    perturbation_scale: float = 0.5,
    **descent_kwargs
) -> List[CriticalPoint]:
    """
    Find multiple critical points using random restarts.

    For each restart:
    1. Perturb the current state randomly
    2. Run gradient descent to find a local minimum
    3. Check if this is a new critical point

    Args:
        system: MultiAgentSystem
        n_restarts: Number of random restarts
        perturbation_scale: Scale of random perturbations
        **descent_kwargs: Arguments passed to find_critical_point_gradient_descent

    Returns:
        List of unique critical points found
    """
    # Save initial state
    initial_state = extract_system_state(system)

    critical_points = []
    found_energies = set()  # Track unique points by energy (approximate)

    for restart in range(n_restarts):
        # Reset to initial state
        set_system_state(system, initial_state)

        # Perturb state
        for agent in system.agents:
            agent.mu_q += perturbation_scale * np.random.randn(*agent.mu_q.shape)
            agent.mu_p += perturbation_scale * np.random.randn(*agent.mu_p.shape)
            if hasattr(agent, 'gauge'):
                agent.gauge.phi += 0.1 * perturbation_scale * np.random.randn(*agent.gauge.phi.shape)

        # Find critical point
        cp = find_critical_point_gradient_descent(system, **descent_kwargs)

        if cp is not None:
            # Check if this is a new point (by energy)
            energy_key = round(cp.energy, 4)
            if energy_key not in found_energies:
                found_energies.add(energy_key)
                critical_points.append(cp)

    # Restore initial state
    set_system_state(system, initial_state)

    return critical_points


def compute_gradient_norm_field(
    system,
    agent_idx: int,
    param: str = 'mu_q',
    dim1: int = 0,
    dim2: int = 1,
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    grid_resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient norm over a 2D slice of parameter space.

    This creates a "potential landscape" visualization showing where
    gradients vanish (critical points appear as valleys in ||∇F||).

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent's parameters to vary
        param: Parameter to vary ('mu_q', 'mu_p')
        dim1, dim2: Which dimensions of the parameter to vary
        grid_range: (min, max) range for grid
        grid_resolution: Number of points per dimension

    Returns:
        (X, Y, Z): Meshgrid coordinates and gradient norm values
    """
    # Save initial state
    initial_state = extract_system_state(system)

    # Create grid
    x = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    y = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    agent = system.agents[agent_idx]

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Set parameter values
            if param == 'mu_q':
                original = agent.mu_q.copy()
                agent.mu_q[dim1] = X[i, j]
                agent.mu_q[dim2] = Y[i, j]
                grad_norm, _ = compute_gradient_norm(system)
                agent.mu_q = original
            elif param == 'mu_p':
                original = agent.mu_p.copy()
                agent.mu_p[dim1] = X[i, j]
                agent.mu_p[dim2] = Y[i, j]
                grad_norm, _ = compute_gradient_norm(system)
                agent.mu_p = original
            else:
                raise ValueError(f"Unknown parameter: {param}")

            Z[i, j] = grad_norm

    # Restore initial state
    set_system_state(system, initial_state)

    return X, Y, Z


def compute_energy_landscape(
    system,
    agent_idx: int,
    param: str = 'mu_q',
    dim1: int = 0,
    dim2: int = 1,
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    grid_resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute energy landscape over a 2D slice of parameter space.

    Critical points appear as local minima/maxima/saddles in this landscape.

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent's parameters to vary
        param: Parameter to vary ('mu_q', 'mu_p')
        dim1, dim2: Which dimensions of the parameter to vary
        grid_range: (min, max) range for grid
        grid_resolution: Number of points per dimension

    Returns:
        (X, Y, Z): Meshgrid coordinates and energy values
    """
    # Save initial state
    initial_state = extract_system_state(system)

    # Create grid
    x = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    y = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    agent = system.agents[agent_idx]

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Set parameter values
            if param == 'mu_q':
                original = agent.mu_q.copy()
                agent.mu_q[dim1] = X[i, j]
                agent.mu_q[dim2] = Y[i, j]
                energy = compute_total_free_energy(system).total
                agent.mu_q = original
            elif param == 'mu_p':
                original = agent.mu_p.copy()
                agent.mu_p[dim1] = X[i, j]
                agent.mu_p[dim2] = Y[i, j]
                energy = compute_total_free_energy(system).total
                agent.mu_p = original
            else:
                raise ValueError(f"Unknown parameter: {param}")

            Z[i, j] = energy

    # Restore initial state
    set_system_state(system, initial_state)

    return X, Y, Z


def scan_for_critical_points(
    system,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    grid_resolution: int = 20,
    gradient_threshold: float = 0.1,
    refine: bool = True,
    verbose: bool = False
) -> CriticalPointScan:
    """
    Scan parameter space for critical points.

    Strategy:
    1. Compute gradient norm on a coarse grid
    2. Find grid cells with low gradient norm
    3. Optionally refine with gradient descent

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent's parameters to scan
        param: Parameter to scan ('mu_q', 'mu_p')
        dims: Which two dimensions to scan
        grid_range: (min, max) range
        grid_resolution: Grid points per dimension
        gradient_threshold: Threshold for candidate critical points
        refine: Whether to refine candidates with gradient descent
        verbose: Print progress

    Returns:
        CriticalPointScan with detected critical points and landscape data
    """
    if verbose:
        print(f"Scanning parameter space for critical points...")
        print(f"Agent: {agent_idx}, Param: {param}, Dims: {dims}")
        print(f"Range: {grid_range}, Resolution: {grid_resolution}")

    # Save initial state
    initial_state = extract_system_state(system)

    # Compute gradient norm field
    X, Y, grad_field = compute_gradient_norm_field(
        system, agent_idx, param, dims[0], dims[1],
        grid_range, grid_resolution
    )

    # Compute energy landscape
    _, _, energy_field = compute_energy_landscape(
        system, agent_idx, param, dims[0], dims[1],
        grid_range, grid_resolution
    )

    # Find candidate critical points (low gradient norm)
    candidates = []
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            if grad_field[i, j] < gradient_threshold:
                candidates.append({
                    'grid_idx': (i, j),
                    'coords': (X[i, j], Y[i, j]),
                    'gradient_norm': grad_field[i, j],
                    'energy': energy_field[i, j]
                })

    if verbose:
        print(f"Found {len(candidates)} candidate points with ||∇F|| < {gradient_threshold}")

    # Refine candidates
    critical_points = []

    for idx, cand in enumerate(candidates):
        if verbose:
            print(f"Refining candidate {idx + 1}/{len(candidates)}...")

        # Set system to candidate location
        set_system_state(system, initial_state)
        agent = system.agents[agent_idx]

        if param == 'mu_q':
            agent.mu_q[dims[0]] = cand['coords'][0]
            agent.mu_q[dims[1]] = cand['coords'][1]
        elif param == 'mu_p':
            agent.mu_p[dims[0]] = cand['coords'][0]
            agent.mu_p[dims[1]] = cand['coords'][1]

        if refine:
            # Refine with gradient descent
            cp = find_critical_point_gradient_descent(
                system,
                max_steps=500,
                learning_rate=0.01,
                gradient_threshold=1e-5,
                verbose=False
            )
        else:
            # Use current state as critical point
            cp = detect_critical_point_at_state(system, gradient_threshold=gradient_threshold)

        if cp is not None:
            cp.metadata['initial_candidate'] = cand
            critical_points.append(cp)

    # Restore initial state
    set_system_state(system, initial_state)

    # Remove duplicates (points with very similar energy)
    unique_cps = []
    seen_energies = set()

    for cp in critical_points:
        energy_key = round(cp.energy, 3)
        if energy_key not in seen_energies:
            seen_energies.add(energy_key)
            unique_cps.append(cp)

    if verbose:
        print(f"\nFound {len(unique_cps)} unique critical points:")
        for cp in unique_cps:
            print(f"  {cp}")

    return CriticalPointScan(
        critical_points=unique_cps,
        scan_parameters={
            'agent_idx': agent_idx,
            'param': param,
            'dims': dims,
            'grid_range': grid_range,
            'grid_resolution': grid_resolution
        },
        gradient_field=grad_field,
        energy_landscape=energy_field
    )


def track_critical_points_under_parameter_change(
    system,
    param_name: str,
    param_values: np.ndarray,
    initial_critical_point: Optional[CriticalPoint] = None,
    continuation_steps: int = 100,
    verbose: bool = False
) -> List[Tuple[float, CriticalPoint]]:
    """
    Track how a critical point moves as a system parameter changes.

    This is numerical continuation: follow a critical point as the control
    parameter varies, detecting bifurcations when branches merge or split.

    Args:
        system: MultiAgentSystem
        param_name: Parameter to vary (e.g., 'lambda_prior_align')
        param_values: Array of parameter values to scan
        initial_critical_point: Starting critical point (or None to find one)
        continuation_steps: Gradient descent steps for each parameter value
        verbose: Print progress

    Returns:
        List of (param_value, CriticalPoint) pairs tracing the branch
    """
    # Save initial config
    original_value = getattr(system.config, param_name)

    branch = []

    # Find initial critical point if not provided
    if initial_critical_point is not None:
        set_system_state(system, initial_critical_point.location)

    for param_val in param_values:
        # Set parameter value
        setattr(system.config, param_name, param_val)

        if verbose:
            print(f"{param_name} = {param_val:.4f}", end=" ")

        # Find critical point near current state
        cp = find_critical_point_gradient_descent(
            system,
            max_steps=continuation_steps,
            learning_rate=0.01,
            gradient_threshold=1e-4,
            verbose=False
        )

        if cp is not None:
            branch.append((param_val, cp))
            if verbose:
                print(f"-> F = {cp.energy:.4f}, ||∇F|| = {cp.gradient_norm:.2e}")
        else:
            if verbose:
                print("-> Lost branch (bifurcation?)")
            # Try to recover by gradient descent from current state
            cp_recovery = find_critical_point_gradient_descent(
                system,
                max_steps=500,
                learning_rate=0.005,  # Smaller step
                gradient_threshold=1e-3,
                verbose=False
            )
            if cp_recovery is not None:
                branch.append((param_val, cp_recovery))
                cp_recovery.metadata['recovered'] = True

    # Restore original parameter
    setattr(system.config, param_name, original_value)

    return branch
