# -*- coding: utf-8 -*-
"""
Hamiltonian Trainer for Multi-Agent Belief Dynamics
===================================================

Integrates the Hamiltonian dynamics framework with the agent training system,
enabling underdamped dynamics as an alternative to gradient flow.

Comparison of Dynamics Regimes:
-------------------------------

**Overdamped (Gradient Flow - Current)**:
    θ̇ = -∇F(θ)
    - First-order dynamics
    - Dissipative (energy always decreases)
    - Fast convergence but no oscillations
    - Like particle moving through thick fluid

**Underdamped (Hamiltonian - New)**:
    dθ/dt = G^{-1} p
    dp/dt = -∇F - (curvature corrections)
    - Second-order dynamics
    - Nearly conservative (energy approximately conserved)
    - Orbital motion, can overshoot
    - Like particle moving in thin fluid

**Critical Damping (Optimal)**:
    Balance between overdamped and underdamped
    - Fastest convergence without oscillation
    - Requires tuning friction parameter

Author: Chris & Claude
Date: November 2025
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time

from gradients.free_energy_clean import compute_total_free_energy, FreeEnergyBreakdown
from config import TrainingConfig
from data_utils.mu_tracking import create_mu_tracker, MuCenterTracking


@dataclass
class HamiltonianHistory:
    """
    Extended history for Hamiltonian dynamics.

    Includes both energy components AND phase space variables.
    """

    steps: List[int] = field(default_factory=list)

    # Energy components (same as TrainingHistory)
    total_energy: List[float] = field(default_factory=list)
    self_energy: List[float] = field(default_factory=list)
    belief_align: List[float] = field(default_factory=list)
    prior_align: List[float] = field(default_factory=list)
    observations: List[float] = field(default_factory=list)

    # Hamiltonian energy components (NEW)
    kinetic_energy: List[float] = field(default_factory=list)
    potential_energy: List[float] = field(default_factory=list)
    total_hamiltonian: List[float] = field(default_factory=list)

    # Phase space tracking (NEW)
    momentum_norms: List[float] = field(default_factory=list)  # ||p||
    velocity_norms: List[float] = field(default_factory=list)  # ||θ̇||

    # Energy conservation diagnostic (NEW)
    energy_drift: List[float] = field(default_factory=list)  # |H(t) - H(0)|/H(0)

    # Mu center tracking
    mu_tracker: Optional[MuCenterTracking] = None

    # Agent snapshots for metric evolution analysis
    agent_snapshots: List[dict] = field(default_factory=list)
    snapshot_steps: List[int] = field(default_factory=list)

    def record(self, step: int, energies: FreeEnergyBreakdown,
               hamiltonian_energies: Optional[Tuple[float, float, float]] = None,
               momentum_norm: float = 0.0,
               velocity_norm: float = 0.0,
               system = None):
        """
        Record metrics for current step.

        Args:
            step: Current step number
            energies: Free energy breakdown
            hamiltonian_energies: (T, V, H) kinetic, potential, total
            momentum_norm: ||p||
            velocity_norm: ||θ̇||
            system: Agent system for mu tracking
        """
        self.steps.append(step)

        # Standard energy components
        self.total_energy.append(energies.total)
        self.self_energy.append(energies.self_energy)
        self.belief_align.append(energies.belief_align)
        self.prior_align.append(energies.prior_align)
        self.observations.append(energies.observations)

        # Hamiltonian energies
        if hamiltonian_energies is not None:
            T, V, H = hamiltonian_energies
            self.kinetic_energy.append(T)
            self.potential_energy.append(V)
            self.total_hamiltonian.append(H)

            # Energy drift
            if len(self.total_hamiltonian) == 1:
                # First step - no drift yet
                self.energy_drift.append(0.0)
            else:
                H0 = self.total_hamiltonian[0]
                drift = abs(H - H0) / (abs(H0) + 1e-10)
                self.energy_drift.append(drift)

        # Phase space norms
        self.momentum_norms.append(momentum_norm)
        self.velocity_norms.append(velocity_norm)

        # Mu tracking
        if self.mu_tracker is None and system is not None:
            self.mu_tracker = create_mu_tracker(system)

        if self.mu_tracker is not None and system is not None:
            self.mu_tracker.record(step, system)

    def save_snapshot(self, step: int, system):
        """
        Save agent state snapshot for later metric analysis.

        Stores copies of all agent fields needed to compute pullback metrics.
        GAUGE-COVARIANT: Stores Σ directly (not Cholesky factors).

        Args:
            step: Current training step
            system: MultiAgentSystem to snapshot
        """
        snapshot = {
            'step': step,
            'agents': []
        }

        for agent in system.agents:
            agent_data = {
                'agent_id': agent.agent_id,
                'mu_q': agent.mu_q.copy(),
                'Sigma_q': agent.Sigma_q.copy(),  # Store Σ directly (gauge-covariant)
                'mu_p': agent.mu_p.copy(),
                'Sigma_p': agent.Sigma_p.copy(),  # Store Σ directly (gauge-covariant)
            }

            # Add gauge field if present
            if hasattr(agent, 'gauge') and hasattr(agent.gauge, 'phi'):
                agent_data['phi'] = agent.gauge.phi.copy()

            snapshot['agents'].append(agent_data)

        self.agent_snapshots.append(snapshot)
        self.snapshot_steps.append(step)


class HamiltonianTrainer:
    """
    Hamiltonian dynamics trainer for belief evolution.

    Implements second-order dynamics:
        dθ/dt = ∂H/∂p = G^{-1} p
        dp/dt = -∂H/∂θ = -∇V - (curvature)

    where:
        - θ: Belief parameters (flattened μ_q, L_q for all agents)
        - p: Conjugate momenta
        - G: Fisher metric (mass tensor)
        - V: Total free energy

    Comparison to GradientFlowTrainer:
        - Gradient flow: θ̇ = -∇V (first-order, dissipative)
        - Hamiltonian: Uses momentum, approximately conserves energy
    """

    def __init__(self, system, config: Optional[TrainingConfig] = None,
                 friction: float = 0.0,
                 mass_scale: float = 1.0):
        """
        Initialize Hamiltonian trainer.

        Args:
            system: MultiAgentSystem instance
            config: Training configuration
            friction: Friction coefficient γ (0 = conservative, >0 = damped)
                     0.0: Pure Hamiltonian (underdamped)
                     0.1: Light damping
                     1.0: Critical damping
                     10.0: Heavy damping (approaches overdamped)
            mass_scale: Overall mass scale (affects oscillation frequency)
        """
        self.system = system
        self.config = config or TrainingConfig()
        self.friction = friction
        self.mass_scale = mass_scale

        # Training state
        self.history = HamiltonianHistory()
        self.current_step = 0

        # Mu tracking
        self.history.mu_tracker = create_mu_tracker(system)

        # Initialize phase space coordinates
        self._initialize_phase_space()

        # Track performance
        self._step_times = []

        print(f"✓ Hamiltonian trainer initialized")
        print(f"  Friction: γ = {friction}")
        print(f"  Mass scale: {mass_scale}")
        print(f"  Regime: {'Conservative' if friction < 0.01 else 'Damped'}")

    def _initialize_phase_space(self):
        """
        Initialize phase space coordinates (θ, p).

        θ: Flattened belief parameters
        p: Initial momenta (start from rest: p = 0)
        """
        # Flatten all agent parameters into θ
        self.theta = self._pack_parameters()

        # Initialize momenta to zero (start from rest)
        self.p = np.zeros_like(self.theta)

        print(f"  Phase space dimension: {len(self.theta)}")

    def _pack_parameters(self) -> np.ndarray:
        """
        Pack all agent parameters into flat vector θ.

        For each agent, includes:
            - mu_q (flattened)
            - L_q (flattened, lower triangular part)

        Returns:
            theta: Flattened parameter vector
        """
        params = []

        for agent in self.system.agents:
            # Flatten mu_q
            mu_flat = agent.mu_q.flatten()
            params.append(mu_flat)

            # Flatten L_q (Cholesky factor)
            # Extract lower triangular part to avoid redundancy
            L_flat = agent.L_q.reshape(-1, agent.config.K, agent.config.K)

            # For each spatial point, extract lower triangle
            for L_mat in L_flat:
                lower_indices = np.tril_indices(agent.config.K)
                L_lower = L_mat[lower_indices]
                params.append(L_lower)

        return np.concatenate(params)

    def _unpack_parameters(self, theta: np.ndarray):
        """
        Unpack flat vector θ back into agent parameters.

        Updates agent.mu_q and agent.L_q in place.

        Args:
            theta: Flattened parameter vector
        """
        idx = 0

        for agent in self.system.agents:
            K = agent.config.K
            n_spatial = agent.mu_q.size // K

            # Unpack mu_q
            mu_size = n_spatial * K
            mu_flat = theta[idx:idx + mu_size]
            agent.mu_q = mu_flat.reshape(agent.mu_q.shape)
            idx += mu_size

            # Unpack L_q
            L_size = K * (K + 1) // 2  # Lower triangular elements per spatial point

            for i in range(n_spatial):
                L_lower = theta[idx:idx + L_size]
                idx += L_size

                # Reconstruct matrix from lower triangular
                L_mat = np.zeros((K, K))
                lower_indices = np.tril_indices(K)
                L_mat[lower_indices] = L_lower

                # Update agent's L_q
                if agent.L_q.ndim == 2:
                    # 0D agent (particle)
                    agent.L_q = L_mat
                elif agent.L_q.ndim == 3:
                    # 1D grid
                    agent.L_q[i] = L_mat
                else:
                    # 2D grid - need to map linear index to (x, y)
                    shape = agent.L_q.shape[:2]
                    x = i // shape[1]
                    y = i % shape[1]
                    agent.L_q[x, y] = L_mat

            # Recompute Sigma_q from L_q
            agent.Sigma_q = agent.L_q @ np.swapaxes(agent.L_q, -2, -1)

    def _compute_metric(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher metric G(θ).

        For now, uses block diagonal approximation:
            G = diag(Σ_1^{-1}, Σ_2^{-1}, ..., Σ_N^{-1})

        Full implementation would include cross-agent coupling.

        Args:
            theta: Parameter vector

        Returns:
            G: Fisher metric matrix (d, d)
        """
        # Simplified: Block diagonal with each block = Σ^{-1}
        d = len(theta)
        G = np.zeros((d, d))

        idx = 0
        for agent in self.system.agents:
            K = agent.config.K
            n_spatial = agent.mu_q.size // K

            # Mu block: G_μ = Σ^{-1}
            for i in range(n_spatial):
                if agent.Sigma_q.ndim == 2:
                    Sigma = agent.Sigma_q
                elif agent.Sigma_q.ndim == 3:
                    Sigma = agent.Sigma_q[i]
                else:
                    shape = agent.Sigma_q.shape[:2]
                    x = i // shape[1]
                    y = i % shape[1]
                    Sigma = agent.Sigma_q[x, y]

                try:
                    Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(K))
                except:
                    Sigma_inv = np.eye(K)

                # Fill diagonal block
                mu_block_size = K
                G[idx:idx+mu_block_size, idx:idx+mu_block_size] = Sigma_inv * self.mass_scale
                idx += mu_block_size

                # L block: Identity (simplified)
                L_block_size = K * (K + 1) // 2
                G[idx:idx+L_block_size, idx:idx+L_block_size] = np.eye(L_block_size) * self.mass_scale
                idx += L_block_size

        return G

    def _compute_potential(self, theta: np.ndarray) -> float:
        """
        Compute potential energy V(θ) = total free energy.

        Args:
            theta: Parameter vector

        Returns:
            V: Potential energy (free energy)
        """
        # Unpack theta into agent parameters
        theta_backup = self._pack_parameters()
        self._unpack_parameters(theta)

        # Compute free energy
        energies = compute_total_free_energy(self.system)
        V = energies.total

        # Restore original parameters
        self._unpack_parameters(theta_backup)

        return V

    def _compute_force(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute force -∇V(θ).

        Uses finite differences to approximate gradient.

        Args:
            theta: Parameter vector

        Returns:
            force: -∇V (d,)
        """
        eps = 1e-5
        grad_V = np.zeros_like(theta)

        V0 = self._compute_potential(theta)

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps

            V_plus = self._compute_potential(theta_plus)
            grad_V[i] = (V_plus - V0) / eps

        return -grad_V

    def _hamiltonian_equations(self, theta: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Hamilton's equations of motion.

        dθ/dt = ∂H/∂p = G^{-1} p
        dp/dt = -∂H/∂θ = -∇V - (curvature)

        If friction > 0, adds damping: dp/dt -= γ*p

        Args:
            theta: Position (parameters)
            p: Momentum

        Returns:
            dtheta_dt: dθ/dt
            dp_dt: dp/dt
        """
        # Get metric
        G = self._compute_metric(theta)

        try:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        except:
            G_inv = np.eye(len(G)) / self.mass_scale

        # dθ/dt = G^{-1} p
        dtheta_dt = G_inv @ p

        # dp/dt = -∇V (ignoring curvature correction for now)
        force = self._compute_force(theta)
        dp_dt = force

        # Add friction if specified
        if self.friction > 0:
            dp_dt -= self.friction * p

        return dtheta_dt, dp_dt

    def step(self, dt: float = 0.01) -> FreeEnergyBreakdown:
        """
        Perform one Hamiltonian evolution step.

        Uses leapfrog integrator (symplectic) for energy conservation.

        Args:
            dt: Time step size

        Returns:
            energies: Free energy breakdown after update
        """
        step_start = time.perf_counter()

        # Leapfrog integration (symplectic)
        # Reference: Neal (2011) "MCMC using Hamiltonian dynamics"

        # Half step for momentum
        _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
        self.p = self.p + 0.5 * dt * dp_dt

        # Full step for position
        dtheta_dt, _ = self._hamiltonian_equations(self.theta, self.p)
        self.theta = self.theta + dt * dtheta_dt

        # Update agent parameters
        self._unpack_parameters(self.theta)

        # Half step for momentum
        _, dp_dt = self._hamiltonian_equations(self.theta, self.p)
        self.p = self.p + 0.5 * dt * dp_dt

        # Compute energies
        energies = compute_total_free_energy(self.system)

        # Compute Hamiltonian energies
        G = self._compute_metric(self.theta)
        try:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        except:
            G_inv = np.eye(len(G)) / self.mass_scale

        T = 0.5 * self.p @ G_inv @ self.p  # Kinetic
        V = energies.total  # Potential
        H = T + V  # Total Hamiltonian

        # Record history
        if self.config.save_history:
            self.history.record(
                step=self.current_step,
                energies=energies,
                hamiltonian_energies=(T, V, H),
                momentum_norm=np.linalg.norm(self.p),
                velocity_norm=np.linalg.norm(dtheta_dt),
                system=self.system
            )

        self.current_step += 1
        self._step_times.append(time.perf_counter() - step_start)

        return energies

    def train(self, n_steps: Optional[int] = None, dt: float = 0.01) -> HamiltonianHistory:
        """
        Run full Hamiltonian training loop.

        Args:
            n_steps: Number of steps (overrides config if provided)
            dt: Time step size

        Returns:
            history: Training history with Hamiltonian diagnostics
        """
        n_steps = n_steps or self.config.n_steps

        print("="*70)
        print("HAMILTONIAN BELIEF DYNAMICS")
        print("="*70)
        print(f"System: {self.system.n_agents} agents")
        print(f"Steps: {n_steps}")
        print(f"Time step: dt = {dt}")
        print(f"Friction: γ = {self.friction}")
        print(f"Regime: {'Conservative (Underdamped)' if self.friction < 0.01 else 'Damped'}")
        print("="*70)

        # Initial energy
        initial_energies = compute_total_free_energy(self.system)
        G = self._compute_metric(self.theta)
        G_inv = np.linalg.inv(G + 1e-6 * np.eye(len(G)))
        T0 = 0.5 * self.p @ G_inv @ self.p
        V0 = initial_energies.total
        H0 = T0 + V0

        print(f"\nInitial state:")
        print(f"  Free energy (V): {V0:.6f}")
        print(f"  Kinetic energy (T): {T0:.6f}")
        print(f"  Total Hamiltonian (H): {H0:.6f}")
        print()

        # Training loop
        try:
            for step in range(n_steps):
                energies = self.step(dt)

                # Logging
                if step % self.config.log_every == 0:
                    self._log_step(step, energies)

                # Snapshot saving for metric analysis
                if (self.config.save_snapshots and
                    step % self.config.snapshot_every == 0):
                    self.history.save_snapshot(step, self.system)

        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")

        # Final summary
        final_energies = compute_total_free_energy(self.system)
        G_final = self._compute_metric(self.theta)
        G_inv_final = np.linalg.inv(G_final + 1e-6 * np.eye(len(G_final)))
        T_final = 0.5 * self.p @ G_inv_final @ self.p
        V_final = final_energies.total
        H_final = T_final + V_final

        energy_conservation = abs(H_final - H0) / (abs(H0) + 1e-10)
        avg_step_time = np.mean(self._step_times[-100:])

        print("\n" + "="*70)
        print("HAMILTONIAN TRAINING COMPLETE")
        print("="*70)
        print(f"Final state:")
        print(f"  Free energy (V): {V_final:.6f} (Δ = {V_final - V0:+.6f})")
        print(f"  Kinetic energy (T): {T_final:.6f}")
        print(f"  Total Hamiltonian (H): {H_final:.6f}")
        print()
        print(f"Energy conservation:")
        print(f"  |H(T) - H(0)|/H(0) = {energy_conservation:.2e}")
        print(f"  Status: {'✓ GOOD' if energy_conservation < 0.01 else '⚠ DRIFT'}")
        print()
        print("Performance:")
        print(f"  Avg step time: {avg_step_time:.4f}s")
        print(f"  Steps/second: {1.0/avg_step_time:.2f}")
        print("="*70)

        return self.history

    def _log_step(self, step: int, energies: FreeEnergyBreakdown):
        """Print progress for current step."""
        if len(self.history.total_hamiltonian) > 0:
            H = self.history.total_hamiltonian[-1]
            T = self.history.kinetic_energy[-1]
            V = self.history.potential_energy[-1]
            drift = self.history.energy_drift[-1]

            msg = f"Step {step:5d}: H = {H:8.4f} [T={T:.3f}, V={V:.3f}]"
            msg += f"  drift={drift:.2e}"

            # Add timing if available
            if self._step_times:
                recent_time = np.mean(self._step_times[-10:])
                msg += f"  ({recent_time:.3f}s)"

            print(msg)
