#!/usr/bin/env python3
"""
Damping Regimes and Epistemic Momentum Simulations
====================================================

Comprehensive simulations demonstrating the Hamiltonian dynamics of belief
evolution from "The Inertia of Belief: Hiding in Plain Sight" manuscript.

NOW WIRED INTO CORE SUITE:
- Uses real Agent, MultiAgentSystem, HamiltonianTrainer infrastructure
- Standalone toy models retained for reference/quick testing
- Full integration with simulation_config.py presets

Simulations implemented:
1. Three Damping Regimes - Overdamped, critically damped, underdamped
2. Two-Agent Momentum Transfer - Recoil effect visualization
3. Confirmation Bias as Stopping Distance - d proportional to Lambda relationship
4. Resonance Curve - Peak at omega_res = sqrt(K/M)
5. Belief Perseverance Decay - tau proportional to Lambda/gamma relationship

Key equations from manuscript:
- Damped oscillator: M mu_ddot + gamma mu_dot + K mu = f(t)
- Mass = Precision: M = Lambda = Sigma^{-1}
- Natural frequency: omega_0 = sqrt(K/M)
- Damping regimes based on Delta = gamma^2 - 4KM
- Stopping distance: d_stop = M||mu_dot||^2 / (2||f||)
- Decay time: tau = M/gamma

Author: Generated from psych_manuscript.pdf theory
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import sys

# =============================================================================
# Core Suite Imports (Wire into real simulation infrastructure)
# =============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation_config import (
    SimulationConfig,
    damping_regime_overdamped_config,
    damping_regime_critical_config,
    damping_regime_underdamped_config,
    momentum_transfer_config,
    resonance_experiment_config,
    belief_perseverance_config,
)
from config import AgentConfig, SystemConfig, TrainingConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from agent.trainer import Trainer, TrainingHistory
from agent.hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from geometry.geometry_base import BaseManifold, TopologyType
from gradients.free_energy_clean import compute_total_free_energy


# =============================================================================
# Core Suite Experiment Wrapper
# =============================================================================

class CoreSuiteExperiment:
    """
    Wrapper for running damping regime experiments using the real simulation infrastructure.

    This bridges the gap between the toy oscillator models and the full
    multi-agent belief dynamics system.

    Key mapping from manuscript equations to core suite:
    - M (mass) = Lambda (precision) = Sigma_p^{-1} (prior precision)
    - K (stiffness) = lambda_prior_align (energy weight for prior alignment)
    - gamma (damping) = hamiltonian_friction in HamiltonianTrainer
    - mu (belief position) = agent.mu_q
    - pi (momentum) = trainer.p (phase space momentum)
    """

    def __init__(self, config: SimulationConfig, seed: int = 42):
        """
        Initialize experiment from SimulationConfig.

        Args:
            config: SimulationConfig specifying system parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # Build system components
        self.manifold = self._build_manifold()
        self.agents = self._build_agents()
        self.system = self._build_system()
        self.trainer = self._build_trainer()

        # Storage for trajectory data
        self.trajectory = {
            't': [],
            'mu': [],  # List of mu_q snapshots per agent
            'pi': [],  # List of momentum snapshots per agent (Hamiltonian only)
            'energy': [],
            'kinetic': [],
            'potential': [],
        }

    def _build_manifold(self) -> BaseManifold:
        """Create 0D point manifold for particle-like dynamics."""
        return BaseManifold(
            shape=self.config.spatial_shape,
            topology=TopologyType.FLAT
        )

    def _build_agents(self) -> List[Agent]:
        """Create agents with configured precision/mass."""
        agent_cfg = AgentConfig(
            spatial_shape=self.config.spatial_shape,
            K=self.config.K_latent,
            mu_scale=self.config.mu_scale,
            sigma_scale=self.config.sigma_scale,
        )

        agents = []
        for i in range(self.config.n_agents):
            agent = Agent(
                agent_id=i,
                config=agent_cfg,
                rng=self.rng,
                base_manifold=self.manifold
            )
            agents.append(agent)

        return agents

    def _build_system(self) -> MultiAgentSystem:
        """Create MultiAgentSystem with configured energy weights."""
        system_cfg = SystemConfig(
            lambda_self=self.config.lambda_self,
            lambda_belief_align=self.config.lambda_belief_align,
            lambda_prior_align=self.config.lambda_prior_align,
            lambda_obs=self.config.lambda_obs,
            lambda_phi=self.config.lambda_phi,
            kappa_beta=self.config.kappa_beta,
            kappa_gamma=self.config.kappa_gamma,
            seed=self.seed,
        )
        return MultiAgentSystem(self.agents, system_cfg)

    def _build_trainer(self) -> Union[Trainer, HamiltonianTrainer]:
        """Create appropriate trainer based on config."""
        training_cfg = TrainingConfig(
            n_steps=self.config.n_steps,
            log_every=self.config.log_every,
            lr_mu_q=self.config.lr_mu_q,
            lr_sigma_q=self.config.lr_sigma_q,
            lr_mu_p=self.config.lr_mu_p,
            lr_sigma_p=self.config.lr_sigma_p,
            lr_phi=self.config.lr_phi,
        )

        if self.config.enable_hamiltonian:
            return HamiltonianTrainer(
                self.system,
                config=training_cfg,
                friction=self.config.hamiltonian_friction,
                mass_scale=self.config.hamiltonian_mass_scale
            )
        else:
            return Trainer(self.system, training_cfg)

    def set_initial_state(self,
                          mu0: Optional[np.ndarray] = None,
                          pi0: Optional[np.ndarray] = None,
                          agent_idx: int = 0):
        """
        Set initial belief position and momentum for an agent.

        Args:
            mu0: Initial belief mean (shape: (K,) for 0D agent)
            pi0: Initial momentum (only for Hamiltonian trainer)
            agent_idx: Which agent to modify
        """
        agent = self.agents[agent_idx]

        if mu0 is not None:
            agent.mu_q = mu0.reshape(agent.mu_q.shape).astype(np.float32)

        if pi0 is not None and self.config.enable_hamiltonian:
            # Set momentum in Hamiltonian trainer's phase space
            trainer = self.trainer
            if hasattr(trainer, 'p'):
                K = agent.config.K
                # Momentum is packed in same order as theta
                # For 0D agent: first K elements are mu momentum
                trainer.p[:K] = pi0.flatten()

    def set_precision(self, precision: float, agent_idx: int = 0):
        """
        Set agent's precision (mass) by modifying prior covariance.

        M = Lambda = Sigma_p^{-1}
        So Sigma_p = 1/precision * I

        Args:
            precision: Desired precision (mass) value
            agent_idx: Which agent to modify
        """
        agent = self.agents[agent_idx]
        K = agent.config.K

        # Sigma_p = (1/precision) * I
        agent.Sigma_p = (1.0 / precision) * np.eye(K, dtype=np.float32)
        # Also update Cholesky factor
        agent.L_p = np.sqrt(1.0 / precision) * np.eye(K, dtype=np.float32)

        # Invalidate caches
        if hasattr(agent, '_precision_p_cache'):
            agent._precision_p_cache = None

    def run(self, n_steps: Optional[int] = None, dt: float = 0.01) -> Dict:
        """
        Run the experiment and record trajectory.

        Args:
            n_steps: Number of steps (overrides config)
            dt: Time step for Hamiltonian integration

        Returns:
            Dictionary with trajectory data
        """
        n_steps = n_steps or self.config.n_steps

        # Clear trajectory
        self.trajectory = {
            't': [],
            'mu': [[] for _ in self.agents],
            'pi': [[] for _ in self.agents],
            'energy': [],
            'kinetic': [],
            'potential': [],
        }

        # Record initial state
        self._record_state(0)

        # Run training
        if self.config.enable_hamiltonian:
            for step in range(n_steps):
                self.trainer.step(dt=dt)
                self._record_state(step + 1)
        else:
            for step in range(n_steps):
                self.trainer.step()
                self._record_state(step + 1)

        # Convert to arrays
        self.trajectory['t'] = np.array(self.trajectory['t'])
        for i in range(len(self.agents)):
            self.trajectory['mu'][i] = np.array(self.trajectory['mu'][i])
            self.trajectory['pi'][i] = np.array(self.trajectory['pi'][i])
        self.trajectory['energy'] = np.array(self.trajectory['energy'])
        self.trajectory['kinetic'] = np.array(self.trajectory['kinetic'])
        self.trajectory['potential'] = np.array(self.trajectory['potential'])

        return self.trajectory

    def _record_state(self, step: int):
        """Record current state to trajectory."""
        self.trajectory['t'].append(step * self.config.hamiltonian_dt
                                    if self.config.enable_hamiltonian else step)

        # Record mu for each agent
        for i, agent in enumerate(self.agents):
            self.trajectory['mu'][i].append(agent.mu_q.copy())

            # Record momentum if Hamiltonian
            if self.config.enable_hamiltonian and hasattr(self.trainer, 'p'):
                K = agent.config.K
                # Extract momentum for this agent from packed vector
                # Simple approximation: first agent's mu momentum is first K elements
                offset = i * (K + K * (K + 1) // 2)  # mu size + Sigma upper triangle
                pi = self.trainer.p[offset:offset + K]
                self.trajectory['pi'][i].append(pi.copy())
            else:
                self.trajectory['pi'][i].append(np.zeros(agent.config.K))

        # Record energies
        energies = compute_total_free_energy(self.system)
        self.trajectory['energy'].append(energies.total)

        if self.config.enable_hamiltonian and hasattr(self.trainer, 'history'):
            if len(self.trainer.history.kinetic_energy) > 0:
                self.trajectory['kinetic'].append(self.trainer.history.kinetic_energy[-1])
                self.trajectory['potential'].append(self.trainer.history.potential_energy[-1])
            else:
                self.trajectory['kinetic'].append(0.0)
                self.trajectory['potential'].append(energies.total)
        else:
            self.trajectory['kinetic'].append(0.0)
            self.trajectory['potential'].append(energies.total)


# =============================================================================
# Toy Model Classes (Kept for Reference and Quick Testing)
# =============================================================================

@dataclass
class BeliefState:
    """State of a single agent's belief (TOY MODEL)."""
    mu: float           # Belief mean (position)
    pi: float           # Belief momentum
    precision: float    # Lambda = 1/sigma^2 (acts as mass)
    t: float = 0.0

    @property
    def mass(self) -> float:
        return self.precision

    @property
    def velocity(self) -> float:
        return self.pi / self.mass


@dataclass
class TwoAgentState:
    """State of two coupled agents (TOY MODEL)."""
    mu1: float
    mu2: float
    pi1: float
    pi2: float
    precision1: float
    precision2: float
    coupling: float     # beta_12 = beta_21 attention coupling
    t: float = 0.0


class EpistemicOscillator:
    """
    Damped epistemic oscillator from Eq. 36-37 of manuscript (TOY MODEL).

    M mu_ddot + gamma mu_dot + K(mu - mu*) = f(t)

    Where:
    - M = precision (epistemic mass)
    - gamma = damping coefficient
    - K = evidence strength (spring constant)
    - mu* = equilibrium (prior or evidence target)
    - f(t) = external forcing (evidence stream)

    NOTE: This is a simplified toy model. For full dynamics, use CoreSuiteExperiment.
    """

    def __init__(
        self,
        precision: float,      # M = Lambda
        stiffness: float,      # K (evidence strength)
        damping: float,        # gamma
        equilibrium: float = 0.0,  # mu* (target)
    ):
        self.M = precision
        self.K = stiffness
        self.gamma = damping
        self.mu_eq = equilibrium

    @property
    def natural_frequency(self) -> float:
        """omega_0 = sqrt(K/M)"""
        return np.sqrt(self.K / self.M)

    @property
    def damping_ratio(self) -> float:
        """zeta = gamma / (2*sqrt(KM))"""
        return self.gamma / (2 * np.sqrt(self.K * self.M))

    @property
    def discriminant(self) -> float:
        """Delta = gamma^2 - 4KM"""
        return self.gamma**2 - 4 * self.K * self.M

    @property
    def regime(self) -> str:
        """Determine damping regime."""
        if self.discriminant > 0:
            return "overdamped"
        elif abs(self.discriminant) < 1e-10:
            return "critical"
        else:
            return "underdamped"

    @property
    def damped_frequency(self) -> float:
        """omega = sqrt(K/M - gamma^2/(4M^2)) for underdamped case."""
        if self.regime == "underdamped":
            return np.sqrt(self.K/self.M - self.gamma**2/(4*self.M**2))
        return 0.0

    @property
    def decay_time(self) -> float:
        """tau = 2M/gamma (Eq. 39)"""
        if self.gamma > 0:
            return 2 * self.M / self.gamma
        return float('inf')

    def equations_of_motion(
        self,
        t: float,
        y: np.ndarray,
        forcing: Optional[callable] = None
    ) -> np.ndarray:
        """
        Hamilton's equations with damping.

        dy/dt = [dmu/dt, dpi/dt]
              = [pi/M, -K(mu - mu*) - gamma*(pi/M) + f(t)]
        """
        mu, pi = y

        # Velocity
        dmu_dt = pi / self.M

        # Force: -dV/dmu - damping + external
        force_potential = -self.K * (mu - self.mu_eq)
        force_damping = -self.gamma * dmu_dt
        force_external = forcing(t) if forcing else 0.0

        dpi_dt = force_potential + force_damping + force_external

        return np.array([dmu_dt, dpi_dt])

    def simulate(
        self,
        mu0: float,
        pi0: float,
        t_end: float,
        dt: float = 0.01,
        forcing: Optional[callable] = None
    ) -> Dict[str, np.ndarray]:
        """Simulate belief evolution."""
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end, dt)
        y0 = np.array([mu0, pi0])

        sol = solve_ivp(
            lambda t, y: self.equations_of_motion(t, y, forcing),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )

        return {
            't': sol.t,
            'mu': sol.y[0],
            'pi': sol.y[1],
            'velocity': sol.y[1] / self.M,
            'kinetic_energy': 0.5 * sol.y[1]**2 / self.M,
            'potential_energy': 0.5 * self.K * (sol.y[0] - self.mu_eq)**2,
        }


class TwoAgentSystem:
    """
    Two coupled agents with momentum transfer (Section 4.6) - TOY MODEL.

    M1*mu1_ddot + gamma1*mu1_dot = -Lambda_bar_1*(mu1 - mu_bar_1) - beta_12*Lambda2*(mu1 - mu2)
    M2*mu2_ddot + gamma2*mu2_dot = -Lambda_bar_2*(mu2 - mu_bar_2) - beta_21*Lambda1*(mu2 - mu1)

    The coupling terms show momentum transfer and recoil.

    NOTE: For full multi-agent dynamics, use CoreSuiteExperiment with n_agents=2.
    """

    def __init__(
        self,
        precision1: float,
        precision2: float,
        coupling: float,        # beta_12 = beta_21
        damping1: float = 0.1,
        damping2: float = 0.1,
        prior1: float = 0.0,
        prior2: float = 0.0,
        prior_strength: float = 0.1,  # Lambda_bar (prior anchoring)
    ):
        self.M1 = precision1
        self.M2 = precision2
        self.beta = coupling
        self.gamma1 = damping1
        self.gamma2 = damping2
        self.mu_bar1 = prior1
        self.mu_bar2 = prior2
        self.Lambda_bar = prior_strength

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Coupled Hamilton's equations.

        y = [mu1, mu2, pi1, pi2]
        """
        mu1, mu2, pi1, pi2 = y

        # Velocities
        v1 = pi1 / self.M1
        v2 = pi2 / self.M2

        # Forces on agent 1
        force_prior1 = -self.Lambda_bar * (mu1 - self.mu_bar1)
        force_consensus1 = -self.beta * self.M2 * (mu1 - mu2)  # Pull toward agent 2
        force_damping1 = -self.gamma1 * v1

        # Forces on agent 2
        force_prior2 = -self.Lambda_bar * (mu2 - self.mu_bar2)
        force_consensus2 = -self.beta * self.M1 * (mu2 - mu1)  # Pull toward agent 1
        force_damping2 = -self.gamma2 * v2

        return np.array([
            v1,  # dmu1/dt
            v2,  # dmu2/dt
            force_prior1 + force_consensus1 + force_damping1,  # dpi1/dt
            force_prior2 + force_consensus2 + force_damping2,  # dpi2/dt
        ])

    def simulate(
        self,
        mu1_0: float,
        mu2_0: float,
        pi1_0: float,
        pi2_0: float,
        t_end: float,
        dt: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """Simulate two-agent dynamics."""
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end, dt)
        y0 = np.array([mu1_0, mu2_0, pi1_0, pi2_0])

        sol = solve_ivp(
            self.equations_of_motion,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            max_step=dt
        )

        # Compute total momentum (should show transfer)
        total_momentum = sol.y[2] + sol.y[3]

        return {
            't': sol.t,
            'mu1': sol.y[0],
            'mu2': sol.y[1],
            'pi1': sol.y[2],
            'pi2': sol.y[3],
            'total_momentum': total_momentum,
            'momentum_diff': sol.y[2] - sol.y[3],
        }


# =============================================================================
# Simulation 1: Three Damping Regimes (Core Suite Version)
# =============================================================================

def simulate_damping_regimes_core(
    precision: float = 2.0,
    stiffness: float = 1.0,
    mu0: float = 1.0,
    n_steps: int = 200,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Simulate three damping regimes using the CORE SUITE.

    Uses real Agent, MultiAgentSystem, and Trainer/HamiltonianTrainer.

    Regimes:
    - Overdamped: Gradient flow (Trainer)
    - Critical: HamiltonianTrainer with friction=1.0
    - Underdamped: HamiltonianTrainer with friction=0.1

    Args:
        precision: Epistemic mass M = Lambda
        stiffness: Evidence strength K (maps to lambda_prior_align)
        mu0: Initial belief displacement
        n_steps: Number of evolution steps
        output_dir: Where to save figures
        seed: Random seed

    Returns:
        Dictionary with simulation results for each regime
    """
    if output_dir is None:
        output_dir = Path("_experiments/damping_regimes_core")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 1: THREE DAMPING REGIMES (CORE SUITE)")
    print("="*70)
    print(f"Precision (Mass) M = {precision}")
    print(f"Stiffness K = {stiffness}")
    print(f"Using real Agent/Trainer infrastructure")
    print()

    results = {}

    # Define regimes with their configs
    regime_configs = {
        'overdamped': damping_regime_overdamped_config(),
        'critical': damping_regime_critical_config(),
        'underdamped': damping_regime_underdamped_config(),
    }

    for regime_name, config in regime_configs.items():
        print(f"Running {regime_name.upper()}...")

        # Override precision via prior covariance
        config.n_steps = n_steps
        config.lambda_prior_align = stiffness

        # Create experiment
        exp = CoreSuiteExperiment(config, seed=seed)

        # Set initial state: displaced from equilibrium
        K = config.K_latent
        mu_init = np.ones(K) * mu0
        exp.set_initial_state(mu0=mu_init)
        exp.set_precision(precision)

        # Run
        trajectory = exp.run(n_steps=n_steps)

        # Extract first component for 1D visualization
        mu_trajectory = np.array([m[0] for m in trajectory['mu'][0]])

        # Compute regime properties
        if config.enable_hamiltonian:
            gamma = config.hamiltonian_friction
        else:
            gamma = 10.0  # Effective high damping for gradient flow

        gamma_critical = 2 * np.sqrt(stiffness * precision)
        damping_ratio = gamma / gamma_critical if gamma_critical > 0 else 0
        discriminant = gamma**2 - 4 * stiffness * precision
        decay_time = 2 * precision / gamma if gamma > 0 else float('inf')

        results[regime_name] = {
            't': trajectory['t'],
            'mu': mu_trajectory,
            'energy': trajectory['energy'],
            'kinetic': trajectory['kinetic'],
            'potential': trajectory['potential'],
            'regime': regime_name,
            'gamma': gamma,
            'damping_ratio': damping_ratio,
            'discriminant': discriminant,
            'decay_time': decay_time,
        }

        print(f"  gamma = {gamma:.3f}, zeta = {damping_ratio:.3f}, tau = {decay_time:.3f}")

    # Create visualization
    _plot_damping_regimes_core(results, output_dir)

    return results


def _plot_damping_regimes_core(results: Dict, output_dir: Path):
    """Create comprehensive damping regimes visualization for core suite results."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'overdamped': '#e74c3c', 'critical': '#2ecc71', 'underdamped': '#3498db'}
    labels = {'overdamped': 'Overdamped (Gradient Flow)',
              'critical': 'Critical (Hamiltonian)',
              'underdamped': 'Underdamped (Hamiltonian)'}

    # Row 1: Time evolution of belief mu(t)
    ax1 = fig.add_subplot(gs[0, :2])
    for regime, result in results.items():
        ax1.plot(result['t'], result['mu'], color=colors[regime],
                 linewidth=2.5, label=labels[regime], alpha=0.9)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief mu(t)', fontsize=12)
    ax1.set_title('Belief Evolution: Three Damping Regimes (CORE SUITE)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Row 1 right: Energy evolution
    ax2 = fig.add_subplot(gs[0, 2])
    for regime, result in results.items():
        ax2.plot(result['t'], result['energy'], color=colors[regime],
                 linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Free Energy', fontsize=12)
    ax2.set_title('Energy Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Row 2: Phase portraits (mu vs velocity approximation)
    for idx, (regime, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])

        # Approximate velocity from mu differences
        mu = result['mu']
        t = result['t']
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        velocity = np.gradient(mu, dt)

        ax.plot(mu, velocity, color=colors[regime], linewidth=2, alpha=0.8)
        ax.plot(mu[0], velocity[0], 'o', color='green', markersize=12, label='Start', zorder=5)
        ax.plot(mu[-1], velocity[-1], 's', color='red', markersize=10, label='End', zorder=5)
        ax.plot(0, 0, '*', color='gold', markersize=15, label='Equilibrium', zorder=5)

        ax.set_xlabel('Belief mu', fontsize=12)
        ax.set_ylabel('Velocity d(mu)/dt', fontsize=12)
        ax.set_title(f'Phase Portrait: {labels[regime]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    # Row 3: Velocity and comparison
    ax5 = fig.add_subplot(gs[2, 0])
    for regime, result in results.items():
        mu = result['mu']
        t = result['t']
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        velocity = np.gradient(mu, dt)
        ax5.plot(t, velocity, color=colors[regime], linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time t', fontsize=12)
    ax5.set_ylabel('Velocity d(mu)/dt', fontsize=12)
    ax5.set_title('Belief Velocity Evolution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Comparison panel
    ax6 = fig.add_subplot(gs[2, 1])
    regimes_list = list(results.keys())
    x_pos = np.arange(len(regimes_list))

    decay_times = [results[r]['decay_time'] for r in regimes_list]
    # Cap infinite decay times for display
    decay_times = [min(d, 100) for d in decay_times]

    ax6.bar(x_pos, decay_times, color=[colors[r] for r in regimes_list], alpha=0.8)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([r.capitalize() for r in regimes_list])
    ax6.set_ylabel('Decay Time tau = 2M/gamma', fontsize=12)
    ax6.set_title('Characteristic Timescales', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = """
    CORE SUITE INTEGRATION:

    Using REAL infrastructure:
    - Agent class with mu_q, Sigma_q
    - MultiAgentSystem with energy weights
    - Trainer (overdamped) or
      HamiltonianTrainer (critical/underdamped)

    Mapping from manuscript:
    - M (mass) = prior precision
    - K (stiffness) = lambda_prior_align
    - gamma = hamiltonian_friction

    Damping Regimes:
    - Overdamped: Gradient flow
    - Critical: Hamiltonian, gamma=1.0
    - Underdamped: Hamiltonian, gamma=0.1
    """
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.suptitle('The Inertia of Belief: Damping Regimes (Core Suite)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "damping_regimes_core.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "damping_regimes_core.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'damping_regimes_core.png'}")


# =============================================================================
# Simulation 2: Two-Agent Momentum Transfer (Core Suite Version)
# =============================================================================

def simulate_momentum_transfer_core(
    precision1: float = 2.0,
    precision2: float = 1.0,
    initial_velocity: float = 1.0,
    n_steps: int = 300,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Simulate momentum transfer between two coupled agents using CORE SUITE.

    Uses real two-agent system with belief alignment coupling.

    Args:
        precision1: Agent 1's precision (influencer)
        precision2: Agent 2's precision (listener)
        initial_velocity: Initial belief velocity for agent 1
        n_steps: Number of steps
        output_dir: Where to save figures
        seed: Random seed
    """
    if output_dir is None:
        output_dir = Path("_experiments/momentum_transfer_core")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 2: MOMENTUM TRANSFER (CORE SUITE)")
    print("="*70)
    print(f"Agent 1: Precision = {precision1} (influencer)")
    print(f"Agent 2: Precision = {precision2} (listener)")
    print()

    # Use momentum transfer config
    config = momentum_transfer_config()
    config.n_steps = n_steps

    # Create experiment
    exp = CoreSuiteExperiment(config, seed=seed)

    # Set different precisions
    exp.set_precision(precision1, agent_idx=0)
    exp.set_precision(precision2, agent_idx=1)

    # Set initial momentum for agent 1
    K = config.K_latent
    pi_init = np.ones(K) * initial_velocity * precision1  # pi = M * v
    exp.set_initial_state(pi0=pi_init, agent_idx=0)

    # Run
    trajectory = exp.run(n_steps=n_steps)

    # Package results
    result = {
        't': trajectory['t'],
        'mu1': np.array([m[0] for m in trajectory['mu'][0]]),
        'mu2': np.array([m[0] for m in trajectory['mu'][1]]),
        'pi1': np.array([p[0] for p in trajectory['pi'][0]]),
        'pi2': np.array([p[0] for p in trajectory['pi'][1]]),
        'precision1': precision1,
        'precision2': precision2,
        'energy': trajectory['energy'],
    }

    result['total_momentum'] = result['pi1'] + result['pi2']
    result['momentum_diff'] = result['pi1'] - result['pi2']

    _plot_momentum_transfer_core(result, output_dir)

    return result


def _plot_momentum_transfer_core(result: Dict, output_dir: Path):
    """Visualize momentum transfer between agents (core suite version)."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = result['t']
    c1, c2 = '#e74c3c', '#3498db'

    # Panel 1: Belief trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['mu1'], color=c1, linewidth=2.5, label='Agent 1 (influencer)')
    ax1.plot(t, result['mu2'], color=c2, linewidth=2.5, label='Agent 2 (listener)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief mu', fontsize=12)
    ax1.set_title('Belief Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Momentum trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['pi1'], color=c1, linewidth=2.5, label='pi_1 (influencer)')
    ax2.plot(t, result['pi2'], color=c2, linewidth=2.5, label='pi_2 (listener)')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(t, result['pi1'], alpha=0.2, color=c1)
    ax2.fill_between(t, result['pi2'], alpha=0.2, color=c2)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Momentum pi', fontsize=12)
    ax2.set_title('Momentum Transfer (Recoil Effect)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Panel 3: Total and difference
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, result['total_momentum'], color='purple', linewidth=2.5, label='Total pi_1 + pi_2')
    ax3.plot(t, result['momentum_diff'], color='orange', linewidth=2, linestyle='--', label='Diff pi_1 - pi_2')
    ax3.axhline(result['total_momentum'][0], color='gray', linestyle=':', alpha=0.5, label='Initial total')
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Momentum', fontsize=12)
    ax3.set_title('Conservation Check', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    # Panel 4: Phase portrait Agent 1
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(result['mu1'], result['pi1'], color=c1, linewidth=2)
    ax4.plot(result['mu1'][0], result['pi1'][0], 'go', markersize=12, label='Start')
    ax4.plot(result['mu1'][-1], result['pi1'][-1], 'rs', markersize=10, label='End')
    ax4.set_xlabel('Belief mu_1', fontsize=12)
    ax4.set_ylabel('Momentum pi_1', fontsize=12)
    ax4.set_title('Phase Portrait: Agent 1', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # Panel 5: Phase portrait Agent 2
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result['mu2'], result['pi2'], color=c2, linewidth=2)
    ax5.plot(result['mu2'][0], result['pi2'][0], 'go', markersize=12, label='Start')
    ax5.plot(result['mu2'][-1], result['pi2'][-1], 'rs', markersize=10, label='End')
    ax5.set_xlabel('Belief mu_2', fontsize=12)
    ax5.set_ylabel('Momentum pi_2', fontsize=12)
    ax5.set_title('Phase Portrait: Agent 2', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Panel 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    pi1_change = result['pi1'][-1] - result['pi1'][0]
    pi2_change = result['pi2'][-1] - result['pi2'][0]

    summary = f"""
    MOMENTUM TRANSFER (CORE SUITE)

    Using REAL two-agent system:
    - MultiAgentSystem with coupling
    - HamiltonianTrainer for dynamics
    - Belief alignment energy (beta)

    Initial State:
    - Agent 1: pi_1(0) = {result['pi1'][0]:.2f}
    - Agent 2: pi_2(0) = {result['pi2'][0]:.2f}

    Final State:
    - Agent 1: pi_1(T) = {result['pi1'][-1]:.2f}
    - Agent 2: pi_2(T) = {result['pi2'][-1]:.2f}

    Changes:
    - Delta pi_1 = {pi1_change:.2f}
    - Delta pi_2 = {pi2_change:.2f}

    Precision ratio: {result['precision1']/result['precision2']:.2f}
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Two-Agent Momentum Transfer (Core Suite)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "momentum_transfer_core.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "momentum_transfer_core.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'momentum_transfer_core.png'}")


# =============================================================================
# Original Toy Model Simulations (Kept for Reference)
# =============================================================================

def simulate_damping_regimes(
    precision: float = 2.0,
    stiffness: float = 1.0,
    mu0: float = 1.0,
    pi0: float = 0.0,
    t_end: float = 20.0,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """
    Simulate three damping regimes using TOY OSCILLATOR MODEL.

    NOTE: For real multi-agent dynamics, use simulate_damping_regimes_core().
    """
    if output_dir is None:
        output_dir = Path("_experiments/damping_regimes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Critical damping value
    gamma_critical = 2 * np.sqrt(stiffness * precision)

    # Three regimes
    regimes = {
        'overdamped': gamma_critical * 3.0,
        'critical': gamma_critical,
        'underdamped': gamma_critical * 0.2,
    }

    results = {}

    print("\n" + "="*70)
    print("SIMULATION 1: THREE DAMPING REGIMES (TOY MODEL)")
    print("="*70)
    print(f"Precision (Mass) M = {precision}")
    print(f"Stiffness K = {stiffness}")
    print(f"Natural frequency omega_0 = sqrt(K/M) = {np.sqrt(stiffness/precision):.3f}")
    print(f"Critical damping gamma_c = 2*sqrt(KM) = {gamma_critical:.3f}")
    print()

    for regime_name, gamma in regimes.items():
        osc = EpistemicOscillator(
            precision=precision,
            stiffness=stiffness,
            damping=gamma,
            equilibrium=0.0
        )

        result = osc.simulate(mu0, pi0, t_end)
        result['regime'] = regime_name
        result['gamma'] = gamma
        result['damping_ratio'] = osc.damping_ratio
        result['decay_time'] = osc.decay_time
        result['oscillator'] = osc
        results[regime_name] = result

        print(f"{regime_name.upper()}:")
        print(f"  gamma = {gamma:.3f}, zeta = {osc.damping_ratio:.3f}")
        print(f"  Discriminant Delta = {osc.discriminant:.3f}")
        print(f"  Decay time tau = {osc.decay_time:.3f}")
        if regime_name == 'underdamped':
            print(f"  Damped frequency omega = {osc.damped_frequency:.3f}")
        print()

    _plot_damping_regimes(results, output_dir)

    return results


def _plot_damping_regimes(results: Dict, output_dir: Path):
    """Create comprehensive damping regimes visualization (toy model)."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'overdamped': '#e74c3c', 'critical': '#2ecc71', 'underdamped': '#3498db'}
    labels = {'overdamped': 'Overdamped (Bayesian-like)',
              'critical': 'Critical (Optimal)',
              'underdamped': 'Underdamped (Oscillatory)'}

    # Row 1: Time evolution of belief mu(t)
    ax1 = fig.add_subplot(gs[0, :2])
    for regime, result in results.items():
        ax1.plot(result['t'], result['mu'], color=colors[regime],
                 linewidth=2.5, label=labels[regime], alpha=0.9)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief mu(t)', fontsize=12)
    ax1.set_title('Belief Evolution: Three Damping Regimes (TOY MODEL)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, results['underdamped']['t'][-1])

    # Row 1 right: Energy evolution
    ax2 = fig.add_subplot(gs[0, 2])
    for regime, result in results.items():
        total_E = result['kinetic_energy'] + result['potential_energy']
        ax2.plot(result['t'], total_E, color=colors[regime],
                 linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.set_title('Energy Dissipation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    # Row 2: Phase portraits
    for idx, (regime, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])
        ax.plot(result['mu'], result['pi'], color=colors[regime], linewidth=2, alpha=0.8)
        ax.plot(result['mu'][0], result['pi'][0], 'o', color='green', markersize=12, label='Start', zorder=5)
        ax.plot(result['mu'][-1], result['pi'][-1], 's', color='red', markersize=10, label='End', zorder=5)
        ax.plot(0, 0, '*', color='gold', markersize=15, label='Equilibrium', zorder=5)
        ax.set_xlabel('Belief mu', fontsize=12)
        ax.set_ylabel('Momentum pi', fontsize=12)
        ax.set_title(f'Phase Portrait: {labels[regime]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    # Row 3: Velocity and comparison
    ax5 = fig.add_subplot(gs[2, 0])
    for regime, result in results.items():
        ax5.plot(result['t'], result['velocity'], color=colors[regime],
                 linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time t', fontsize=12)
    ax5.set_ylabel('Velocity mu_dot = pi/M', fontsize=12)
    ax5.set_title('Belief Velocity Evolution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Comparison panel
    ax6 = fig.add_subplot(gs[2, 1])
    regimes_list = list(results.keys())
    x_pos = np.arange(len(regimes_list))
    decay_times = [results[r]['decay_time'] for r in regimes_list]
    ax6.bar(x_pos, decay_times, color=[colors[r] for r in regimes_list], alpha=0.8)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([r.capitalize() for r in regimes_list])
    ax6.set_ylabel('Decay Time tau = 2M/gamma', fontsize=12)
    ax6.set_title('Characteristic Timescales', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    summary_text = """
    TOY MODEL (EpistemicOscillator):

    M mu_ddot + gamma mu_dot + K mu = 0

    Three Regimes (Delta = gamma^2 - 4KM):

    - OVERDAMPED (Delta > 0):
      Monotonic decay, no oscillation

    - CRITICAL (Delta = 0):
      Fastest convergence
      gamma_c = 2*sqrt(KM)

    - UNDERDAMPED (Delta < 0):
      Oscillatory with overshooting

    For REAL dynamics, use:
    simulate_damping_regimes_core()
    """
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Damping Regimes (Toy Model)', fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "damping_regimes.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "damping_regimes.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'damping_regimes.png'}")


def simulate_momentum_transfer(
    precision1: float = 2.0,
    precision2: float = 1.0,
    coupling: float = 0.5,
    initial_momentum1: float = 2.0,
    t_end: float = 30.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate momentum transfer between two coupled agents (TOY MODEL).

    NOTE: For real multi-agent dynamics, use simulate_momentum_transfer_core().
    """
    if output_dir is None:
        output_dir = Path("_experiments/momentum_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 2: TWO-AGENT MOMENTUM TRANSFER (TOY MODEL)")
    print("="*70)
    print(f"Agent 1: Precision = {precision1} (influencer)")
    print(f"Agent 2: Precision = {precision2} (listener)")
    print(f"Coupling beta = {coupling}")
    print()

    system = TwoAgentSystem(
        precision1=precision1,
        precision2=precision2,
        coupling=coupling,
        damping1=0.05,
        damping2=0.05,
        prior1=0.0,
        prior2=0.0,
        prior_strength=0.1
    )

    result = system.simulate(
        mu1_0=0.0,
        mu2_0=0.0,
        pi1_0=initial_momentum1,
        pi2_0=0.0,
        t_end=t_end
    )

    result['precision1'] = precision1
    result['precision2'] = precision2
    result['coupling'] = coupling

    _plot_momentum_transfer(result, output_dir)

    return result


def _plot_momentum_transfer(result: Dict, output_dir: Path):
    """Visualize momentum transfer (toy model)."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = result['t']
    c1, c2 = '#e74c3c', '#3498db'

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['mu1'], color=c1, linewidth=2.5, label='Agent 1')
    ax1.plot(t, result['mu2'], color=c2, linewidth=2.5, label='Agent 2')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Belief mu')
    ax1.set_title('Belief Trajectories (TOY)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['pi1'], color=c1, linewidth=2.5, label='pi_1')
    ax2.plot(t, result['pi2'], color=c2, linewidth=2.5, label='pi_2')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Momentum pi')
    ax2.set_title('Momentum Transfer')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, result['total_momentum'], color='purple', linewidth=2.5, label='Total')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Momentum')
    ax3.set_title('Conservation Check')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(result['mu1'], result['pi1'], color=c1, linewidth=2)
    ax4.plot(result['mu1'][0], result['pi1'][0], 'go', markersize=12)
    ax4.set_xlabel('mu_1')
    ax4.set_ylabel('pi_1')
    ax4.set_title('Phase: Agent 1')
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result['mu2'], result['pi2'], color=c2, linewidth=2)
    ax5.plot(result['mu2'][0], result['pi2'][0], 'go', markersize=12)
    ax5.set_xlabel('mu_2')
    ax5.set_ylabel('pi_2')
    ax5.set_title('Phase: Agent 2')
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.text(0.1, 0.5, "TOY MODEL\n\nUse simulate_momentum_transfer_core()\nfor real dynamics",
             transform=ax6.transAxes, fontsize=12, verticalalignment='center')

    plt.suptitle('Momentum Transfer (Toy Model)', fontsize=16, fontweight='bold')
    plt.savefig(output_dir / "momentum_transfer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'momentum_transfer.png'}")


def simulate_stopping_distance(
    precision_range: np.ndarray = None,
    initial_velocity: float = 1.0,
    counter_force: float = 0.5,
    stiffness: float = 0.0,
    damping: float = 0.1,
    t_end: float = 50.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Simulate confirmation bias as stopping distance (TOY MODEL)."""
    if output_dir is None:
        output_dir = Path("_experiments/stopping_distance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 3: STOPPING DISTANCE (TOY MODEL)")
    print("="*70)

    results = {
        'precision': precision_range,
        'stopping_distance': [],
        'theoretical_distance': [],
        'trajectories': {}
    }

    def counter_evidence(t):
        return -counter_force

    for Lambda in precision_range:
        pi0 = Lambda * initial_velocity
        osc = EpistemicOscillator(precision=Lambda, stiffness=stiffness + 0.01, damping=damping)
        result = osc.simulate(mu0=0.0, pi0=pi0, t_end=t_end, forcing=counter_evidence)

        velocity = result['velocity']
        stop_idx = np.argmax(velocity <= 0) if np.any(velocity <= 0) else len(velocity)-1
        stopping_distance = result['mu'][stop_idx]
        theoretical = Lambda * initial_velocity**2 / (2 * counter_force)

        results['stopping_distance'].append(stopping_distance)
        results['theoretical_distance'].append(theoretical)
        results['trajectories'][Lambda] = result
        print(f"Lambda = {Lambda:.1f}: d = {stopping_distance:.3f}, theory = {theoretical:.3f}")

    results['stopping_distance'] = np.array(results['stopping_distance'])
    results['theoretical_distance'] = np.array(results['theoretical_distance'])

    _plot_stopping_distance(results, output_dir)
    return results


def _plot_stopping_distance(results: Dict, output_dir: Path):
    """Plot stopping distance results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results['precision'])))
    for i, Lambda in enumerate(results['precision']):
        traj = results['trajectories'][Lambda]
        ax1.plot(traj['t'], traj['mu'], color=colors[i], label=f'Lambda={Lambda}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Belief mu')
    ax1.set_title('Trajectories')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(results['precision'], results['stopping_distance'], s=100, c='blue', label='Simulated')
    ax2.plot(results['precision'], results['theoretical_distance'], 'r--', label='Theory')
    ax2.set_xlabel('Precision Lambda')
    ax2.set_ylabel('Stopping Distance')
    ax2.set_title('d proportional to Lambda')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('Stopping Distance (Toy Model)')
    plt.tight_layout()
    plt.savefig(output_dir / "stopping_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'stopping_distance.png'}")


def simulate_resonance(
    precision: float = 2.0,
    stiffness: float = 1.0,
    damping: float = 0.3,
    forcing_amplitude: float = 0.5,
    omega_range: np.ndarray = None,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Simulate resonance curve (TOY MODEL)."""
    if output_dir is None:
        output_dir = Path("_experiments/resonance")
    output_dir.mkdir(parents=True, exist_ok=True)

    omega_0 = np.sqrt(stiffness / precision)
    if omega_range is None:
        omega_range = np.linspace(0.1, 3*omega_0, 30)

    print("\n" + "="*70)
    print("SIMULATION 4: RESONANCE (TOY MODEL)")
    print("="*70)

    results = {'omega': omega_range, 'amplitude': [], 'omega_0': omega_0}
    osc = EpistemicOscillator(precision=precision, stiffness=stiffness, damping=damping)

    for omega in omega_range:
        forcing = lambda t, w=omega: forcing_amplitude * np.cos(w * t)
        result = osc.simulate(mu0=0.0, pi0=0.0, t_end=t_end, forcing=forcing)

        steady_start = int(0.8 * len(result['t']))
        steady = result['mu'][steady_start:]
        amplitude = (np.max(steady) - np.min(steady)) / 2
        results['amplitude'].append(amplitude)

    results['amplitude'] = np.array(results['amplitude'])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(omega_range, results['amplitude'], 'b-', linewidth=2)
    plt.axvline(omega_0, color='r', linestyle='--', label=f'omega_0 = {omega_0:.3f}')
    plt.xlabel('Driving Frequency omega')
    plt.ylabel('Amplitude')
    plt.title('Resonance Curve (Toy Model)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "resonance_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'resonance_curve.png'}")

    return results


def simulate_belief_perseverance(
    precision_range: np.ndarray = None,
    damping: float = 0.5,
    initial_displacement: float = 2.0,
    stiffness: float = 1.0,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Simulate belief perseverance decay (TOY MODEL)."""
    if output_dir is None:
        output_dir = Path("_experiments/perseverance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 5: BELIEF PERSEVERANCE (TOY MODEL)")
    print("="*70)

    results = {
        'precision': precision_range,
        'measured_tau': [],
        'theoretical_tau': [],
        'trajectories': {}
    }

    threshold = 0.05

    for Lambda in precision_range:
        osc = EpistemicOscillator(precision=Lambda, stiffness=stiffness, damping=damping)
        result = osc.simulate(mu0=initial_displacement, pi0=0.0, t_end=t_end)

        below_threshold = np.abs(result['mu']) < threshold
        if np.any(below_threshold):
            tau_idx = np.argmax(below_threshold)
            measured_tau = result['t'][tau_idx]
        else:
            measured_tau = t_end

        theoretical_tau = 2 * Lambda / damping

        results['measured_tau'].append(measured_tau)
        results['theoretical_tau'].append(theoretical_tau)
        results['trajectories'][Lambda] = result
        print(f"Lambda = {Lambda:.1f}: tau = {measured_tau:.2f}, theory = {theoretical_tau:.2f}")

    results['measured_tau'] = np.array(results['measured_tau'])
    results['theoretical_tau'] = np.array(results['theoretical_tau'])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(precision_range)))
    for i, Lambda in enumerate(precision_range):
        traj = results['trajectories'][Lambda]
        axes[0].plot(traj['t'], traj['mu'], color=colors[i], label=f'Lambda={Lambda}')
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Belief')
    axes[0].set_title('Decay Trajectories')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    Lambda_over_gamma = precision_range / damping
    axes[1].scatter(Lambda_over_gamma, results['measured_tau'], s=100, c='blue', label='Measured')
    axes[1].plot(Lambda_over_gamma, results['theoretical_tau'], 'r--', label='Theory')
    axes[1].set_xlabel('Lambda/gamma')
    axes[1].set_ylabel('Decay Time tau')
    axes[1].set_title('tau proportional to Lambda/gamma')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Belief Perseverance (Toy Model)')
    plt.tight_layout()
    plt.savefig(output_dir / "belief_perseverance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'belief_perseverance.png'}")

    return results


# =============================================================================
# Master Runners
# =============================================================================

def run_all_simulations_core(output_base: Optional[Path] = None):
    """
    Run all simulations using the CORE SUITE (real agents).

    This is the recommended entry point for production use.
    """
    if output_base is None:
        output_base = Path("_experiments/psych_manuscript_core")
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EPISTEMIC MOMENTUM SIMULATIONS (CORE SUITE)")
    print("Using real Agent, MultiAgentSystem, HamiltonianTrainer")
    print("="*70 + "\n")

    all_results = {}

    # 1. Damping Regimes
    all_results['damping_regimes'] = simulate_damping_regimes_core(
        output_dir=output_base / "1_damping_regimes"
    )

    # 2. Momentum Transfer
    all_results['momentum_transfer'] = simulate_momentum_transfer_core(
        output_dir=output_base / "2_momentum_transfer"
    )

    print("\n" + "="*70)
    print("CORE SUITE SIMULATIONS COMPLETE")
    print(f"Results: {output_base}")
    print("="*70 + "\n")

    return all_results


def run_all_simulations(output_base: Optional[Path] = None):
    """
    Run all simulations using TOY MODELS (for reference/quick testing).

    For production, use run_all_simulations_core() instead.
    """
    if output_base is None:
        output_base = Path("_experiments/psych_manuscript_simulations")
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EPISTEMIC MOMENTUM SIMULATIONS (TOY MODELS)")
    print("="*70 + "\n")

    all_results = {}

    all_results['damping_regimes'] = simulate_damping_regimes(
        output_dir=output_base / "1_damping_regimes"
    )
    all_results['momentum_transfer'] = simulate_momentum_transfer(
        output_dir=output_base / "2_momentum_transfer"
    )
    all_results['stopping_distance'] = simulate_stopping_distance(
        output_dir=output_base / "3_stopping_distance"
    )
    all_results['resonance'] = simulate_resonance(
        output_dir=output_base / "4_resonance"
    )
    all_results['perseverance'] = simulate_belief_perseverance(
        output_dir=output_base / "5_perseverance"
    )

    print("\n" + "="*70)
    print("TOY MODEL SIMULATIONS COMPLETE")
    print(f"Results: {output_base}")
    print("="*70 + "\n")

    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run epistemic momentum simulations from the psych manuscript"
    )
    parser.add_argument("--sim", type=int, default=0,
                        help="Which simulation (1-5, 0=all)")
    parser.add_argument("--mode", type=str, default="core",
                        choices=["core", "toy"],
                        help="Use core suite (real agents) or toy models")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    if args.mode == "core":
        if args.sim == 0:
            run_all_simulations_core(output)
        elif args.sim == 1:
            simulate_damping_regimes_core(output_dir=output)
        elif args.sim == 2:
            simulate_momentum_transfer_core(output_dir=output)
        else:
            print(f"Core suite sim {args.sim} not yet implemented")
            print("Available: 1 (damping), 2 (momentum)")
    else:
        if args.sim == 0:
            run_all_simulations(output)
        elif args.sim == 1:
            simulate_damping_regimes(output_dir=output)
        elif args.sim == 2:
            simulate_momentum_transfer(output_dir=output)
        elif args.sim == 3:
            simulate_stopping_distance(output_dir=output)
        elif args.sim == 4:
            simulate_resonance(output_dir=output)
        elif args.sim == 5:
            simulate_belief_perseverance(output_dir=output)
        else:
            print(f"Unknown simulation: {args.sim}")
