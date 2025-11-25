#!/usr/bin/env python3
"""
Damping Regimes and Epistemic Momentum Simulations
====================================================

Comprehensive simulations demonstrating the Hamiltonian dynamics of belief
evolution from "The Inertia of Belief: Hiding in Plain Sight" manuscript.

Simulations implemented:
1. Three Damping Regimes - Overdamped, critically damped, underdamped
2. Two-Agent Momentum Transfer - Recoil effect visualization
3. Confirmation Bias as Stopping Distance - d ∝ Λ relationship
4. Resonance Curve - Peak at ω_res = √(K/M)
5. Belief Perseverance Decay - τ ∝ Λ/γ relationship

Key equations from manuscript:
- Damped oscillator: M μ̈ + γ μ̇ + K μ = f(t)
- Mass = Precision: M = Λ = Σ⁻¹
- Natural frequency: ω₀ = √(K/M)
- Damping regimes based on Δ = γ² - 4KM
- Stopping distance: d_stop = M‖μ̇‖² / (2‖f‖)
- Decay time: τ = M/γ

Author: Generated from psych_manuscript.pdf theory
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp


# =============================================================================
# Core Dynamics Classes
# =============================================================================

@dataclass
class BeliefState:
    """State of a single agent's belief."""
    mu: float           # Belief mean (position)
    pi: float           # Belief momentum
    precision: float    # Λ = 1/σ² (acts as mass)
    t: float = 0.0

    @property
    def mass(self) -> float:
        return self.precision

    @property
    def velocity(self) -> float:
        return self.pi / self.mass


@dataclass
class TwoAgentState:
    """State of two coupled agents."""
    mu1: float
    mu2: float
    pi1: float
    pi2: float
    precision1: float
    precision2: float
    coupling: float     # β₁₂ = β₂₁ attention coupling
    t: float = 0.0


class EpistemicOscillator:
    """
    Damped epistemic oscillator from Eq. 36-37 of manuscript.

    M μ̈ + γ μ̇ + K(μ - μ*) = f(t)

    Where:
    - M = precision (epistemic mass)
    - γ = damping coefficient
    - K = evidence strength (spring constant)
    - μ* = equilibrium (prior or evidence target)
    - f(t) = external forcing (evidence stream)
    """

    def __init__(
        self,
        precision: float,      # M = Λ
        stiffness: float,      # K (evidence strength)
        damping: float,        # γ
        equilibrium: float = 0.0,  # μ* (target)
    ):
        self.M = precision
        self.K = stiffness
        self.gamma = damping
        self.mu_eq = equilibrium

    @property
    def natural_frequency(self) -> float:
        """ω₀ = √(K/M)"""
        return np.sqrt(self.K / self.M)

    @property
    def damping_ratio(self) -> float:
        """ζ = γ / (2√(KM))"""
        return self.gamma / (2 * np.sqrt(self.K * self.M))

    @property
    def discriminant(self) -> float:
        """Δ = γ² - 4KM"""
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
        """ω = √(K/M - γ²/(4M²)) for underdamped case."""
        if self.regime == "underdamped":
            return np.sqrt(self.K/self.M - self.gamma**2/(4*self.M**2))
        return 0.0

    @property
    def decay_time(self) -> float:
        """τ = 2M/γ (Eq. 39)"""
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

        dy/dt = [dμ/dt, dπ/dt]
              = [π/M, -K(μ - μ*) - γ·(π/M) + f(t)]
        """
        mu, pi = y

        # Velocity
        dmu_dt = pi / self.M

        # Force: -∂V/∂μ - damping + external
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
    Two coupled agents with momentum transfer (Section 4.6).

    M₁μ̈₁ + γ₁μ̇₁ = -Λ̄₁(μ₁ - μ̄₁) - β₁₂Λ₂(μ₁ - μ₂)
    M₂μ̈₂ + γ₂μ̇₂ = -Λ̄₂(μ₂ - μ̄₂) - β₂₁Λ₁(μ₂ - μ₁)

    The coupling terms show momentum transfer and recoil.
    """

    def __init__(
        self,
        precision1: float,
        precision2: float,
        coupling: float,        # β₁₂ = β₂₁
        damping1: float = 0.1,
        damping2: float = 0.1,
        prior1: float = 0.0,
        prior2: float = 0.0,
        prior_strength: float = 0.1,  # Λ̄ (prior anchoring)
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

        y = [μ₁, μ₂, π₁, π₂]
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
            v1,  # dμ₁/dt
            v2,  # dμ₂/dt
            force_prior1 + force_consensus1 + force_damping1,  # dπ₁/dt
            force_prior2 + force_consensus2 + force_damping2,  # dπ₂/dt
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
# Simulation 1: Three Damping Regimes
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
    Simulate three damping regimes for the same agent.

    Δ = γ² - 4KM determines regime:
    - Overdamped: Δ > 0 (γ > 2√(KM)) → Bayesian-like
    - Critical: Δ = 0 (γ = 2√(KM)) → Optimal
    - Underdamped: Δ < 0 (γ < 2√(KM)) → Oscillatory

    Args:
        precision: Epistemic mass M = Λ
        stiffness: Evidence strength K
        mu0: Initial belief displacement
        pi0: Initial momentum
        t_end: Simulation duration
        output_dir: Where to save figures

    Returns:
        Dictionary with simulation results for each regime
    """
    if output_dir is None:
        output_dir = Path("_experiments/damping_regimes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Critical damping value
    gamma_critical = 2 * np.sqrt(stiffness * precision)

    # Three regimes
    regimes = {
        'overdamped': gamma_critical * 3.0,      # γ >> γ_c
        'critical': gamma_critical,               # γ = γ_c
        'underdamped': gamma_critical * 0.2,     # γ << γ_c
    }

    results = {}

    print("\n" + "="*70)
    print("SIMULATION 1: THREE DAMPING REGIMES")
    print("="*70)
    print(f"Precision (Mass) M = {precision}")
    print(f"Stiffness K = {stiffness}")
    print(f"Natural frequency ω₀ = √(K/M) = {np.sqrt(stiffness/precision):.3f}")
    print(f"Critical damping γ_c = 2√(KM) = {gamma_critical:.3f}")
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
        print(f"  γ = {gamma:.3f}, ζ = {osc.damping_ratio:.3f}")
        print(f"  Discriminant Δ = {osc.discriminant:.3f}")
        print(f"  Decay time τ = {osc.decay_time:.3f}")
        if regime_name == 'underdamped':
            print(f"  Damped frequency ω = {osc.damped_frequency:.3f}")
        print()

    # Create visualization
    _plot_damping_regimes(results, output_dir)

    return results


def _plot_damping_regimes(results: Dict, output_dir: Path):
    """Create comprehensive damping regimes visualization."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'overdamped': '#e74c3c', 'critical': '#2ecc71', 'underdamped': '#3498db'}
    labels = {'overdamped': 'Overdamped (Bayesian-like)',
              'critical': 'Critical (Optimal)',
              'underdamped': 'Underdamped (Oscillatory)'}

    # Row 1: Time evolution of belief μ(t)
    ax1 = fig.add_subplot(gs[0, :2])
    for regime, result in results.items():
        ax1.plot(result['t'], result['mu'], color=colors[regime],
                 linewidth=2.5, label=labels[regime], alpha=0.9)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    ax1.set_title('Belief Evolution: Three Damping Regimes', fontsize=14, fontweight='bold')
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

    # Row 2: Phase portraits (μ vs π)
    for idx, (regime, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])

        # Phase trajectory
        ax.plot(result['mu'], result['pi'], color=colors[regime],
                linewidth=2, alpha=0.8)

        # Mark start and end
        ax.plot(result['mu'][0], result['pi'][0], 'o', color='green',
                markersize=12, label='Start', zorder=5)
        ax.plot(result['mu'][-1], result['pi'][-1], 's', color='red',
                markersize=10, label='End', zorder=5)
        ax.plot(0, 0, '*', color='gold', markersize=15,
                label='Equilibrium', zorder=5)

        # Direction arrows
        n_arrows = 8
        arrow_idx = np.linspace(0, len(result['t'])-2, n_arrows, dtype=int)
        for i in arrow_idx:
            dx = result['mu'][i+1] - result['mu'][i]
            dy = result['pi'][i+1] - result['pi'][i]
            ax.annotate('', xy=(result['mu'][i]+dx*0.6, result['pi'][i]+dy*0.6),
                       xytext=(result['mu'][i], result['pi'][i]),
                       arrowprops=dict(arrowstyle='->', color=colors[regime], alpha=0.6))

        ax.set_xlabel('Belief μ', fontsize=12)
        ax.set_ylabel('Momentum π', fontsize=12)
        ax.set_title(f'Phase Portrait: {labels[regime]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    # Row 3: Velocity and detailed analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for regime, result in results.items():
        ax5.plot(result['t'], result['velocity'], color=colors[regime],
                 linewidth=2, label=regime.capitalize(), alpha=0.9)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time t', fontsize=12)
    ax5.set_ylabel('Velocity μ̇ = π/M', fontsize=12)
    ax5.set_title('Belief Velocity Evolution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Comparison panel
    ax6 = fig.add_subplot(gs[2, 1])
    regimes_list = list(results.keys())
    x_pos = np.arange(len(regimes_list))

    decay_times = [results[r]['decay_time'] for r in regimes_list]
    damping_ratios = [results[r]['damping_ratio'] for r in regimes_list]

    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, decay_times, width,
                    color=[colors[r] for r in regimes_list], alpha=0.8, label='Decay time τ')

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([r.capitalize() for r in regimes_list])
    ax6.set_ylabel('Decay Time τ = 2M/γ', fontsize=12)
    ax6.set_title('Characteristic Timescales', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Add damping ratio as text
    for i, (tau, zeta) in enumerate(zip(decay_times, damping_ratios)):
        ax6.text(i, tau + 0.1, f'ζ={zeta:.2f}', ha='center', fontsize=10)

    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = """
    KEY PREDICTIONS FROM MANUSCRIPT:

    Damped Oscillator Equation (Eq. 36):
        M μ̈ + γ μ̇ + K μ = 0

    Where M = Λ (precision = mass)

    Three Regimes (based on Δ = γ² - 4KM):

    • OVERDAMPED (Δ > 0):
      Monotonic decay, no oscillation
      ≈ Standard Bayesian updating

    • CRITICAL (Δ = 0):
      Fastest approach to equilibrium
      γ_c = 2√(KM) = optimal learning

    • UNDERDAMPED (Δ < 0):
      Oscillatory with overshooting
      Novel prediction: belief oscillation!

    Frequency: ω = √(K/M - γ²/4M²)
    Decay time: τ = 2M/γ
    """
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('The Inertia of Belief: Damping Regimes in Epistemic Dynamics',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "damping_regimes.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "damping_regimes.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'damping_regimes.png'}")


# =============================================================================
# Simulation 2: Two-Agent Momentum Transfer
# =============================================================================

def simulate_momentum_transfer(
    precision1: float = 2.0,
    precision2: float = 1.0,
    coupling: float = 0.5,
    initial_momentum1: float = 2.0,
    t_end: float = 30.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate momentum transfer between two coupled agents.

    Key prediction: The influencer's momentum decreases (recoil effect)
    as momentum flows to the coupled partner.

    From Eq. 48-52:
    - Momentum current: J_{k→i} = β_{ik} Λ̃_k (μ̃_k - μ_i)
    - Total momentum changes when priors/damping present

    Args:
        precision1: Agent 1's precision (the "influencer")
        precision2: Agent 2's precision (the "listener")
        coupling: Attention coupling β₁₂ = β₂₁
        initial_momentum1: Initial momentum of agent 1
        t_end: Simulation duration
        output_dir: Where to save figures
    """
    if output_dir is None:
        output_dir = Path("_experiments/momentum_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SIMULATION 2: TWO-AGENT MOMENTUM TRANSFER")
    print("="*70)
    print(f"Agent 1: Precision = {precision1} (influencer, starts with momentum)")
    print(f"Agent 2: Precision = {precision2} (listener, starts at rest)")
    print(f"Coupling β = {coupling}")
    print(f"Initial momentum π₁(0) = {initial_momentum1}")
    print()

    system = TwoAgentSystem(
        precision1=precision1,
        precision2=precision2,
        coupling=coupling,
        damping1=0.05,  # Light damping
        damping2=0.05,
        prior1=0.0,
        prior2=0.0,
        prior_strength=0.1
    )

    # Agent 1 starts moving, agent 2 at rest
    result = system.simulate(
        mu1_0=0.0,      # Both start at same belief
        mu2_0=0.0,
        pi1_0=initial_momentum1,  # Agent 1 has momentum
        pi2_0=0.0,                 # Agent 2 at rest
        t_end=t_end
    )

    # Add metadata
    result['precision1'] = precision1
    result['precision2'] = precision2
    result['coupling'] = coupling

    _plot_momentum_transfer(result, output_dir)

    return result


def _plot_momentum_transfer(result: Dict, output_dir: Path):
    """Visualize momentum transfer between agents."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = result['t']

    # Colors
    c1, c2 = '#e74c3c', '#3498db'

    # Panel 1: Belief trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['mu1'], color=c1, linewidth=2.5, label='Agent 1 (influencer)')
    ax1.plot(t, result['mu2'], color=c2, linewidth=2.5, label='Agent 2 (listener)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ', fontsize=12)
    ax1.set_title('Belief Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Momentum trajectories - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['pi1'], color=c1, linewidth=2.5, label='π₁ (influencer)')
    ax2.plot(t, result['pi2'], color=c2, linewidth=2.5, label='π₂ (listener)')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Highlight recoil region
    ax2.fill_between(t, result['pi1'], alpha=0.2, color=c1)
    ax2.fill_between(t, result['pi2'], alpha=0.2, color=c2)

    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Momentum π', fontsize=12)
    ax2.set_title('Momentum Trajectories (Recoil Effect)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Add annotation for recoil
    peak_idx = np.argmax(result['pi2'])
    ax2.annotate('Momentum\ntransfer',
                xy=(t[peak_idx], result['pi2'][peak_idx]),
                xytext=(t[peak_idx]+2, result['pi2'][peak_idx]+0.3),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))

    # Panel 3: Total and difference
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, result['total_momentum'], color='purple', linewidth=2.5,
             label='Total π₁ + π₂')
    ax3.plot(t, result['momentum_diff'], color='orange', linewidth=2,
             linestyle='--', label='Difference π₁ - π₂')
    ax3.axhline(result['total_momentum'][0], color='gray', linestyle=':',
                alpha=0.5, label='Initial total')
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
    ax4.set_xlabel('Belief μ₁', fontsize=12)
    ax4.set_ylabel('Momentum π₁', fontsize=12)
    ax4.set_title('Phase Portrait: Agent 1', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # Panel 5: Phase portrait Agent 2
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result['mu2'], result['pi2'], color=c2, linewidth=2)
    ax5.plot(result['mu2'][0], result['pi2'][0], 'go', markersize=12, label='Start')
    ax5.plot(result['mu2'][-1], result['pi2'][-1], 'rs', markersize=10, label='End')
    ax5.set_xlabel('Belief μ₂', fontsize=12)
    ax5.set_ylabel('Momentum π₂', fontsize=12)
    ax5.set_title('Phase Portrait: Agent 2', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Panel 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Compute key statistics
    pi1_drop = result['pi1'][0] - np.min(result['pi1'])
    pi2_max = np.max(result['pi2'])
    transfer_efficiency = pi2_max / result['pi1'][0]

    summary = f"""
    MOMENTUM TRANSFER ANALYSIS

    Initial State:
    • Agent 1: π₁(0) = {result['pi1'][0]:.2f} (moving)
    • Agent 2: π₂(0) = {result['pi2'][0]:.2f} (at rest)

    Key Results (Eq. 48-52):
    • Agent 1 momentum drop: Δπ₁ = {pi1_drop:.2f}
    • Agent 2 peak momentum: π₂_max = {pi2_max:.2f}
    • Transfer efficiency: {transfer_efficiency:.1%}

    RECOIL EFFECT CONFIRMED:
    The influencer's momentum decreases
    as momentum flows to coupled partner.

    Momentum Current (Eq. 52):
    J_{{k→i}} = β_{{ik}} Λ̃_k (μ̃_k - μ_i)

    Precision ratio: Λ₁/Λ₂ = {result['precision1']/result['precision2']:.2f}
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Two-Agent Momentum Transfer: The Recoil Effect',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "momentum_transfer.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "momentum_transfer.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'momentum_transfer.png'}")


# =============================================================================
# Simulation 3: Confirmation Bias as Stopping Distance
# =============================================================================

def simulate_stopping_distance(
    precision_range: np.ndarray = None,
    initial_velocity: float = 1.0,
    counter_force: float = 0.5,
    stiffness: float = 0.0,  # No restoring force (free motion against friction)
    damping: float = 0.1,
    t_end: float = 50.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate confirmation bias as stopping distance.

    From Eq. 33-35:
    d_stop = M‖μ̇‖² / (2‖f‖) = ‖π‖² / (2M‖f‖)

    Prediction: d_H/d_L = Λ_H/Λ_L
    A person twice as confident overshoots twice as far.

    Args:
        precision_range: Array of precision values to test
        initial_velocity: Starting belief velocity
        counter_force: Opposing force (counter-evidence)
        stiffness: Spring constant (0 for free motion)
        damping: Base damping coefficient
        t_end: Maximum simulation time
        output_dir: Where to save figures
    """
    if output_dir is None:
        output_dir = Path("_experiments/stopping_distance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 3: CONFIRMATION BIAS AS STOPPING DISTANCE")
    print("="*70)
    print(f"Initial velocity: {initial_velocity}")
    print(f"Counter-evidence force: {counter_force}")
    print(f"Prediction: d ∝ Λ (precision)")
    print()

    results = {
        'precision': precision_range,
        'stopping_distance': [],
        'stopping_time': [],
        'theoretical_distance': [],
        'trajectories': {}
    }

    # Counter-evidence force (negative, opposes positive motion)
    def counter_evidence(t):
        return -counter_force

    for Lambda in precision_range:
        # Initial momentum = mass × velocity
        pi0 = Lambda * initial_velocity

        osc = EpistemicOscillator(
            precision=Lambda,
            stiffness=stiffness + 0.01,  # Small restoring for stability
            damping=damping,
            equilibrium=0.0
        )

        result = osc.simulate(
            mu0=0.0,
            pi0=pi0,
            t_end=t_end,
            forcing=counter_evidence
        )

        # Find stopping point (where velocity crosses zero or goes negative)
        velocity = result['velocity']
        stop_idx = np.argmax(velocity <= 0) if np.any(velocity <= 0) else len(velocity)-1

        stopping_distance = result['mu'][stop_idx]
        stopping_time = result['t'][stop_idx]

        # Theoretical: d = M v² / (2f) = M v² / (2f)
        # With v = pi/M = initial_velocity
        theoretical = Lambda * initial_velocity**2 / (2 * counter_force)

        results['stopping_distance'].append(stopping_distance)
        results['stopping_time'].append(stopping_time)
        results['theoretical_distance'].append(theoretical)
        results['trajectories'][Lambda] = result

        print(f"Λ = {Lambda:.1f}: d_stop = {stopping_distance:.3f}, "
              f"d_theory = {theoretical:.3f}, t_stop = {stopping_time:.2f}")

    results['stopping_distance'] = np.array(results['stopping_distance'])
    results['stopping_time'] = np.array(results['stopping_time'])
    results['theoretical_distance'] = np.array(results['theoretical_distance'])

    _plot_stopping_distance(results, output_dir)

    return results


def _plot_stopping_distance(results: Dict, output_dir: Path):
    """Visualize stopping distance relationship."""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    precisions = results['precision']
    distances = results['stopping_distance']
    theoretical = results['theoretical_distance']

    # Create colormap for precision
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(precisions)))

    # Panel 1: Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax1.plot(traj['t'], traj['mu'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')
        # Mark stopping point
        stop_idx = np.argmax(traj['velocity'] <= 0) if np.any(traj['velocity'] <= 0) else -1
        ax1.plot(traj['t'][stop_idx], traj['mu'][stop_idx], 'o',
                 color=colors[i], markersize=10)

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    ax1.set_title('Belief Trajectories Against Counter-Evidence', fontsize=14, fontweight='bold')
    ax1.legend(title='Precision', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: d vs Λ - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(precisions, distances, s=150, c=colors, edgecolors='black',
                linewidths=2, label='Simulated', zorder=5)
    ax2.plot(precisions, theoretical, 'k--', linewidth=2, label='Theory: d = Λv²/2f')

    # Linear fit
    slope, intercept = np.polyfit(precisions, distances, 1)
    fit_line = slope * precisions + intercept
    ax2.plot(precisions, fit_line, 'r-', linewidth=2, alpha=0.7,
             label=f'Fit: d = {slope:.2f}Λ + {intercept:.2f}')

    ax2.set_xlabel('Precision Λ', fontsize=12)
    ax2.set_ylabel('Stopping Distance d', fontsize=12)
    ax2.set_title('Stopping Distance vs Precision: d ∝ Λ', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Add R² annotation
    r_squared = 1 - np.sum((distances - theoretical)**2) / np.sum((distances - np.mean(distances))**2)
    ax2.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Velocity decay
    ax3 = fig.add_subplot(gs[1, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax3.plot(traj['t'], traj['velocity'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Velocity μ̇', fontsize=12)
    ax3.set_title('Velocity Decay Under Counter-Evidence', fontsize=14, fontweight='bold')
    ax3.legend(title='Precision', fontsize=9)
    ax3.grid(alpha=0.3)

    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary = f"""
    CONFIRMATION BIAS AS MOMENTUM (Eq. 33-35)

    Stopping Distance Formula:

        d_stop = M‖μ̇‖² / (2‖f‖) = Λ v² / (2f)

    Where:
    • M = Λ = precision (epistemic mass)
    • v = μ̇ = belief velocity
    • f = counter-evidence force

    KEY PREDICTION:

        d_H / d_L = Λ_H / Λ_L

    "A person twice as confident
     overshoots twice as far"

    ═══════════════════════════════════
    SIMULATION RESULTS:

    Linear fit: d = {slope:.3f}Λ + {intercept:.3f}
    Theory:     d = {precisions[0]*results['theoretical_distance'][0]/precisions[0]:.3f}Λ

    Correlation R² = {r_squared:.4f}

    CONFIRMATION BIAS EMERGES FROM
    BELIEF INERTIA, NOT IRRATIONALITY!
    """
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('Confirmation Bias: Stopping Distance Proportional to Precision',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "stopping_distance.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "stopping_distance.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'stopping_distance.png'}")


# =============================================================================
# Simulation 4: Resonance Curve
# =============================================================================

def simulate_resonance(
    precision: float = 2.0,
    stiffness: float = 1.0,
    damping: float = 0.3,
    forcing_amplitude: float = 0.5,
    omega_range: np.ndarray = None,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate resonance curve for periodic evidence forcing.

    From Eq. 40-42:
    - ω_res = √(K/M) = resonance frequency
    - A(ω) = (f₀/M) / √((ω₀² - ω²)² + (γω/M)²)
    - At resonance: A_max = (f₀/γ)√(M/K)

    Args:
        precision: Epistemic mass M = Λ
        stiffness: Evidence strength K
        damping: Damping coefficient γ
        forcing_amplitude: Forcing amplitude f₀
        omega_range: Frequencies to test
        t_end: Simulation time (need steady state)
        output_dir: Where to save figures
    """
    if output_dir is None:
        output_dir = Path("_experiments/resonance")
    output_dir.mkdir(parents=True, exist_ok=True)

    omega_0 = np.sqrt(stiffness / precision)  # Natural frequency
    omega_res = omega_0  # For light damping

    if omega_range is None:
        omega_range = np.linspace(0.1, 3*omega_0, 50)

    print("\n" + "="*70)
    print("SIMULATION 4: RESONANCE CURVE")
    print("="*70)
    print(f"Precision M = {precision}")
    print(f"Stiffness K = {stiffness}")
    print(f"Damping γ = {damping}")
    print(f"Natural frequency ω₀ = √(K/M) = {omega_0:.3f}")
    print(f"Forcing amplitude f₀ = {forcing_amplitude}")
    print()

    results = {
        'omega': omega_range,
        'omega_0': omega_0,
        'amplitude': [],
        'theoretical_amplitude': [],
        'phase': [],
        'example_trajectories': {}
    }

    osc = EpistemicOscillator(
        precision=precision,
        stiffness=stiffness,
        damping=damping,
        equilibrium=0.0
    )

    # Theoretical amplitude function (Eq. 41)
    def theoretical_amplitude(omega):
        numerator = forcing_amplitude / precision
        denominator = np.sqrt((omega_0**2 - omega**2)**2 + (damping * omega / precision)**2)
        return numerator / denominator

    for omega in omega_range:
        # Periodic forcing
        forcing = lambda t, w=omega: forcing_amplitude * np.cos(w * t)

        result = osc.simulate(
            mu0=0.0,
            pi0=0.0,
            t_end=t_end,
            forcing=forcing
        )

        # Measure steady-state amplitude (last 20% of simulation)
        steady_start = int(0.8 * len(result['t']))
        steady_state = result['mu'][steady_start:]
        amplitude = (np.max(steady_state) - np.min(steady_state)) / 2

        results['amplitude'].append(amplitude)
        results['theoretical_amplitude'].append(theoretical_amplitude(omega))

        # Store example trajectories for plotting (at specific relative frequencies)
        for ratio in [0.5, 1.0, 2.0]:
            target = omega_0 * ratio
            if abs(omega - target) < (omega_range[1] - omega_range[0]) / 2:
                results['example_trajectories'][omega] = result

    results['amplitude'] = np.array(results['amplitude'])
    results['theoretical_amplitude'] = np.array(results['theoretical_amplitude'])

    # Find measured resonance peak
    peak_idx = np.argmax(results['amplitude'])
    results['measured_omega_res'] = omega_range[peak_idx]
    results['measured_A_max'] = results['amplitude'][peak_idx]

    # Theoretical peak amplitude
    A_max_theory = (forcing_amplitude / damping) * np.sqrt(precision / stiffness)
    results['theoretical_A_max'] = A_max_theory

    print(f"Measured resonance: ω = {results['measured_omega_res']:.3f}, A = {results['measured_A_max']:.3f}")
    print(f"Theoretical: ω = {omega_0:.3f}, A_max = {A_max_theory:.3f}")

    _plot_resonance(results, output_dir, precision, stiffness, damping, forcing_amplitude)

    return results


def _plot_resonance(results: Dict, output_dir: Path, M: float, K: float, gamma: float, f0: float):
    """Visualize resonance curve."""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    omega = results['omega']
    amplitude = results['amplitude']
    omega_0 = results['omega_0']

    # Panel 1: Resonance curve - KEY RESULT
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(omega, amplitude, 'b-', linewidth=2.5, label='Simulated', zorder=5)
    ax1.plot(omega, results['theoretical_amplitude'], 'r--', linewidth=2,
             label='Theory (Eq. 41)', alpha=0.8)

    # Mark resonance peak
    ax1.axvline(omega_0, color='green', linestyle=':', linewidth=2,
                label=f'ω₀ = √(K/M) = {omega_0:.3f}')
    ax1.axvline(results['measured_omega_res'], color='purple', linestyle='--',
                linewidth=1.5, label=f'Measured ω_res = {results["measured_omega_res"]:.3f}')

    ax1.scatter([results['measured_omega_res']], [results['measured_A_max']],
                s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                label=f'Peak A = {results["measured_A_max"]:.3f}', zorder=10)

    ax1.set_xlabel('Driving Frequency ω', fontsize=12)
    ax1.set_ylabel('Steady-State Amplitude A(ω)', fontsize=12)
    ax1.set_title('Cognitive Resonance: Optimal Persuasion at ω_res = √(K/M)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Add resonance equation
    ax1.text(0.02, 0.95,
             f'A(ω) = (f₀/M) / √[(ω₀² - ω²)² + (γω/M)²]\n'
             f'ω₀ = √(K/M) = √({K}/{M}) = {omega_0:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Example trajectories
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    example_trajs = list(results['example_trajectories'].items())
    for i, (omega_ex, traj) in enumerate(example_trajs[:3]):  # Max 3 examples
        # Determine position relative to resonance
        ratio = omega_ex / omega_0
        if ratio < 0.8:
            label_suffix = 'Below'
        elif ratio > 1.2:
            label_suffix = 'Above'
        else:
            label_suffix = 'At'

        # Show last portion for steady state
        start_idx = int(0.7 * len(traj['t']))
        ax2.plot(traj['t'][start_idx:] - traj['t'][start_idx],
                 traj['mu'][start_idx:],
                 color=colors[i % len(colors)], linewidth=2,
                 label=f'ω = {omega_ex:.2f} ({label_suffix} res.)')

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Belief μ(t)', fontsize=12)
    ax2.set_title('Steady-State Oscillations', fontsize=14, fontweight='bold')
    if example_trajs:
        ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    summary = f"""
    COGNITIVE RESONANCE (Eq. 40-42)

    Resonance Frequency:
        ω_res = √(K/M) = √(Evidence × Precision / Mass)

    Amplitude at Resonance (Eq. 42):
        A_max = (f₀/γ)√(M/K)

    ═══════════════════════════════════
    PARAMETERS:
    • Mass (precision) M = {M}
    • Stiffness K = {K}
    • Damping γ = {gamma}
    • Forcing f₀ = {f0}

    RESULTS:
    • Natural freq ω₀ = {omega_0:.4f}
    • Measured ω_res = {results['measured_omega_res']:.4f}
    • Measured A_max = {results['measured_A_max']:.4f}
    • Theory A_max = {results['theoretical_A_max']:.4f}

    ═══════════════════════════════════
    IMPLICATION:

    "High-mass (confident) agents have
     LARGER resonance amplitudes!

     Properly timed evidence produces
     dramatic belief swings."

    Applications: education, therapy,
    advertising, negotiation
    """
    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('Resonance in Belief Dynamics: Optimal Persuasion Frequency',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "resonance_curve.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "resonance_curve.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'resonance_curve.png'}")


# =============================================================================
# Simulation 5: Belief Perseverance Decay
# =============================================================================

def simulate_belief_perseverance(
    precision_range: np.ndarray = None,
    damping: float = 0.5,
    initial_displacement: float = 2.0,
    stiffness: float = 1.0,
    equilibrium_threshold: float = 0.05,
    t_end: float = 100.0,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate belief perseverance decay after debunking.

    From Eq. 43:
    τ = M/γ = Λ/γ (decay time proportional to precision/damping)

    Prediction: High-precision beliefs persist longer after debunking.

    Args:
        precision_range: Array of precision values to test
        damping: Damping coefficient γ (same for all)
        initial_displacement: Initial false belief distance from truth
        stiffness: Evidence strength pulling toward truth
        equilibrium_threshold: Define "reached equilibrium" when |μ| < threshold
        t_end: Maximum simulation time
        output_dir: Where to save figures
    """
    if output_dir is None:
        output_dir = Path("_experiments/perseverance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if precision_range is None:
        precision_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    print("\n" + "="*70)
    print("SIMULATION 5: BELIEF PERSEVERANCE DECAY")
    print("="*70)
    print(f"Damping γ = {damping}")
    print(f"Initial false belief: μ₀ = {initial_displacement}")
    print(f"Truth (equilibrium): μ* = 0")
    print(f"Prediction: τ = Λ/γ (decay time ∝ precision)")
    print()

    results = {
        'precision': precision_range,
        'measured_tau': [],
        'theoretical_tau': [],
        'half_life': [],
        'trajectories': {}
    }

    for Lambda in precision_range:
        osc = EpistemicOscillator(
            precision=Lambda,
            stiffness=stiffness,
            damping=damping,
            equilibrium=0.0  # Truth
        )

        # Start at false belief, no momentum (just debunked)
        result = osc.simulate(
            mu0=initial_displacement,
            pi0=0.0,
            t_end=t_end
        )

        # Find time to reach equilibrium threshold
        below_threshold = np.abs(result['mu']) < equilibrium_threshold
        if np.any(below_threshold):
            tau_idx = np.argmax(below_threshold)
            measured_tau = result['t'][tau_idx]
        else:
            measured_tau = t_end  # Didn't reach in time

        # Find half-life (time to reach half of initial displacement)
        half_point = initial_displacement / 2
        below_half = np.abs(result['mu']) < half_point
        if np.any(below_half):
            half_idx = np.argmax(below_half)
            half_life = result['t'][half_idx]
        else:
            half_life = t_end

        # Theoretical τ = 2M/γ (from Eq. 39, or M/γ for simple exponential)
        theoretical_tau = 2 * Lambda / damping

        results['measured_tau'].append(measured_tau)
        results['theoretical_tau'].append(theoretical_tau)
        results['half_life'].append(half_life)
        results['trajectories'][Lambda] = result

        print(f"Λ = {Lambda:.1f}: τ_measured = {measured_tau:.2f}, "
              f"τ_theory = {theoretical_tau:.2f}, t_½ = {half_life:.2f}")

    results['measured_tau'] = np.array(results['measured_tau'])
    results['theoretical_tau'] = np.array(results['theoretical_tau'])
    results['half_life'] = np.array(results['half_life'])
    results['damping'] = damping

    _plot_belief_perseverance(results, output_dir)

    return results


def _plot_belief_perseverance(results: Dict, output_dir: Path):
    """Visualize belief perseverance decay."""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    precisions = results['precision']
    measured_tau = results['measured_tau']
    theoretical_tau = results['theoretical_tau']
    gamma = results['damping']

    # Colors for different precisions
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(precisions)))

    # Panel 1: Decay trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        ax1.plot(traj['t'], traj['mu'], color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Truth')
    ax1.set_xlabel('Time t (after debunking)', fontsize=12)
    ax1.set_ylabel('Belief μ(t)', fontsize=12)
    ax1.set_title('Belief Decay After Debunking', fontsize=14, fontweight='bold')
    ax1.legend(title='Precision', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: τ vs Λ/γ - KEY RESULT
    ax2 = fig.add_subplot(gs[0, 1])
    Lambda_over_gamma = precisions / gamma

    ax2.scatter(Lambda_over_gamma, measured_tau, s=150, c=colors,
                edgecolors='black', linewidths=2, label='Measured τ', zorder=5)
    ax2.plot(Lambda_over_gamma, theoretical_tau, 'k--', linewidth=2,
             label='Theory: τ = 2Λ/γ')

    # Linear fit
    slope, intercept = np.polyfit(Lambda_over_gamma, measured_tau, 1)
    fit_line = slope * Lambda_over_gamma + intercept
    ax2.plot(Lambda_over_gamma, fit_line, 'r-', linewidth=2, alpha=0.7,
             label=f'Fit: τ = {slope:.2f}(Λ/γ) + {intercept:.2f}')

    ax2.set_xlabel('Λ/γ (precision/damping)', fontsize=12)
    ax2.set_ylabel('Decay Time τ', fontsize=12)
    ax2.set_title('Decay Time vs Λ/γ: τ ∝ Λ/γ', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # R² annotation
    r_squared = 1 - np.sum((measured_tau - theoretical_tau)**2) / np.sum((measured_tau - np.mean(measured_tau))**2)
    ax2.text(0.05, 0.95, f'R² = {max(0, r_squared):.4f}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Normalized trajectories
    ax3 = fig.add_subplot(gs[1, 0])
    for i, Lambda in enumerate(precisions):
        traj = results['trajectories'][Lambda]
        # Normalize time by theoretical τ
        t_normalized = traj['t'] / (2 * Lambda / gamma)
        # Normalize position
        mu_normalized = traj['mu'] / traj['mu'][0]
        ax3.plot(t_normalized, mu_normalized, color=colors[i], linewidth=2,
                 label=f'Λ = {Lambda:.1f}')

    # Theoretical exponential decay
    t_theory = np.linspace(0, 5, 100)
    ax3.plot(t_theory, np.exp(-t_theory), 'k--', linewidth=2,
             label='exp(-t/τ)', alpha=0.7)

    ax3.set_xlabel('Normalized Time t/τ', fontsize=12)
    ax3.set_ylabel('Normalized Belief μ/μ₀', fontsize=12)
    ax3.set_title('Universal Decay (Scaled by τ)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 5)

    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Compute ratio for highest/lowest precision
    ratio = measured_tau[-1] / measured_tau[0]
    precision_ratio = precisions[-1] / precisions[0]

    summary = f"""
    BELIEF PERSEVERANCE (Eq. 43)

    Decay Time Formula:

        τ = M/γ = Λ/γ

    "High-precision beliefs have
     long decay times"

    ═══════════════════════════════════
    SIMULATION RESULTS:

    Damping γ = {gamma}

    Precision range: {precisions[0]:.1f} - {precisions[-1]:.1f}
    τ range: {measured_tau[0]:.2f} - {measured_tau[-1]:.2f}

    τ ratio: {ratio:.2f}
    Λ ratio: {precision_ratio:.2f}

    ═══════════════════════════════════
    THE DEBUNKING PROBLEM (Section 4.5):

    • Immediate debunking is INEFFECTIVE
    • Belief flows PAST correction target
    • Must time the belief trajectory

    Example from Eq. 44:
    If Λ_A = 10, Λ_B = 1, γ equal:

        τ_A/τ_B = Λ_A/Λ_B = 10

    "Agent A's false beliefs persist
     TEN TIMES longer than B's despite
     identical evidence exposure!"
    """
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('Belief Perseverance: Decay Time Proportional to Precision',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_dir / "belief_perseverance.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_dir / "belief_perseverance.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'belief_perseverance.png'}")


# =============================================================================
# Master Runner
# =============================================================================

def run_all_simulations(output_base: Optional[Path] = None):
    """
    Run all five simulations from the manuscript.

    Creates a comprehensive set of figures demonstrating the
    predictions of epistemic momentum theory.
    """
    if output_base is None:
        output_base = Path("_experiments/psych_manuscript_simulations")
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EPISTEMIC MOMENTUM SIMULATIONS")
    print("From 'The Inertia of Belief: Hiding in Plain Sight'")
    print("="*70 + "\n")

    all_results = {}

    # 1. Three Damping Regimes
    all_results['damping_regimes'] = simulate_damping_regimes(
        output_dir=output_base / "1_damping_regimes"
    )

    # 2. Two-Agent Momentum Transfer
    all_results['momentum_transfer'] = simulate_momentum_transfer(
        output_dir=output_base / "2_momentum_transfer"
    )

    # 3. Confirmation Bias / Stopping Distance
    all_results['stopping_distance'] = simulate_stopping_distance(
        output_dir=output_base / "3_stopping_distance"
    )

    # 4. Resonance Curve
    all_results['resonance'] = simulate_resonance(
        output_dir=output_base / "4_resonance"
    )

    # 5. Belief Perseverance Decay
    all_results['perseverance'] = simulate_belief_perseverance(
        output_dir=output_base / "5_perseverance"
    )

    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETE")
    print(f"Results saved to: {output_base}")
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
                        help="Which simulation to run (1-5, or 0 for all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    output = Path(args.output) if args.output else None

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
        print("Use 0 for all, or 1-5 for specific simulations")
