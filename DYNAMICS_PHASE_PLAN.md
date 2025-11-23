# Dynamics Phase Implementation Plan
## Pullback Geometries & Agent Belief Trajectories

**Date**: November 2025
**Phase**: Dynamics, Pullback Metrics, Belief Trajectories

---

## 🎯 Core Objectives

1. **Time-Evolving Pullback Metrics**: Track metric signature along belief trajectories
2. **Dynamics Integration**: Connect VFE gradient flow to Hamiltonian/Lagrangian formulations
3. **Lorentzian Structure Emergence**: Detect and analyze signature transitions
4. **Gauge Orbit Dynamics**: Implement parallel transport and gauge evolution

---

## 📋 Implementation Tasks

### **Task 1: Trajectory Snapshots & Metric Evolution**

**Goal**: Enable time-dependent pullback metric computation

**Changes Needed**:

1. **Extend `agent/trainer.py`**:
   ```python
   # Add snapshot storage during training
   class TrainingHistory:
       # ... existing fields ...
       agent_snapshots: List[Dict[str, np.ndarray]]  # NEW
       snapshot_steps: List[int]  # NEW

   # In training loop:
   if step % snapshot_every == 0:
       history.agent_snapshots.append({
           'mu_q': agent.mu_q.copy(),
           'Sigma_q': agent.Sigma_q.copy(),
           'mu_p': agent.mu_p.copy(),
           'Sigma_p': agent.Sigma_p.copy(),
           'phi': agent.phi.copy() if hasattr(agent, 'phi') else None
       })
   ```

2. **Create `dynamics/trajectory_metrics.py`**:
   ```python
   def compute_trajectory_pullback_metrics(
       history: TrainingHistory,
       agent_idx: int,
       point_idx: int,
       dx: float = 1.0
   ) -> List[InducedMetric]:
       """
       Compute pullback metric at each snapshot along trajectory.

       Returns:
           List of InducedMetric objects, one per snapshot
       """
       metrics = []
       for snapshot in history.agent_snapshots:
           # Reconstruct metric from snapshot
           mu = snapshot['mu_q']
           Sigma = snapshot['Sigma_q']

           if mu.ndim > 1:  # Field agent
               metric = pullback_metric_gaussian(mu, Sigma, dx=dx)
               metrics.append(metric)

       return metrics

   def analyze_signature_evolution(
       metrics: List[InducedMetric],
       steps: List[int]
   ) -> Dict[str, Any]:
       """
       Analyze how metric signature changes over time.

       Returns dict with:
           - signature_sequence: List of MetricSignature enums
           - transition_points: Steps where signature changes
           - eigenvalue_trajectories: (T, d) array
           - lorentzian_fraction: Fraction of time in Lorentzian regime
       """
       ...
   ```

3. **Update `analysis/plots/belief_trajectories.py`**:
   ```python
   def plot_trajectory_metric_signature(
       history,
       system,
       out_dir,
       point_idx=0,
       agent_idx=0
   ):
       # REPLACE current static computation with:
       from dynamics.trajectory_metrics import (
           compute_trajectory_pullback_metrics,
           analyze_signature_evolution
       )

       metrics = compute_trajectory_pullback_metrics(
           history, agent_idx, point_idx
       )

       sig_analysis = analyze_signature_evolution(
           metrics, history.snapshot_steps
       )

       # Plot ACTUAL time-varying eigenvalues
       # Highlight signature transitions
       # Show correlation with trajectory speed ||dμ/dt||
   ```

**Validation**:
- Run on simple 1D→3D agent with known dynamics
- Verify signature stays Riemannian for simple quadratic potential
- Test on chaotic dynamics to observe transitions

---

### **Task 2: Hamiltonian vs Gradient Flow Experiments**

**Goal**: Compare overdamped (current) vs underdamped (Hamiltonian) dynamics

**Implementation**:

1. **Create `dynamics/experiments/regime_comparison.py`**:
   ```python
   def compare_dynamics_regimes(
       agent_config: AgentConfig,
       potential_type: str = "quadratic",
       n_steps: int = 500,
       regimes: List[str] = ["overdamped", "underdamped", "critical"]
   ):
       """
       Run same agent with different dynamical regimes:

       - Overdamped: θ̇ = -∇V (current VFE)
       - Underdamped: Full Hamiltonian with momentum
       - Critical: Optimally damped

       Compare:
           - Convergence speed
           - Energy conservation (Hamiltonian should conserve)
           - Trajectory length
           - Final states
       """

       results = {}

       for regime in regimes:
           if regime == "overdamped":
               # Use existing trainer
               history = run_gradient_flow(agent_config, n_steps)

           elif regime == "underdamped":
               # Use Hamiltonian integrator
               from dynamics.hamiltonian import BeliefHamiltonian
               hamiltonian = BeliefHamiltonian(
                   potential=lambda q: free_energy(q),
                   metric=lambda q: fisher_metric(q)
               )
               history = run_hamiltonian_evolution(
                   hamiltonian, agent_config, n_steps
               )

           results[regime] = analyze_trajectory(history)

       # Plot comparative dashboard
       plot_regime_comparison(results)
   ```

2. **Metrics to Track**:
   - Energy conservation: |H(t) - H(0)| / H(0)
   - Convergence: ||θ(t) - θ*||
   - Symplectic structure preservation
   - Pullback metric evolution

**Expected Outcomes**:
- Overdamped: Monotonic energy decrease, no oscillations
- Underdamped: Energy conserved, orbital motion around minimum
- Critical: Fastest convergence without overshoot

---

### **Task 3: Lorentzian Structure Search**

**Goal**: Systematically find regions where pullback metric becomes Lorentzian

**Approach**:

1. **Create `geometry/lorentzian_detector.py`**:
   ```python
   def scan_parameter_space_for_lorentzian(
       agent: Agent,
       grid_density: int = 20,
       parameters: List[str] = ["lambda_belief_align", "lambda_prior_align"]
   ) -> LorentzianMap:
       """
       Sweep parameter space to find Lorentzian regions.

       For each parameter combination:
           1. Compute pullback metric
           2. Check signature
           3. If Lorentzian, analyze timelike/spacelike structure

       Returns:
           LorentzianMap with signature classified at each grid point
       """

       # Create parameter grid
       param_grid = create_parameter_grid(parameters, grid_density)

       signature_map = np.zeros(param_grid.shape[:len(parameters)], dtype=int)

       for idx, param_values in enumerate(param_grid):
           # Set parameters
           config = update_config(agent.config, parameters, param_values)

           # Recompute agent with new config
           agent_temp = Agent(agent.agent_id, config)

           # Compute pullback metric
           G_belief, G_prior = agent_induced_metrics(agent_temp)

           # Analyze signature
           sig = analyze_metric_signature(G_belief.G[0])  # At first point

           signature_map[idx] = sig.signature_tuple[0]  # n_negative

       return LorentzianMap(
           param_grid=param_grid,
           signature_map=signature_map,
           parameters=parameters
       )

   def plot_lorentzian_phase_diagram(lorentzian_map):
       """
       Visualize which parameter combinations yield Lorentzian metrics.

       Create 2D heatmap showing:
           - Blue: Riemannian (0 negative eigenvalues)
           - Red: Lorentzian (1 negative eigenvalue)
           - Purple: Indefinite (>1 negative)
       """
       ...
   ```

2. **Hypothesis Testing**:
   - Does belief-prior coupling strength affect signature?
   - Do gauge fields introduce Lorentzian structure?
   - Are Lorentzian regions correlated with phase transitions?

---

### **Task 4: Tangent Vector Classification**

**Goal**: Determine if trajectory tangent dθ/dt is timelike or spacelike

**Theory**:
For a metric with signature (-,+,+,...,+), a tangent vector v is:
- **Timelike**: g(v, v) < 0
- **Spacelike**: g(v, v) > 0
- **Null**: g(v, v) = 0

**Implementation**:

```python
def classify_trajectory_tangent(
    metric: InducedMetric,
    tangent: np.ndarray,
    point_idx: int = 0
) -> str:
    """
    Classify tangent vector as timelike, spacelike, or null.

    Args:
        metric: Pullback metric at this time
        tangent: dθ/dt trajectory tangent vector
        point_idx: Which spatial point

    Returns:
        "timelike" | "spacelike" | "null"
    """
    G = metric.G[point_idx]  # Extract metric at this point

    # Compute g(v, v) = v^T G v
    norm_squared = tangent @ G @ tangent

    if abs(norm_squared) < 1e-10:
        return "null"
    elif norm_squared < 0:
        return "timelike"
    else:
        return "spacelike"

def analyze_trajectory_causality(
    history: TrainingHistory,
    metrics: List[InducedMetric],
    tangents: np.ndarray
):
    """
    Track causal structure along trajectory.

    At each step:
        - Compute g(dθ/dt, dθ/dt)
        - Classify as timelike/spacelike/null
        - Check for causality violations

    Returns:
        CausalityAnalysis with:
            - tangent_norms: g(v,v) at each step
            - causal_type: Classification at each step
            - light_cone_violations: Any null→timelike→null transitions
    """
    ...
```

**Key Question**:
Does the "natural time direction" from belief evolution align with a timelike direction in an emergent Lorentzian metric?

---

### **Task 5: Gauge Orbit Parallel Transport**

**Goal**: Implement parallel transport of beliefs along gauge orbits

**Theory**:
Given SO(3) gauge symmetry, each belief μ has an orbit:
```
O_μ = {R · μ : R ∈ SO(3)}
```

Parallel transport moves along this orbit while preserving geometric structure.

**Implementation**:

```python
# Create dynamics/gauge_transport.py

def parallel_transport_along_orbit(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rotation_path: List[np.ndarray],  # List of SO(3) matrices
    connection: Optional[np.ndarray] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parallel transport (μ, Σ) along gauge orbit.

    Args:
        mu: Initial mean (K,)
        Sigma: Initial covariance (K, K)
        rotation_path: Sequence of SO(3) rotations
        connection: Gauge connection (if None, use flat connection)

    Returns:
        List of (μ_t, Σ_t) at each step along path
    """

    trajectory = [(mu.copy(), Sigma.copy())]

    for R in rotation_path:
        # Transform under gauge rotation
        mu_next = R @ mu
        Sigma_next = R @ Sigma @ R.T

        # Apply connection (if non-flat)
        if connection is not None:
            # Parallel transport correction
            mu_next = apply_connection_correction(mu_next, connection)

        trajectory.append((mu_next, Sigma_next))

    return trajectory

def compute_holonomy(
    trajectory: List[Tuple[np.ndarray, np.ndarray]],
    metric_func: Callable
):
    """
    Compute holonomy: parallel transport around closed loop.

    For a closed path γ, holonomy measures how much the
    state "rotates" after completing the loop.

    Non-zero holonomy → curvature in gauge bundle
    """
    ...
```

---

### **Task 6: Experimental Suite**

**Create `dynamics/experiments/comprehensive_dynamics_suite.py`**:

```python
def run_comprehensive_dynamics_experiments(
    output_dir: Path,
    configs: List[AgentConfig]
):
    """
    Complete experimental suite covering:

    1. Trajectory-Metric Evolution
    2. Regime Comparison (overdamped vs underdamped)
    3. Lorentzian Structure Search
    4. Causality Analysis
    5. Gauge Orbit Dynamics
    6. Holonomy Computation

    Generates full dashboard of results.
    """

    results = {
        'trajectory_metrics': [],
        'regime_comparison': [],
        'lorentzian_maps': [],
        'causality_analysis': [],
        'gauge_transport': [],
        'holonomy': []
    }

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Experiment: {config}")
        print(f"{'='*60}")

        # 1. Run training with snapshots
        history = train_agent_with_snapshots(config)

        # 2. Compute trajectory metrics
        metrics = compute_trajectory_pullback_metrics(history)
        results['trajectory_metrics'].append(metrics)

        # 3. Compare dynamics regimes
        regime_data = compare_dynamics_regimes(config)
        results['regime_comparison'].append(regime_data)

        # 4. Search for Lorentzian regions
        lorentz_map = scan_parameter_space_for_lorentzian(agent)
        results['lorentzian_maps'].append(lorentz_map)

        # 5. Analyze causality
        causality = analyze_trajectory_causality(history, metrics)
        results['causality_analysis'].append(causality)

        # 6. Gauge transport
        if hasattr(agent, 'phi'):
            transport = compute_gauge_orbit_transport(agent)
            results['gauge_transport'].append(transport)

    # Generate comprehensive report
    generate_dynamics_report(results, output_dir)
```

---

## 🔧 Technical Implementation Checklist

### Phase 1: Infrastructure (Week 1)
- [ ] Extend `TrainingHistory` with snapshot storage
- [ ] Create `dynamics/trajectory_metrics.py`
- [ ] Update `analysis/plots/belief_trajectories.py` for time-evolving metrics
- [ ] Add tests for snapshot storage and metric reconstruction

### Phase 2: Dynamics Comparison (Week 2)
- [ ] Implement `dynamics/experiments/regime_comparison.py`
- [ ] Connect Hamiltonian integrator to agent system
- [ ] Create comparative visualization dashboards
- [ ] Validate energy conservation in Hamiltonian regime

### Phase 3: Lorentzian Analysis (Week 3)
- [ ] Create `geometry/lorentzian_detector.py`
- [ ] Implement parameter space scanning
- [ ] Add tangent vector classification (timelike/spacelike)
- [ ] Build phase diagram visualizations

### Phase 4: Gauge Dynamics (Week 4)
- [ ] Implement `dynamics/gauge_transport.py`
- [ ] Add parallel transport along orbits
- [ ] Compute holonomy for closed paths
- [ ] Visualize gauge orbit evolution

### Phase 5: Integration & Experiments (Week 5)
- [ ] Create comprehensive experiment suite
- [ ] Run experiments on multiple configurations
- [ ] Generate final analysis dashboards
- [ ] Document findings

---

## 📊 Expected Outputs

1. **Time-Evolving Signature Plots**:
   - Eigenvalue trajectories λ_i(t)
   - Signature transitions (Riemannian ↔ Lorentzian)
   - Correlation with training dynamics

2. **Regime Comparison Dashboard**:
   - Energy evolution: E(t) for each regime
   - Convergence rates
   - Phase portraits

3. **Lorentzian Phase Diagrams**:
   - 2D heatmaps of signature in parameter space
   - Regions of emergent spacetime structure
   - Critical boundaries

4. **Causality Analysis**:
   - Tangent vector classifications over time
   - Light cone structure (if Lorentzian)
   - Causal consistency checks

5. **Gauge Orbit Visualizations**:
   - 3D plots of orbits in belief space
   - Parallel transport trajectories
   - Holonomy around closed loops

---

## 🎓 Research Questions to Address

1. **Does variational inference naturally induce Lorentzian structure?**
   - When/where does signature flip?
   - What physical interpretation?

2. **How does the "inference arrow of time" relate to geometric time direction?**
   - Is dμ/dt timelike in Lorentzian regions?
   - Does this define emergent causality?

3. **What role do gauge fields play in signature transitions?**
   - Do φ fields break/restore symmetry?
   - Connection to spontaneous symmetry breaking?

4. **Can we detect phase transitions via signature changes?**
   - Order parameters
   - Critical exponents

5. **Does holonomy have information-theoretic meaning?**
   - Berry phase in belief space?
   - Geometric phase in gauge orbits?

---

## 📚 References to Integrate

- Fisher-Rao metric geometry (Amari)
- Lorentzian geometry in statistical manifolds (recent papers)
- Gauge theory and holonomy (physics)
- Hamiltonian Monte Carlo (Neal, 2011)
- Riemannian manifold optimization (Absil et al.)

---

## ⚠️ Potential Pitfalls

1. **Numerical Stability**:
   - Metric computation can be ill-conditioned
   - Eigenvalue decomposition may fail
   - Add robust regularization

2. **Storage Overhead**:
   - Snapshots can be memory-intensive
   - Consider sparse sampling
   - Use memory-mapped arrays for large runs

3. **Interpretation Challenges**:
   - Lorentzian signature doesn't automatically mean "spacetime"
   - Be careful with physical claims
   - Focus on mathematical structure first

4. **Gauge Ambiguity**:
   - SO(3) orbits are 3D, hard to visualize
   - Need good projection methods
   - Gauge fixing may be necessary

---

## 🚀 Next Immediate Actions

1. **TODAY**: Implement snapshot storage in `agent/trainer.py`
2. **THIS WEEK**: Create `dynamics/trajectory_metrics.py` with time-evolving metric computation
3. **NEXT WEEK**: Build regime comparison experiments
4. **ONGOING**: Document findings and update theory

---

**Status**: Ready to begin implementation
**Priority**: HIGH - Critical for next phase of research
**Complexity**: MEDIUM-HIGH - Requires careful integration across modules
