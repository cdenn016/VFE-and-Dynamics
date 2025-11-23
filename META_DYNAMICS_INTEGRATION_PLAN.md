# Meta-Agent Ouroboros Tower with Dynamics Integration
## Full Multi-Scale Free Energy Minimization over 2-3D Base Manifolds

**Date:** November 23, 2025
**Authors:** Chris & Claude
**Goal:** Wire up complete hierarchical meta-agent emergence with Hamiltonian/gradient flow dynamics over dimension-agnostic spatial fields

---

## 🎯 Vision

Build the **full ouroboros tower** where:

1. **Constituent agents** (ζ=0) minimize free energy over 2-3D spatial fields
2. **Meta-agents** (ζ=1,2,3...) emerge from consensus clusters
3. **Dynamics propagate** top-down (priors) and bottom-up (observations)
4. **Hamiltonian/gradient flow** dynamics at ALL scales
5. **Pullback geometry** analyzed across hierarchical levels
6. **Dimension agnostic** - works for 1D, 2D, 3D, nD base manifolds

---

## 📊 Current State Assessment

### ✅ What Already Exists

#### **1. Meta-Agent Architecture** (`meta/` module)

**Files:**
- `meta/emergence.py` - Core hierarchical agent and multi-scale system
- `meta/hierarchical_evolution.py` - Evolution engine with cross-scale dynamics
- `meta/consensus.py` - Consensus detection and clustering
- `meta/pointwise_emergence.py` - Spatial consensus emergence

**Key Classes:**
```python
class HierarchicalAgent(Agent):
    """Agent with scale awareness and constituent tracking"""
    - scale: int (ζ ∈ {0,1,2,...})
    - constituent_indices: List[ScaleIndex]
    - parent_meta: Optional[HierarchicalAgent]
    - hyperprior_mus, hyperprior_Sigmas  # Ouroboros tower!

class MultiScaleSystem:
    """Container for agents at multiple scales"""
    - agents: Dict[int, List[HierarchicalAgent]]
    - condensations: List[MetaAgentDescriptor]
    - base_manifold: BaseManifold

class HierarchicalEvolutionEngine:
    """Main dynamics loop for hierarchical system"""
    - Top-down prior updates
    - Bottom-up observation generation
    - Timescale-separated updates
    - Automatic consensus detection
```

**Features:**
- ✅ Scale indexing (ζ=0,1,2,...)
- ✅ Constituent tracking
- ✅ Parent/child relationships
- ✅ Ouroboros hyperprior tower (non-Markovian)
- ✅ Timescale separation (τ_ζ = 10^ζ bits)
- ✅ Consensus detection
- ✅ Meta-agent formation
- ✅ Cross-scale coupling (top-down + bottom-up)

#### **2. Dynamics Framework**

**Files:**
- `agent/trainer.py` - Gradient flow trainer with snapshot storage
- `agent/hamiltonian_trainer.py` - Hamiltonian dynamics trainer
- `dynamics/trajectory_metrics.py` - Time-evolving pullback metrics

**Features:**
- ✅ Gradient flow (overdamped) dynamics
- ✅ Hamiltonian (underdamped) dynamics with leapfrog integration
- ✅ Snapshot storage during training
- ✅ Time-evolving metric computation
- ✅ Signature evolution analysis

#### **3. Geometry Framework**

**Files:**
- `geometry/pullback_metrics.py` - Pullback metric computation
- `geometry/signature_analysis.py` - Metric signature classification
- `geometry/gauge_consensus.py` - Gauge-aware consensus

**Features:**
- ✅ Belief-based pullback metrics G_q
- ✅ Prior-based pullback metrics G_p
- ✅ Signature classification (Riemannian, Lorentzian, indefinite)
- ✅ Dimension-agnostic implementation
- ✅ Works with arbitrary spatial_shape: Tuple[int, ...]

#### **4. Base Manifold Support**

**Current Support:**
- ✅ 1D grids: `spatial_shape=(N,)`
- ✅ 2D grids: `spatial_shape=(H,W)` - architecture supports it
- ✅ 3D grids: `spatial_shape=(H,W,D)` - architecture supports it
- ✅ Dimension-agnostic iteration via `np.ndindex(spatial_shape)`

---

## ❌ What's Missing (Integration Gaps)

### **Gap 1: Dynamics Not Integrated with MultiScaleSystem**

**Problem:**
- `Trainer` and `HamiltonianTrainer` work with `MultiAgentSystem` (single-scale)
- `MultiScaleSystem` has its own `HierarchicalEvolutionEngine`
- These two dynamics systems don't talk to each other!

**Impact:**
- Can't run Hamiltonian dynamics on hierarchical systems
- Can't analyze trajectory metrics across scales
- Meta-agents don't benefit from conservative dynamics

**Solution:**
Create `HierarchicalHamiltonianTrainer` that:
- Operates on `MultiScaleSystem`
- Applies Hamiltonian dynamics at each scale
- Respects cross-scale coupling
- Enables trajectory metric analysis across hierarchy

### **Gap 2: No Cross-Scale Trajectory Metrics**

**Problem:**
- `trajectory_metrics.py` analyzes single-scale agent trajectories
- No tools for analyzing metric evolution across hierarchical levels

**Impact:**
- Can't study how signature changes across scales (ζ=0 → ζ=1 → ζ=2)
- Can't detect scale-dependent Lorentzian structure
- Missing emergent geometric phenomena

**Solution:**
Extend trajectory metrics with:
```python
def compute_hierarchical_trajectory_metrics(
    history: HierarchicalHistory,
    scale: int,
    agent_idx: int,
    point_idx: int
) -> TrajectoryMetrics:
    """Compute metrics for agent at specific scale."""

def analyze_cross_scale_geometry(
    history: HierarchicalHistory,
    constituent_idx: ScaleIndex,
    meta_idx: ScaleIndex,
    point_idx: int
) -> CrossScaleGeometryAnalysis:
    """
    Compare geometry at different scales:
    - How does G_q change from ζ=0 → ζ=1?
    - Does signature change across scales?
    - Emergent Lorentzian structure at meta level?
    """
```

### **Gap 3: No 2-3D Examples or Tests**

**Problem:**
- All current tests use 1D grids
- No validation that 2-3D actually works end-to-end
- No example workflows for spatial fields

**Impact:**
- Uncertain if geometry calculations work in 2-3D
- No reference implementations for users
- Potential hidden bugs in higher dimensions

**Solution:**
Create comprehensive 2-3D test suite:
- `test_2d_meta_dynamics.py` - 2D spatial field with meta-agents
- `test_3d_meta_dynamics.py` - 3D spatial field (if computationally feasible)
- Example notebooks showing spatial emergence patterns

### **Gap 4: Snapshot Storage for Hierarchical Systems**

**Problem:**
- Snapshot storage implemented for single-scale `TrainingHistory`
- `MultiScaleSystem` has no snapshot mechanism
- Can't replay hierarchical evolution

**Impact:**
- Can't compute time-evolving metrics for meta-agents
- Can't visualize cross-scale dynamics over time
- Missing critical analysis capability

**Solution:**
Create `HierarchicalHistory` with:
- Snapshots at ALL scales
- Cross-scale coupling snapshots (priors, observations)
- Condensation event tracking
- Scale-indexed trajectory reconstruction

---

## 🗺️ Integration Roadmap

### **Phase 1: Foundation (2-3 days)**

**Tasks:**
1. ✅ Create this integration plan document
2. **Validate 2-3D support** - Create simple 2D/3D examples
3. **Audit geometry code** - Ensure all operations are truly dimension-agnostic
4. **Design HierarchicalHistory** - Data structure for multi-scale snapshots

**Deliverables:**
- `test_2d_field_agent.py` - Validate 2D geometry works
- `test_3d_field_agent.py` - Validate 3D (if feasible)
- `HierarchicalHistory` dataclass design

### **Phase 2: Core Integration (3-5 days)**

**Tasks:**
1. **Create HierarchicalHamiltonianTrainer**
   - Extends `HamiltonianTrainer` for multi-scale systems
   - Phase space (θ_ζ, p_ζ) for each scale
   - Cross-scale Hamiltonian with coupling terms

2. **Implement HierarchicalHistory**
   - Multi-scale snapshot storage
   - Condensation event tracking
   - Trajectory reconstruction at all scales

3. **Extend trajectory_metrics.py**
   - `compute_hierarchical_trajectory_metrics()`
   - `analyze_cross_scale_geometry()`
   - Scale-indexed eigenvalue evolution

**Deliverables:**
- `agent/hierarchical_hamiltonian_trainer.py` (500+ lines)
- `HierarchicalHistory` in `meta/history.py`
- Extended `dynamics/trajectory_metrics.py`

### **Phase 3: Cross-Scale Analysis (2-3 days)**

**Tasks:**
1. **Cross-scale pullback metrics**
   - G_q^(ζ) at each scale
   - Compare geometry: constituent vs meta-agent
   - Detect emergent Lorentzian structure at meta level

2. **Hierarchical visualization**
   - Signature evolution across scales
   - Multi-scale dashboards
   - Cross-scale coupling visualization

**Deliverables:**
- `analysis/plots/hierarchical_metrics.py`
- `analysis/plots/cross_scale_geometry.py`

### **Phase 4: Complete 2-3D Meta-Agent System (3-4 days)**

**Tasks:**
1. **2D meta-agent emergence example**
   - 2D spatial field (e.g., 16×16 grid)
   - N base agents (ζ=0)
   - Consensus → meta-agents (ζ=1)
   - Hamiltonian dynamics at both scales
   - Full trajectory metric analysis

2. **3D example (stretch goal)**
   - 3D spatial field (e.g., 8×8×8 grid)
   - Verify dimension-agnostic implementation
   - Performance profiling

3. **Comprehensive test suite**
   - `test_2d_meta_hamiltonian.py`
   - `test_cross_scale_metrics.py`
   - Integration tests

**Deliverables:**
- `examples/2d_meta_emergence_hamiltonian.py` (comprehensive example)
- `examples/3d_meta_emergence.py` (if feasible)
- Full test coverage

### **Phase 5: Research Applications (Ongoing)**

**Tasks:**
1. **Study emergent phenomena**
   - When does Lorentzian structure emerge at meta level?
   - How do signatures differ across scales?
   - Timescale separation effects on geometry

2. **Parameter space exploration**
   - Phase diagrams for meta-agent formation
   - Optimal damping for hierarchical systems
   - Critical phenomena in consensus emergence

**Deliverables:**
- Research notebooks
- Phase diagrams
- Publication-ready figures

---

## 🏗️ Technical Architecture

### **Hierarchical Hamiltonian System**

**Phase Space:**
```python
# At each scale ζ:
θ_ζ = pack_parameters(agents[ζ])  # All parameters at scale ζ
p_ζ = conjugate_momenta[ζ]        # Conjugate momenta

# Total phase space (all scales):
Θ = [θ_0, θ_1, θ_2, ...]  # Constituent → meta → meta-meta → ...
P = [p_0, p_1, p_2, ...]
```

**Hamiltonian:**
```python
H_total = Σ_ζ H_ζ(θ_ζ, p_ζ) + H_coupling(θ_0, θ_1, ..., p_0, p_1, ...)

where:
  H_ζ(θ_ζ, p_ζ) = K_ζ(p_ζ) + V_ζ(θ_ζ)
  K_ζ = (1/2) p_ζᵀ G_ζ^{-1} p_ζ       # Kinetic energy with Fisher metric
  V_ζ = F_ζ(θ_ζ)                       # Free energy potential

  H_coupling = cross-scale prior alignment + observations
```

**Equations of Motion:**
```python
# At each scale ζ:
dθ_ζ/dt = ∂H/∂p_ζ = G_ζ^{-1} p_ζ
dp_ζ/dt = -∂H/∂θ_ζ = -∇_θ F_ζ - ∂H_coupling/∂θ_ζ

# With damping (Nosé-Hoover):
dp_ζ/dt = -∇_θ F_ζ - γ_ζ p_ζ - ∂H_coupling/∂θ_ζ
```

**Cross-Scale Coupling:**
```python
# Top-down (meta → constituents):
∂H_coupling/∂θ_i^(ζ) includes λ_prior * KL(q_i || p_parent)

# Bottom-up (constituents → meta):
∂H_coupling/∂θ_M^(ζ+1) includes λ_obs_meta * Σ_i KL(q_i || q_M)
```

### **Hierarchical History Structure**

```python
@dataclass
class HierarchicalHistory:
    """Training history for multi-scale system."""

    # Per-scale histories
    scale_histories: Dict[int, ScaleHistory]

    # Condensation events
    condensation_events: List[CondensationEvent]

    # Snapshots at all scales
    snapshots: List[HierarchicalSnapshot]
    snapshot_steps: List[int]

    # Energy decomposition
    energy_by_scale: Dict[int, List[float]]
    coupling_energy: List[float]

    def get_agent_trajectory(self, scale_index: ScaleIndex) -> AgentTrajectory:
        """Extract trajectory for specific agent at specific scale."""

    def get_cross_scale_metrics(self,
                                 constituent: ScaleIndex,
                                 meta: ScaleIndex,
                                 point_idx: int) -> CrossScaleMetrics:
        """Compare geometry between constituent and meta-agent."""

@dataclass
class HierarchicalSnapshot:
    """Snapshot at specific training step across all scales."""
    step: int
    scale_snapshots: Dict[int, ScaleSnapshot]  # One per scale
    cross_scale_transports: Dict[Tuple[int,int], np.ndarray]  # Ω matrices

@dataclass
class ScaleSnapshot:
    """Snapshot of all agents at a specific scale."""
    scale: int
    agents: List[AgentStateSnapshot]
    active_agents: List[int]  # Which agents are active

@dataclass
class CondensationEvent:
    """Record of meta-agent formation."""
    step: int
    constituents: List[ScaleIndex]
    meta_agent: ScaleIndex
    coherence_before: float
    coherence_after: float
```

---

## 🎨 Example Workflow (2D Case)

```python
import numpy as np
from pathlib import Path

from agent.agents import Agent
from config import AgentConfig, SystemConfig
from meta.emergence import MultiScaleSystem, create_agents_at_scale
from agent.hierarchical_hamiltonian_trainer import HierarchicalHamiltonianTrainer
from dynamics.trajectory_metrics import compute_hierarchical_trajectory_metrics
from analysis.plots.hierarchical_metrics import plot_cross_scale_dashboard

# === 1. Create 2D spatial field ===
spatial_shape = (16, 16)  # 2D grid
K = 3  # Latent dimension
n_agents = 20

# Agent config
agent_config = AgentConfig(
    spatial_shape=spatial_shape,
    K=K,
    mu_scale=0.2,
    sigma_scale=0.1
)

# System config
system_config = SystemConfig(
    lambda_self=1.0,
    lambda_belief_align=1.0,
    lambda_obs_meta=1.0,  # Meta-agent observations
    lambda_prior_align=0.5  # Parent priors
)

# === 2. Create multi-scale system ===
from geometry.geometry_base import RegularGrid

base_manifold = RegularGrid(spatial_shape)
multi_scale_system = MultiScaleSystem(
    base_manifold=base_manifold,
    max_emergence_levels=3  # Allow up to ζ=2
)

# Create base agents (ζ=0)
base_agents = create_agents_at_scale(
    scale=0,
    n_agents=n_agents,
    agent_config=agent_config,
    base_manifold=base_manifold,
    rng=np.random.default_rng(42)
)

multi_scale_system.add_agents_at_scale(0, base_agents)

# === 3. Train with Hamiltonian dynamics ===
from config import HamiltonianConfig

ham_config = HamiltonianConfig(
    n_steps=500,
    dt=0.01,
    friction=0.1,  # Light damping
    save_snapshots=True,
    snapshot_every=10,

    # Consensus detection
    consensus_check_interval=50,
    consensus_threshold=0.01,

    # Cross-scale dynamics
    enable_top_down_priors=True,
    enable_bottom_up_obs=True,
    enable_hyperprior_tower=True  # Ouroboros!
)

trainer = HierarchicalHamiltonianTrainer(
    multi_scale_system,
    ham_config
)

history = trainer.train()

# === 4. Analyze results ===

# How many meta-agents emerged?
print(f"Final structure:")
for scale in multi_scale_system.agents:
    n_active = sum(1 for a in multi_scale_system.agents[scale] if a.is_active)
    print(f"  ζ={scale}: {n_active} active agents")

# Condensation events
print(f"\nCondensation events: {len(history.condensation_events)}")
for event in history.condensation_events:
    print(f"  Step {event.step}: {len(event.constituents)} → 1 meta-agent")

# === 5. Compute trajectory metrics ===

# Pick a meta-agent
meta_agent = multi_scale_system.agents[1][0]  # First meta-agent at ζ=1
point_idx = (8, 8)  # Center of 2D grid

# Compute metrics at meta level
meta_metrics = compute_hierarchical_trajectory_metrics(
    history,
    scale=1,
    agent_idx=0,
    point_idx=point_idx
)

# Analyze signature evolution
from dynamics.trajectory_metrics import analyze_signature_evolution
sig_analysis = analyze_signature_evolution(meta_metrics)

print(f"\nMeta-agent signature evolution:")
print(f"  Dominant: {sig_analysis['dominant_signature']}")
print(f"  Lorentzian: {sig_analysis['lorentzian_fraction']:.1%}")

# === 6. Cross-scale comparison ===

# Compare constituent vs meta geometry
from dynamics.trajectory_metrics import analyze_cross_scale_geometry

cross_scale = analyze_cross_scale_geometry(
    history,
    constituent_idx=meta_agent.constituent_indices[0],
    meta_idx=meta_agent.scale_index,
    point_idx=point_idx
)

print(f"\nCross-scale geometry:")
print(f"  Constituent signature: {cross_scale.constituent_signature}")
print(f"  Meta signature: {cross_scale.meta_signature}")
print(f"  Signature changed: {cross_scale.signature_changed}")

# === 7. Visualize ===
out_dir = Path("_results/2d_meta_emergence")

plot_cross_scale_dashboard(
    history,
    constituent_idx=meta_agent.constituent_indices[0],
    meta_idx=meta_agent.scale_index,
    point_idx=point_idx,
    out_dir=out_dir
)

print(f"\n✓ Visualization saved to {out_dir}")
```

**Expected Output:**
```
Final structure:
  ζ=0: 5 active agents  (15 condensed into meta-agents)
  ζ=1: 3 active agents  (3 meta-agents formed)

Condensation events: 3
  Step 100: 5 → 1 meta-agent
  Step 200: 6 → 1 meta-agent
  Step 350: 4 → 1 meta-agent

Meta-agent signature evolution:
  Dominant: riemannian
  Lorentzian: 0.0%

Cross-scale geometry:
  Constituent signature: riemannian
  Meta signature: riemannian
  Signature changed: False

✓ Visualization saved to _results/2d_meta_emergence
```

---

## 🔬 Research Questions

With this infrastructure, we can study:

### **1. Emergent Geometry Across Scales**
- Does metric signature change from ζ=0 → ζ=1 → ζ=2?
- Are meta-agents more/less likely to exhibit Lorentzian structure?
- How does consensus formation affect local geometry?

### **2. Dynamics Regime Effects**
- Gradient flow vs Hamiltonian at different scales
- Does conservative dynamics facilitate/inhibit emergence?
- Optimal damping schedules for hierarchical systems

### **3. Dimensional Scaling**
- How does emergence differ in 1D vs 2D vs 3D?
- Are there critical dimensions for consensus formation?
- Computational scaling with spatial dimension

### **4. Ouroboros Tower Phenomena**
- How many ancestral levels are beneficial?
- Non-Markovian memory effects on geometry
- Information flow in hyperprior tower

### **5. Timescale Separation**
- Emergent timescales: τ_ζ = 10^ζ natural or parameter-dependent?
- Effect on energy conservation
- Adiabatic vs sudden condensation

---

## 📦 Deliverables Summary

### **Code (New Files)**
1. `agent/hierarchical_hamiltonian_trainer.py` - Hamiltonian dynamics for multi-scale
2. `meta/history.py` - HierarchicalHistory data structures
3. `dynamics/hierarchical_metrics.py` - Cross-scale trajectory metrics
4. `analysis/plots/hierarchical_metrics.py` - Cross-scale visualization
5. `analysis/plots/cross_scale_geometry.py` - Multi-scale dashboards

### **Examples**
1. `examples/2d_meta_emergence_hamiltonian.py` - Complete 2D workflow
2. `examples/3d_meta_emergence.py` - 3D demonstration (stretch)

### **Tests**
1. `test_2d_field_agent.py` - Validate 2D geometry
2. `test_3d_field_agent.py` - Validate 3D geometry
3. `test_2d_meta_hamiltonian.py` - 2D meta-agent dynamics
4. `test_cross_scale_metrics.py` - Cross-scale analysis
5. `test_hierarchical_integration.py` - End-to-end integration

### **Documentation**
1. `META_DYNAMICS_INTEGRATION_PLAN.md` - This document
2. `HIERARCHICAL_HAMILTONIAN_GUIDE.md` - Usage guide
3. Updated `INTEGRATION_SUMMARY.md` with meta-agent section

---

## ⚡ Quick Start (Once Built)

```bash
# 1. Validate 2D support
python test_2d_field_agent.py

# 2. Run 2D meta-agent emergence with Hamiltonian dynamics
python examples/2d_meta_emergence_hamiltonian.py

# 3. Analyze cross-scale geometry
python -c "
from examples.analyze_2d_results import analyze_cross_scale_metrics
analyze_cross_scale_metrics('_results/2d_meta_emergence/history.pkl')
"

# 4. View visualizations
open _results/2d_meta_emergence/cross_scale_dashboard.png
```

---

## 🎯 Success Criteria

**Phase 1 (Foundation):** ✅
- [ ] 2D field agent test passes
- [ ] 3D field agent test passes (or documented limitations)
- [ ] HierarchicalHistory design complete

**Phase 2 (Core Integration):** ✅
- [ ] HierarchicalHamiltonianTrainer implemented
- [ ] Snapshot storage working for multi-scale systems
- [ ] Cross-scale trajectory metrics computable

**Phase 3 (Analysis):** ✅
- [ ] Can compare G_q^(ζ=0) vs G_q^(ζ=1)
- [ ] Signature evolution tracked across scales
- [ ] Visualizations show multi-scale dynamics

**Phase 4 (Complete System):** ✅
- [ ] 2D meta-agent emergence example runs end-to-end
- [ ] Energy conserved in Hamiltonian regime (≤5% drift)
- [ ] Consensus detection and meta-formation working
- [ ] Full trajectory metric analysis pipeline operational

**Phase 5 (Research):** 🔬
- [ ] At least 3 research questions investigated
- [ ] Publication-quality figures generated
- [ ] Novel geometric phenomena documented

---

## 🚀 Next Steps

**Immediate (Today):**
1. Get user confirmation on this plan
2. Start Phase 1: Create `test_2d_field_agent.py`
3. Validate dimension-agnostic geometry code

**This Week:**
1. Complete Phase 1 (2-3D validation)
2. Begin Phase 2 (HierarchicalHamiltonianTrainer)
3. Design HierarchicalHistory structure

**Next Week:**
1. Complete Phase 2 (core integration)
2. Begin Phase 3 (cross-scale analysis)

---

## 📚 References

**Existing Documentation:**
- `SUITE_EVALUATION.md` - Initial codebase assessment
- `DYNAMICS_PHASE_PLAN.md` - Original dynamics roadmap
- `INTEGRATION_SUMMARY.md` - Hamiltonian integration summary
- `HAMILTONIAN_INTEGRATION_README.md` - Hamiltonian dynamics guide

**Related Code:**
- `meta/` - Meta-agent framework
- `agent/hamiltonian_trainer.py` - Single-scale Hamiltonian dynamics
- `dynamics/trajectory_metrics.py` - Trajectory metric analysis
- `geometry/pullback_metrics.py` - Pullback metric computation

---

**Status:** 📋 Plan Complete - Awaiting Approval to Begin Implementation

**Total Estimated Effort:** 2-3 weeks for full integration (Phases 1-4)
