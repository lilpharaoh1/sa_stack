# MCTS Trajectory Generator — Implementation Spec
### Adapted from TreeIRL for Belief-Driven Assistive Driving with Optimisation Warm-Starting

---

## 0. Purpose of This Document

This spec describes how to implement the MCTS trajectory generation component from
**TreeIRL** (Tomov et al., 2025) and adapt it for an assistive driving pipeline where:

1. A **belief-driven human driver model** provides a distribution over driver intent/behaviour.
2. **MCTS generates k candidate trajectories** that are contextually appropriate for the
   current scene and belief state.
3. The **top-k candidates are passed to an optimisation-based motion planner** as warm starts,
   rather than being scored by an IRL network.

The implementation is broken into self-contained modules. At each section marked
**⚠️ CLARIFICATION REQUIRED**, Claude Code **must pause and ask the user the listed
questions before writing code for that section**. Do not assume answers; the user's
system has non-trivial design choices that affect correctness.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scene Context (c)                        │
│  ego kinematics · agent states · agent predictions · map/route  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Belief Model Interface    │
              │  b(t) = P(intent | history) │
              └─────────────┬──────────────┘
                            │  belief state b
              ┌─────────────▼──────────────┐
              │    MCTS Trajectory          │
              │      Generator              │
              │  n iterations → k trajs     │
              └─────────────┬──────────────┘
                            │  {τ₁ … τ_k}
              ┌─────────────▼──────────────┐
              │   Candidate Ranking /       │
              │   Filtering  (optional)     │
              └─────────────┬──────────────┘
                            │  {τ̃₁ … τ̃_m}, m ≤ k
              ┌─────────────▼──────────────┐
              │  Optimisation-Based Planner │
              │  (warm-started from τ̃_i)   │
              └─────────────┬──────────────┘
                            │  τ*
              ┌─────────────▼──────────────┐
              │  Control / Assistance Layer │
              └─────────────────────────────┘
```

---

## 2. Repository Layout

Create the following directory structure:

```
mcts_planner/
├── mdp/
│   ├── __init__.py
│   ├── state.py          # State dataclass
│   ├── action.py         # Action space definition
│   ├── transition.py     # Kinematic transition function
│   ├── reward.py         # Reward function (configurable weights)
│   └── termination.py    # Termination / horizon check
├── mcts/
│   ├── __init__.py
│   ├── node.py           # Tree node (Q, N, children)
│   ├── search.py         # Core MCTS loop (Algorithm 1 from paper)
│   ├── policies.py       # Rollout / padding / prior policies (IDM, CS, RL)
│   └── trajectory.py     # DFS extraction of top-k leaf trajectories
├── belief/
│   ├── __init__.py
│   └── interface.py      # Abstract BeliefModel + adapter to user's model
├── warm_start/
│   ├── __init__.py
│   ├── converter.py      # 1-D longitudinal → 2-D Cartesian waypoints
│   └── interface.py      # Abstract WarmStartReceiver for the optimiser
├── planner.py            # Top-level MCTSPlanner class (entry point)
├── config.py             # All hyperparameters as a dataclass / YAML schema
└── tests/
    ├── test_mdp.py
    ├── test_mcts.py
    └── test_warm_start.py
```

---

## 3. MDP Formulation

### 3.1 State Space

Implement a `State` dataclass with the following fields, mirroring the paper (§IV-A):

```python
@dataclass
class State:
    # Ego longitudinal (projected onto reference path)
    x_ego:    float   # position (m)
    v_ego:    float   # velocity (m/s)
    a_ego:    float   # acceleration (m/s²)

    # Lead agent longitudinal (may be None if no lead agent)
    x_lead:   Optional[float]
    v_lead:   Optional[float]
    a_lead:   Optional[float]

    # Static / contextual
    t:        float   # time offset in planning horizon (s)
    x_max:    float   # max longitudinal offset (goal or red light)
    v_max:    float   # speed limit (m/s)

    # ⚠️  Belief extension — see §3.1 clarification below
    belief:   Optional[Any] = None
```

⚠️ **CLARIFICATION REQUIRED — §3.1 Belief State Integration**

Before implementing the State class and the transition/reward functions, ask the user:

> **Q1.** How is the belief state `b(t)` represented in your system?  
>   (a) A discrete probability vector over a fixed set of intents (e.g., "slow", "normal", "aggressive")?  
>   (b) A continuous distribution (e.g., Gaussian over a latent intent variable)?  
>   (c) A scalar summary statistic (e.g., estimated aggressiveness 0–1)?  
>   (d) Something else — please describe the data structure and its dimensionality.

> **Q2.** Should the belief state be part of the MDP state `s` seen by MCTS (affecting
>   the reward and/or transition function), or should it only influence the initial
>   conditions and reward weights set *before* the MCTS search begins?

> **Q3.** Does the belief state update *within* a single MCTS rollout (i.e., does it
>   evolve as actions are taken during simulation), or is it treated as fixed for the
>   entire planning call?

> **Q4.** Is the "human driver" the ego vehicle being assisted, or is it one of the
>   other agents whose behaviour you are modelling with a belief? Clarify the
>   perspective: who does the planner output a trajectory *for*?

---

### 3.2 Action Space

```python
# Longitudinal jerk commands (m/s³), discretised into 5 levels
JERK_ACTIONS = [-2.0, -1.0, 0.0, 1.0, 2.0]  # m/s³
```

⚠️ **CLARIFICATION REQUIRED — §3.2 Action Space**

> **Q5.** The paper restricts MCTS to 1-D longitudinal control only (lane following / ACC).
>   Does your assistive system also need lateral control (lane changes, steering)?  
>   If yes, how should the action space be extended?  
>   Note: adding lateral actions increases the branching factor exponentially and
>   will require significantly more MCTS iterations or a guided prior policy.

> **Q6.** Are there assistive-context constraints on the action space — e.g., the
>   planner should never command harder braking than the human currently is, or
>   should stay within a comfort envelope defined by the human's current inputs?

---

### 3.3 Transition Function

Implement a deterministic kinematic integrator (∆t = 0.5 s default, configurable):

```
x_ego'  = max(x_ego,  x_ego + v_ego·∆t + ½·a_ego·∆t² + ⅙·j'·∆t³)
v_ego'  = max(0,      v_ego + a_ego·∆t + ½·j'·∆t²)
a_ego'  = clip(a_ego + j·∆t,  [a_min, a_max])   # paper: [-7, 2] m/s²
j'      = (a_ego' - a_ego) / ∆t                  # effective jerk after clip
t'      = t + ∆t
x_max'  = x_max   (static within a planning call)
v_max'  = v_max   (static within a planning call)
```

Lead agent propagation: look up predicted trajectory from PBP/upstream predictor
at time `t'`; select closest agent on reference path within 2 m lateral threshold.

⚠️ **CLARIFICATION REQUIRED — §3.3 World Model**

> **Q7.** What prediction model are you using for other agents?  
>   (a) A replay of logged trajectories (as in nuPlan)?  
>   (b) A learned predictor (e.g., PBP, MTR, or similar)?  
>   (c) Constant-velocity / constant-acceleration extrapolation?  
>   The interface signature for `get_lead_agent_state(x_ego, predictions, t)` will
>   differ depending on the prediction format.

> **Q8.** Should the world model inside MCTS be **reactive** (other agents respond to
>   the ego's actions) or **non-reactive** (other agents follow their predicted
>   trajectories regardless of ego)? The paper uses non-reactive for latency reasons.
>   Is that acceptable for your assistive setting, or does the belief-driven human
>   need to react to the planned trajectory within the search?

---

### 3.4 Reward Function

Implement modular reward with configurable weights (paper §IV-A, Equations 2–11):

| Term | Description | Default weight |
|------|-------------|----------------|
| `r_jerk` | −jerk² | 0.05 |
| `r_accel` | −accel² | 0.20 |
| `r_speed` | penalise deviation from speed limit | 0.10 |
| `r_collision` | heavy penalty if ego ≥ lead or ≥ x_max with velocity | 10.0 |
| `r_clearance` | penalise closing within clearance buffer δ | 10.0 |
| `r_stop` | reward appropriate stop positioning | 0.10 |

Expose all weights as fields in `config.py` so they can be overridden.

⚠️ **CLARIFICATION REQUIRED — §3.4 Reward Function Adaptations**

> **Q9.** In an **assistive** (not fully autonomous) setting, the reward function may
>   need to change. For example:
>   - Should the reward penalise deviation from the *human's current trajectory* as
>     a baseline (to avoid unnecessary intervention)?
>   - Should there be a term rewarding "minimal intervention" — i.e., staying close
>     to what the human would do naturally?
>   - Should the belief state modulate reward weights (e.g., lower collision penalty
>     weight when belief says the human is a highly cautious driver)?
>   Please describe any such adaptations.

> **Q10.** What is the clearance buffer δ you want during MCTS planning vs. evaluation?
>   The paper uses δ = 1 m during training and δ = 2 m during evaluation.
>   In an assistive context the appropriate value may differ.

---

## 4. MCTS Implementation

### 4.1 Tree Node

```python
@dataclass
class Node:
    state:    State
    parent:   Optional['Node']
    action:   Optional[float]    # jerk that led to this node
    Q:        float = 0.0        # mean action-value
    N:        int   = 0          # visit count
    children: Dict[float, 'Node'] = field(default_factory=dict)
```

### 4.2 UCB / PUCTS Selection

```python
def ucb(node: Node, action: float, prior: float,
        c_puct: float = 1.0, q_max: float = 1.0, eps: float = 1e-6) -> float:
    sum_n = sum(child.N for child in node.children.values()) + 1
    exploration = c_puct * prior * (sum_n ** 0.5) / (node.children[action].N + 1 + eps)
    exploitation = node.children[action].Q / q_max
    return exploitation + exploration + random.uniform(0, 0.001)  # tie-breaking ε
```

### 4.3 Core Search Loop (Algorithm 1)

Implement `search(node, mdp, rollout_policy, config) -> float` recursively:

1. **Termination check**: if `F(s)` return 0.
2. **Selection**: choose `a = argmax_a UCB(s, a)` over all actions.
3. **Expansion**: if `N(s, a) == 0`, evaluate the new state.
4. **Evaluation**: use rollout policy (IDM default) to compute Monte Carlo return,
   or bootstrap with value network if available.
5. **Backup**: update `Q(s,a)` as running mean, increment `N(s,a)`.
6. Return `q = r + γ·v`.

Run for `n` iterations (configurable, default 400, paper §V-A).

### 4.4 Trajectory Extraction (DFS top-k)

After `n` iterations:

1. Run DFS from root, visiting children in **decreasing order of N**.
2. Collect the first `k` leaf nodes visited (default k = 100).
3. For each leaf, roll out with `padding_policy` (IDM default) until horizon H.
4. Return list of `k` complete `Trajectory` objects.

```python
@dataclass
class Trajectory:
    states:   List[State]        # length = H / ∆t + 1
    actions:  List[float]        # jerk commands
    returns:  float              # cumulative discounted reward
    cartesian_waypoints: Optional[np.ndarray] = None  # shape (T, 2), filled by post-processor
```

### 4.5 Rollout / Padding Policies

Implement the following interchangeable policies behind a common interface:

```python
class RolloutPolicy(Protocol):
    def act(self, state: State) -> float: ...  # returns jerk (or accel for IDM/CS)
```

| Policy | Description |
|--------|-------------|
| `IDMPolicy` | Intelligent Driver Model — compute safe following accel, convert to jerk |
| `ConstantSpeedPolicy` | Always return jerk = 0 |
| `UniformRandomPolicy` | Sample uniformly from JERK_ACTIONS |
| `NeuralPolicy` | Optional: wrap trained RL network `f_θ` |

IDM parameters (configurable): `v0` (desired speed), `T` (time headway = 1.5 s),
`a_max` (2.0 m/s²), `b` (comfortable decel = 3.0 m/s²), `s0` (min gap = 2.0 m).

---

## 5. Belief Model Interface

```python
class BeliefModel(ABC):
    @abstractmethod
    def get_belief(self, scene_context: Any) -> Any:
        """Return current belief state b(t)."""
        ...

    @abstractmethod
    def update(self, observation: Any) -> None:
        """Update belief given new observation."""
        ...
```

Provide a `NullBeliefModel` that always returns `None` so the MCTS can run
without a belief model for testing.

⚠️ **CLARIFICATION REQUIRED — §5 Belief Integration**

> **Q11.** What class / module implements your belief model? Please provide:
>   (a) The import path or class name.
>   (b) The method signatures for querying the current belief and for updating it.
>   (c) Whether it requires any special initialisation (e.g., from a config file,
>       pre-loaded weights, a specific observation history format).

> **Q12.** At what point in the planning loop should the belief be queried?
>   (a) Once per planning call, before MCTS starts (belief is fixed throughout search)?
>   (b) At every MCTS node expansion (belief updates as the simulated trajectory unfolds)?
>   Option (b) is richer but potentially much slower — is that acceptable?

---

## 6. Warm-Start Interface to the Optimisation Planner

This is the key adaptation from the original TreeIRL paper, which uses IRL scoring.
Instead of scoring trajectories, we pass the top-k candidates directly to the
downstream optimiser as warm starts.

### 6.1 Post-processing: 1-D → 2-D

```python
def longitudinal_to_cartesian(
    traj: Trajectory,
    reference_path: np.ndarray,   # shape (M, 2), Cartesian path points
) -> np.ndarray:                  # shape (T, 2), 2-D waypoints
    """
    Map each longitudinal offset x_ego[t] to the corresponding
    point along the reference path via arc-length interpolation.
    """
```

### 6.2 Candidate Selection Before Warm-Starting

Before passing to the optimiser, optionally rank or filter candidates:

- **By MCTS return**: sort descending by `trajectory.returns` (default).
- **By safety filter**: remove trajectories with any collision term triggered.
- **By belief alignment**: optionally score against the belief model.
- Pass the top `m` candidates to the optimiser (configurable, default m = 5).

⚠️ **CLARIFICATION REQUIRED — §6 Optimisation Planner Interface**

> **Q13.** What optimisation-based planner are you warm-starting?
>   (a) An MPC (e.g., CasADi, FORCES Pro, custom)?
>   (b) An iLQR / DDP solver?
>   (c) A trajectory optimisation library (e.g., GPOPS, TRAJOPT)?
>   (d) Something else?
>   The warm-start format (initial state sequence, control sequence, or both) depends
>   on the solver's API.

> **Q14.** What does the optimiser expect as a warm start?
>   (a) An initial **state trajectory** `x(t)` (positions, velocities)?
>   (b) An initial **control sequence** `u(t)` (jerk, acceleration, or steering)?
>   (c) Both?
>   (d) A single initial guess trajectory as a flat vector?

> **Q15.** Does the optimiser solve in 1-D longitudinal space (same as MCTS) or in
>   full 2-D / 3-D Cartesian space? This determines whether the 1-D → 2-D conversion
>   in §6.1 is needed before or after passing to the optimiser.

> **Q16.** How many warm-start candidates does the optimiser accept?
>   (a) One (best guess only)?
>   (b) Multiple (run optimiser from each, take best result)?
>   (c) A bundle that the optimiser solves jointly?

> **Q17.** Is there a hard latency budget for the entire pipeline (MCTS + optimisation)?
>   This will determine how many MCTS iterations `n` and candidates `k` are feasible.
>   The paper achieves ~10 ms for MCTS alone with `n=400, k=100` using IDM rollouts
>   on a single CPU thread.

---

## 7. Top-Level Planner API

```python
class MCTSPlanner:
    def __init__(self, config: PlannerConfig, belief_model: BeliefModel):
        ...

    def plan(self, scene_context: SceneContext) -> PlannerOutput:
        """
        Main entry point. Called at each planning cycle.

        Returns:
            PlannerOutput:
                .warm_starts    : List[Trajectory]   # top-m candidates for optimiser
                .best_trajectory: Trajectory          # highest-return candidate (fallback)
                .belief         : Any                 # belief state used this cycle
                .mcts_tree      : Node                # root of search tree (for debugging)
        """
        b = self.belief_model.get_belief(scene_context)
        s0 = build_initial_state(scene_context, b)
        root = run_mcts(s0, self.config)
        candidates = extract_top_k(root, self.config.k, self.config.padding_policy)
        candidates = postprocess(candidates, scene_context.reference_path)
        warm_starts = rank_and_filter(candidates, self.config.m)
        return PlannerOutput(warm_starts=warm_starts, ...)
```

---

## 8. Configuration Schema

All hyperparameters in a single `PlannerConfig` dataclass / YAML file:

```yaml
mcts:
  n_iterations: 400          # MCTS budget
  k_trajectories: 100        # leaves to extract
  m_warm_starts: 5           # candidates passed to optimiser
  gamma: 0.99                # discount factor
  c_puct: 1.0                # UCB exploration constant
  prior_policy: "uniform"    # "uniform" | "idm" | "neural"
  rollout_policy: "idm"      # "idm" | "constant_speed" | "neural"
  padding_policy: "idm"      # "idm" | "constant_speed" | "neural"

mdp:
  dt: 0.5                    # time step (s)
  horizon: 8.0               # planning horizon (s)  → max depth 16
  a_min: -7.0                # m/s²
  a_max:  2.0                # m/s²
  clearance_buffer_plan: 2.0 # δ during planning (m)

reward_weights:
  jerk:      0.05
  accel:     0.20
  speed:     0.10
  collision: 10.0
  clearance: 10.0
  stop:      0.10

idm:
  v0:   15.0   # desired speed (m/s)
  T:     1.5   # time headway (s)
  a:     2.0   # max accel (m/s²)
  b:     3.0   # comfortable decel (m/s²)
  s0:    2.0   # min gap (m)
  delta: 4.0   # accel exponent
```

---

## 9. Testing Requirements

Implement the following tests before considering any module complete:

### `test_mdp.py`
- Transition from rest with jerk=+2 reaches expected velocity after 4 steps.
- Transition clips at `a_max`/`a_min` correctly.
- Termination fires at `t = H`.
- Reward returns large negative value when `x_ego >= x_lead` with non-zero velocity.

### `test_mcts.py`
- With `n=1` iteration and uniform prior, root has exactly one child with N=1.
- With `n=400`, `extract_top_k(k=100)` returns exactly 100 trajectories.
- All returned trajectories have length `H/dt` steps.
- In a simple free-road scenario, MCTS with IDM rollout should produce at least one
  trajectory that reaches `v_max` within the horizon.

### `test_warm_start.py`
- `longitudinal_to_cartesian` maps arc-length = 0 to the start of the reference path.
- `rank_and_filter` with m=5 returns ≤ 5 trajectories sorted by return.
- `WarmStartReceiver.receive(trajectories)` is called with the correct format.

---

## 10. Implementation Order

Claude Code should implement modules in this order to allow incremental testing:

1. `config.py` — all configuration constants.
2. `mdp/state.py`, `mdp/action.py` — pure data structures.
3. `mdp/transition.py` — kinematic model (unit-testable in isolation).
4. `mdp/reward.py`, `mdp/termination.py`.
5. `mcts/policies.py` — IDM, CS policies.
6. `mcts/node.py`, `mcts/search.py` — core MCTS.
7. `mcts/trajectory.py` — DFS extraction.
8. `belief/interface.py` — abstract interface + NullBeliefModel.
9. `warm_start/converter.py`, `warm_start/interface.py`.
10. `planner.py` — integration.
11. `tests/` — all test files.

**At each section boundary marked ⚠️ CLARIFICATION REQUIRED, stop, ask the user
the listed questions, and wait for answers before proceeding.**

---

## 11. Deferred / Out-of-Scope Items

The following are explicitly out of scope for this implementation pass and should
be left as stub interfaces with `NotImplementedError`:

- Training of the RL network `f_θ` (the paper's PPO-trained policy).
- IRL scorer / DriveIRL scoring (replaced by return-based ranking for warm-starting).
- Lateral control / lane-change actions (§3.2, Q5).
- Multi-modal prediction integration (single top-mode prediction assumed).
- Reactive world model (non-reactive as in paper).

---

## 12. Reference

Tomov, M. S. et al. "TreeIRL: Safe Urban Driving with Tree Search and Inverse
Reinforcement Learning." arXiv:2509.13579v4, 2025.

Key paper sections to re-read during implementation:
- §III-C: MCTS algorithm (Algorithm 1) — canonical reference for search loop.
- §IV-A: MDP components — state/action/transition/reward definitions.
- §IV-B: MCTS components — UCB, rollout, padding policy choices.
- §V-A: Tuning MCTS — justification for `n=400`, uniform prior, IDM rollout.
