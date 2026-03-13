# Per-Belief MCTS Planner Specification

## Context

We have an existing MCTS planner for autonomous driving motion planning. We need to extend it to support **per-belief value estimates** for a shared autonomy system where the human driver may have incorrect beliefs about the traffic scene (e.g., not seeing certain vehicles). The system infers the human's beliefs from their actions and plans accordingly.

## Problem Setting

- An ego vehicle is jointly controlled by a human driver and an assistive system
- There are `n` traffic participants in the scene
- The human has **latent parameters** `θ = (d¹, d², ..., dⁿ)` where `dⁱ ∈ {0, 1}` indicates whether the human believes participant `i` is present
- This gives `|Θ| = 2^n` possible latent configurations (in practice n is small, 1-3)
- The assistive system maintains a **belief distribution** `b(θ)` over these configurations
- Both the human and the assistive system follow the same near-optimal driving policy, but conditioned on different beliefs about which traffic participants exist
- The latent parameters only affect **costs and constraints** (which collision avoidance constraints are active), NOT the physical dynamics. A trajectory is physically identical under all θ — only the cost evaluation differs.

## What Needs to Change

### 1. Per-Belief Q Values

**Current**: each node stores `Q(s, a)` — a single scalar value per action.

**New**: each node stores `Q_θ(s, a)` — a **vector** of values per action, one for each latent configuration `θ ∈ Θ`.

Implementation:
- `Q` should become a dictionary or 2D structure: `Q[action][theta_config] -> float`
- `N(s, a)` visit counts remain scalar (visits are shared across beliefs)
- Additionally store or be able to compute a belief-weighted aggregate: `Q_bar(s, a) = sum over θ of b(θ) * Q_θ(s, a)`

### 2. Simulation Loop Changes

**Current (standard MCTS)**: sample state, traverse tree, rollout, evaluate, backup with single cost.

**New**: the simulation loop changes as follows:

#### a) Selection (traversing the tree)
At the start of each simulation:
1. **Sample** a latent configuration `θ_sampled` from the current belief distribution `b(θ)`
2. During tree traversal, use UCB with the **sampled θ's Q values**:
   ```
   a* = argmax_a [ Q_θ_sampled(s, a) + c * sqrt(ln(sum_a' N(s,a')) / N(s,a)) ]
   ```
   This means different simulations may follow different paths through the tree depending on which θ was sampled.

#### b) Expansion
No change — add new nodes as normal when reaching unexplored states.

#### c) Evaluation (rollout)
Perform the rollout as normal using a default policy (e.g., constant speed, IDM, or any existing rollout policy). The rollout produces a single physical trajectory.

**Key change**: evaluate the rollout trajectory under **ALL** latent configurations simultaneously:
- For each `θ ∈ Θ`, compute the return `R_θ` by summing discounted costs along the rollout trajectory using θ-specific costs
- The cost difference between configurations comes from **collision avoidance**: if `dⁱ = 0` in configuration θ, then collisions with participant `i` are NOT penalised under that θ. If `dⁱ = 1`, collisions with participant `i` ARE penalised.
- All other cost terms (progress, comfort, road boundaries, actuator limits) are identical across all θ

This means: one rollout → `|Θ|` different return values.

#### d) Backup
Propagate returns back up the tree for **ALL** θ simultaneously:
- For each ancestor node along the visited path, and for each `θ ∈ Θ`:
  ```
  N(s, a) += 1  (shared across all θ)
  Q_θ(s, a) += (R_θ - Q_θ(s, a)) / N(s, a)
  ```
  
### 3. Belief Update After Search

After the MCTS search completes (all iterations done), use the root node's per-belief Q values to update the belief about the human's latent parameters.

Given the human's **observed action** `u_H_k` (the action the human actually took / is taking):

1. For each `θ ∈ Θ`, compute the Boltzmann likelihood:
   ```
   P(u_H_k | s_0, θ) = exp(β * Q_θ(s_0, u_H_k)) / sum_a' exp(β * Q_θ(s_0, a'))
   ```
   where `β > 0` is a rationality parameter (higher β = human acts more optimally).

2. Map the human's continuous action to the nearest discrete action in `A` for lookup.

3. Update the belief using Bayes' rule:
   ```
   b_new(θ) ∝ P(u_H_k | s_0, θ) * b_old(θ)
   ```
   Then normalise so `sum_θ b_new(θ) = 1`.

Note: if latent parameter dynamics `P_θ(θ' | θ, s, u)` are implemented, include a prediction step before the observation update:
   ```
   b_predicted(θ') = sum_θ P_θ(θ' | θ, s, u) * b_old(θ)
   b_new(θ') ∝ P(u_H_k | s_0, θ') * b_predicted(θ')
   ```
For now, if no transition model is implemented, assume θ is static within the planning horizon (i.e., `P_θ(θ' | θ, s, u) = δ(θ' = θ)`), which simplifies to the direct Bayes update above.

### 4. Trajectory Extraction

After the belief update, extract a coarse trajectory for NLP warmstarting:

1. Determine the most likely configuration: `θ* = argmax_θ b(θ)`
2. Perform a **greedy traversal** of the tree from the root, at each node selecting:
   ```
   a* = argmax_a Q_θ*(s, a)
   ```
3. If a leaf is reached before the planning horizon, pad with a default policy (e.g., constant speed or IDM) as in the TreeIRL approach.
4. The resulting sequence of states and actions is the coarse trajectory to pass to the NLP solver.

## Cost Function Specification

The cost function for evaluating trajectories needs to accept a θ configuration. The key difference between configurations:

```python
def compute_cost(trajectory, theta, participants):
    cost = 0
    for step in trajectory:
        # These costs are the SAME for all theta
        cost += progress_cost(step)
        cost += comfort_cost(step)  # jerk, acceleration penalties
        cost += road_boundary_cost(step)
        cost += actuator_limit_cost(step)
        
        # These costs DEPEND on theta
        for i, participant in enumerate(participants):
            if theta[i] == 1:  # participant is "visible" under this belief
                cost += collision_avoidance_cost(step, participant)
            # if theta[i] == 0, no collision cost for this participant
    
    return cost
```

## Data Structures

```python
class MCTSNode:
    state: VehicleState          # observable state s_k
    children: dict[Action, MCTSNode]
    N: dict[Action, int]         # visit counts per action (shared across θ)
    Q: dict[Action, dict[ThetaConfig, float]]  # per-belief Q values
    
class BeliefState:
    configs: list[ThetaConfig]   # all possible θ configurations
    probs: dict[ThetaConfig, float]  # b(θ) for each configuration
    
    def sample(self) -> ThetaConfig:
        """Sample θ from current belief distribution"""
        
    def update(self, likelihood: dict[ThetaConfig, float]):
        """Bayesian update: b_new(θ) ∝ likelihood(θ) * b_old(θ)"""
        
class ThetaConfig:
    """A specific latent parameter configuration, e.g., (1, 0, 1) for 3 participants"""
    visibility: tuple[int, ...]  # (d¹, d², ..., dⁿ)
```

## Algorithm Pseudocode

```
function PLAN(root_state, belief, n_iterations, participants):
    root = MCTSNode(root_state)
    
    for i in 1..n_iterations:
        # Sample belief for this simulation
        θ_sampled = belief.sample()
        
        # Run one MCTS simulation
        SIMULATE(root, θ_sampled, participants)
    
    # After search: update belief from human's observed action
    u_human = get_human_action()
    likelihood = {}
    for θ in belief.configs:
        Q_vals = {a: root.Q[a][θ] for a in actions}
        likelihood[θ] = boltzmann(u_human, Q_vals, β)
    belief.update(likelihood)
    
    # Extract trajectory under best belief
    θ_star = belief.most_likely()
    coarse_trajectory = EXTRACT_TRAJECTORY(root, θ_star)
    
    return coarse_trajectory, belief


function SIMULATE(node, θ_sampled, participants):
    if node is terminal:
        return {θ: 0 for θ in all_configs}
    
    # Selection: use sampled θ for UCB
    if node is fully expanded:
        a = argmax_a [Q_θ_sampled(node, a) + UCB_exploration(node, a)]
    else:
        a = expand next untried action
        
    child = node.children[a]
    next_state = transition(node.state, a)  # deterministic dynamics
    
    if child is new (leaf):
        # Rollout and evaluate under ALL θ
        rollout_trajectory = rollout(next_state, default_policy)
        returns = {}
        for θ in all_configs:
            returns[θ] = evaluate_cost(rollout_trajectory, θ, participants)
    else:
        returns = SIMULATE(child, θ_sampled, participants)
    
    # Backup: update Q for ALL θ
    reward = {}
    for θ in all_configs:
        reward[θ] = step_cost(node.state, a, θ, participants)
    
    node.N[a] += 1
    for θ in all_configs:
        total_return = reward[θ] + γ * returns[θ]
        node.Q[a][θ] += (total_return - node.Q[a][θ]) / node.N[a]
    
    return {θ: reward[θ] + γ * returns[θ] for θ in all_configs}


function EXTRACT_TRAJECTORY(root, θ_star):
    trajectory = []
    node = root
    while node is not terminal and node has children:
        a = argmax_a Q_θ_star(node, a)
        trajectory.append((node.state, a))
        node = node.children[a]
    
    # Pad with default policy if needed
    while not at horizon:
        a = default_policy(current_state)
        trajectory.append((current_state, a))
        current_state = transition(current_state, a)
    
    return trajectory
```

## Important Notes

- `|Θ| = 2^n` where n is the number of traffic participants. For n=1 this is 2 configs, n=2 is 4, n=3 is 8. Keep n small (1-3 relevant participants).
- The per-belief evaluation adds a constant factor of `|Θ|` to each simulation's cost, NOT exponential growth in the tree.
- Visit counts `N(s, a)` are shared across all θ — we don't maintain separate visit counts per belief.
- The physical state transitions are IDENTICAL across all θ. Only the cost computation differs.
- The belief `b(θ)` is updated in the outer loop (between planning cycles), not inside the tree search.
- The tree search output is used for **belief inference only**. The actual executed trajectory comes from the NLP solver that is warmstarted with the extracted coarse trajectory.
