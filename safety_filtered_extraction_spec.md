# Safety-Filtered Trajectory Extraction Specification

## Context

We have an MCTS tree with per-configuration Q values Q_θ(s, a) at each node, for each latent configuration θ ∈ Θ and each discrete action a ∈ A. After the tree search and belief update, we need to extract a coarse trajectory that follows the human's intended actions as closely as possible while remaining safe under the true state of the environment.

## Inputs

- `root`: the root node of the MCTS tree
- `theta_star`: the most likely latent configuration θ* = argmax_θ b(θ) — the system's best estimate of the human's beliefs
- `theta_R`: the ground truth latent configuration — the true state known to the assistive system
- `gamma`: safety parameter ∈ [0, 1]. Higher γ = stricter safety (γ=1 only accepts optimal action under θ_R, γ=0 accepts anything)
- `max_depth`: planning horizon N

## Algorithm

At each node during the forward traversal:

### Step 1: Determine the human's preferred action

```python
a_H = argmax_a Q[theta_star][a]  # what the human would choose under their belief
```

### Step 2: Compute relative regret of the human's action under the true state

```python
Q_R = {a: node.Q[theta_R][a] for a in actions}
Q_max = max(Q_R.values())
Q_min = min(Q_R.values())

if Q_max == Q_min:
    # All actions have equal value under theta_R, no safety concern
    relative_regret = 0.0
else:
    relative_regret = (Q_max - Q_R[a_H]) / (Q_max - Q_min)
```

### Step 3: Apply safety filter

```python
if relative_regret <= (1 - gamma):
    # Human's action is safe enough, follow it
    chosen_action = a_H
    intervention = False
else:
    # Human's action is not safe, take the best action under theta_R
    chosen_action = argmax_a Q[theta_R][a]
    intervention = True
```

### Step 4: Advance to child node and repeat

```python
next_node = node.children[chosen_action]
```

## Full Extraction Loop

```python
def extract_safe_trajectory(root, theta_star, theta_R, gamma, actions, max_depth):
    """
    Extract a coarse trajectory from the MCTS tree that follows the human's
    preferred actions where safe, and overrides where necessary.
    
    Returns:
        trajectory: list of (state, action, intervention_flag) tuples
    """
    trajectory = []
    node = root
    
    for step in range(max_depth):
        if node is None or node has no children:
            break
        
        # Human's preferred action under their estimated belief
        a_H = argmax_a node.Q[theta_star][a]
        
        # Relative regret under true state
        Q_R = {a: node.Q[theta_R][a] for a in actions}
        Q_max = max(Q_R.values())
        Q_min = min(Q_R.values())
        
        if Q_max == Q_min:
            relative_regret = 0.0
        else:
            relative_regret = (Q_max - Q_R[a_H]) / (Q_max - Q_min)
        
        # Safety filter
        if relative_regret <= (1 - gamma):
            chosen_action = a_H
            intervention = False
        else:
            chosen_action = max(actions, key=lambda a: Q_R[a])
            intervention = True
        
        trajectory.append({
            'state': node.state,
            'action': chosen_action,
            'human_action': a_H,
            'intervention': intervention,
            'relative_regret': relative_regret,
        })
        
        # Advance to child
        if chosen_action in node.children:
            node = node.children[chosen_action]
        else:
            break  # action not expanded in tree, stop here
    
    return trajectory
```

## Outputs

A list of tuples, one per timestep, containing:

- `state`: the observable state at this node
- `action`: the action selected (either human's or robot's override)
- `human_action`: what the human would have chosen
- `intervention`: boolean flag indicating whether the system overrode the human
- `relative_regret`: the regret value at this node (useful for diagnostics)

This trajectory is then passed to the NLP solver as a warmstart.

## Edge Cases

- **All Q values equal for θ_R at a node**: relative regret is 0, human's action is always accepted. Correct — if the true state doesn't distinguish between actions, there's no safety concern.
- **Human's action is also optimal under θ_R**: relative regret is 0, no intervention. Correct — beliefs agree.
- **No children for the chosen action**: the tree wasn't expanded along this branch. Stop extraction and pad with a default policy for remaining steps.
- **γ = 0**: the safety condition is ρ ≤ 1, which is always true. No intervention ever — pure human control.
- **γ = 1**: the safety condition is ρ ≤ 0, meaning only the optimal action under θ_R is accepted. Maximum intervention — full robot override.

## Parameters

| Parameter | Role | Suggested Range |
|-----------|------|-----------------|
| `gamma` | Safety strictness | 0.3 - 0.8 |

Lower γ = more permissive, higher agency, potentially less safe.
Higher γ = stricter, lower agency, safer.

This parameter directly controls the safety-agency trade-off. Sweeping γ produces the Pareto curve comparing collision rate against intervention frequency.

## Important Notes

- The Q values used for computing relative regret are Q_{θ_R} — these reflect future costs under the true state, so the safety check accounts for downstream consequences, not just immediate danger.
- The Q values used for determining the human's preferred action are Q_{θ*} — these reflect what the human thinks is best under their (potentially incorrect) belief.
- The Q values used for selecting the override action are Q_{θ_R} — the system overrides with the best action under the true state.
- This is a greedy forward pass. It does not guarantee global optimality of the intervention strategy — there are edge cases where an earlier intervention would prevent a worse later intervention. A backward pass would handle this but adds complexity.
