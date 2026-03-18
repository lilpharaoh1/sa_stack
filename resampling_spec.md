# Information-Guided Resampling Specification

## Context

We have an existing per-belief MCTS planner where:
- Each node stores `Q_θ(s, a)` for each latent configuration `θ ∈ Θ`
- At the start of each simulation, a `θ` is sampled from the belief distribution `b(θ)`
- That `θ` is used for UCB action selection throughout the entire simulation
- After rollout, all `Q_θ` values are updated for all configurations

We want to add a **resampling mechanism** during the selection phase so that the active `θ` can switch mid-simulation at nodes where the latent configurations disagree about the best action.

## What to Add

### 1. Policy Divergence Computation

At each node visited during the selection phase, compute how much the configurations disagree about what action to take.

**Input**: `Q_θ(s, a)` for all `θ ∈ Θ` and all `a ∈ A` at the current node, plus a rationality parameter `β`.

**Procedure**:

```python
def compute_policy_divergence(Q_values, theta_configs, actions, beta):
    """
    Q_values: dict[theta][action] -> float
    Returns: scalar I(s) >= 0
    """
    # Step 1: convert Q values to Boltzmann policies per configuration
    policies = {}
    for theta in theta_configs:
        logits = [beta * Q_values[theta][a] for a in actions]
        max_logit = max(logits)  # subtract max for numerical stability
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        policies[theta] = [e / total for e in exps]
    
    # Step 2: compute mixture distribution (uniform weights)
    n_configs = len(theta_configs)
    mixture = [0.0] * len(actions)
    for i in range(len(actions)):
        for theta in theta_configs:
            mixture[i] += policies[theta][i] / n_configs
    
    # Step 3: compute KL(π_θ || M) for each configuration
    kl_values = []
    for theta in theta_configs:
        kl = 0.0
        for i in range(len(actions)):
            if policies[theta][i] > 1e-10:  # avoid log(0)
                kl += policies[theta][i] * math.log(policies[theta][i] / mixture[i])
        kl_values.append(kl)
    
    # Step 4: JSD = average of KL divergences
    jsd = sum(kl_values) / n_configs
    
    return jsd
```

### 2. Resampling Decision

At each node during selection, after computing the policy divergence, decide whether to resample.

**Parameters**:
- `gamma`: temperature controlling sigmoid sensitivity (suggested starting value: 3-5)
- `eta`: threshold for resampling (suggested starting value: 0.6-0.8)

```python
def should_resample(info_value, gamma, eta):
    """Returns True if we should resample θ at this node"""
    p_resample = 1.0 / (1.0 + math.exp(-gamma * info_value))  # sigmoid
    return p_resample > eta
```

### 3. Resampling Action

When resampling is triggered, draw a new `θ` **uniformly** from all configurations (NOT from the belief distribution).

```python
def resample_theta(theta_configs):
    """Draw uniformly from all latent configurations"""
    return random.choice(theta_configs)
```

**Important**: the initial `θ` at the root is still sampled from `b(θ)`. Only mid-simulation resampling uses uniform.

### 4. Integration into Selection Loop

Modify the existing selection phase. Currently it looks something like:

```python
# CURRENT: single θ for entire simulation
theta = sample_from_belief(b)
node = root
while node is not leaf:
    action = select_ucb(node, theta)
    node = node.children[action]
```

Change to:

```python
# NEW: θ can be resampled at informative nodes
theta = sample_from_belief(b)  # initial sample from belief
node = root
while node is not leaf:
    # Check for resampling
    info_value = compute_policy_divergence(node.Q, theta_configs, actions, beta)
    if should_resample(info_value, gamma, eta):
        theta = resample_theta(theta_configs)  # uniform resample
    
    # Select action using (possibly resampled) θ
    action = select_ucb(node, theta)
    node = node.children[action]
```

### 5. What Does NOT Change

- **Expansion**: unchanged
- **Rollout/evaluation**: unchanged — still evaluate under all θ simultaneously
- **Backup**: unchanged — still update all Q_θ values along the path
- **Belief update**: unchanged — still uses root Q values with Boltzmann likelihood
- **Trajectory extraction**: unchanged — still traverses under θ* = argmax b(θ)
- **Visit counts**: unchanged — N(s, a) is still shared across all θ

## Edge Cases

- **Early iterations**: Q values are uninformative, so the Boltzmann policies will be near-uniform for all θ, JSD will be near zero, and resampling will rarely trigger. This is correct — there's nothing meaningful to resample on yet.
- **Single configuration**: if |Θ| = 1, JSD is always 0 and resampling never triggers. Correct.
- **All Q values equal for an action**: the Boltzmann policies will be identical across θ, JSD = 0, no resampling. Correct.
- **Very large Q differences**: ensure numerical stability in the Boltzmann computation by subtracting the max logit before exponentiating (included in the pseudocode above).

## Parameters Summary

| Parameter | Role | Suggested Range |
|-----------|------|-----------------|
| `beta` | Boltzmann rationality for policy divergence | 0.5 - 2.0 |
| `gamma` | Sigmoid temperature for resampling probability | 3.0 - 10.0 |
| `eta` | Resampling threshold | 0.6 - 0.8 |

Higher `gamma` = sharper transition between "no resample" and "always resample".
Higher `eta` = less resampling, more coherent single-θ trajectories.
Lower `eta` = more resampling, more hybrid trajectories explored.
