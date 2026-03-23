# Kalman-Filtered Awareness Dynamics for Per-Belief MCTS

## Overview

This spec describes how to integrate a Kalman filter model of human awareness into the existing per-belief MCTS planner. The Kalman filter tracks the robot's estimate of the human's awareness of each traffic participant, and this estimate is propagated forward during MCTS simulations to guide resampling of latent configurations. Between planning cycles, the estimate is updated using the human's observed action.

## Awareness Representation

### Continuous Awareness State

For each traffic participant i ∈ {1, ..., n}, the robot maintains:

- `phi_i` ∈ [0, 1]: awareness probability (0 = unaware, 1 = fully aware, 0.5 = uncertain)
- `psi_i` ∈ ℝ: logit-transformed awareness, where `psi = log(phi / (1 - phi))`

The Kalman filter operates on `psi` (unconstrained space). Map back via `phi = sigmoid(psi) = 1 / (1 + exp(-psi))`.

### Discrete Configurations

The discrete latent configurations θ = (d¹, ..., dⁿ) with d^i ∈ {0, 1} are derived by thresholding:

```python
d_i = 1 if phi_i > phi_th else 0
```

With `phi_th = 0.5`, which corresponds to `psi = 0` in logit space.

### From Kalman State to b(θ)

Given the Kalman estimate (psi_hat, P) with diagonal covariance, the probability of each configuration θ = (d¹, ..., dⁿ) is:

```python
def compute_b_theta(psi_hat, P_diag, phi_th=0.5):
    """
    psi_hat: array of shape (n,) — mean awareness estimate per participant
    P_diag: array of shape (n,) — variance per participant (diagonal of P)
    
    Returns: dict mapping theta_config -> probability
    """
    psi_th = math.log(phi_th / (1 - phi_th))  # logit of threshold, = 0 for phi_th=0.5
    
    # Probability of aware per participant
    p_aware = []
    for i in range(n):
        # P(d_i = 1) = 1 - Phi((psi_th - psi_hat_i) / sqrt(P_ii))
        z = (psi_th - psi_hat[i]) / math.sqrt(P_diag[i])
        p_aware.append(1 - norm_cdf(z))
    
    # Enumerate all configurations and compute joint probability (independence)
    b_theta = {}
    for config in all_theta_configs:
        prob = 1.0
        for i in range(n):
            if config[i] == 1:
                prob *= p_aware[i]
            else:
                prob *= (1 - p_aware[i])
        b_theta[config] = prob
    
    return b_theta
```

## Kalman Filter Dynamics

### Prediction Step

```
psi_{k+1} = A * psi_k + B * f(s_k) + w_k,    w_k ~ N(0, Q)
```

### Parameters

**A = 1.0** (scalar, applied per participant)
- Awareness does not decay spontaneously
- In logit space, psi = 0 corresponds to phi = 0.5 (maximum uncertainty)
- With A = 1, awareness is monotonically non-decreasing given non-negative features

**B = b * I** (diagonal, single scalar b applied per participant)
- Each participant's awareness responds independently to its own features
- Starting value: b = 1.0, tune from there
- Larger b = awareness shifts faster in response to features

**Q = q * I** (diagonal, single scalar q)
- Process noise, controls uncertainty growth during prediction
- Starting value: q = 0.05
- Range: 0.01 to 0.1

**Initial conditions:**
- psi_hat_0 = 0.0 per participant (phi = 0.5, neutral prior)
- P_0 = 1.0 per participant (moderate uncertainty)

### Feature Vector f(s)

For each participant i, the feature is the sum of two radial basis functions:

```python
def compute_feature_i(ego_state, participant_state):
    """
    ego_state: (x, y, phi, v) — ego vehicle state
    participant_state: (x_i, y_i, phi_i, v_i) — participant state
    
    Returns: scalar feature value for participant i
    """
    # Ego and participant positions
    p_ego = np.array([ego_state.x, ego_state.y])
    p_i = np.array([participant_state.x, participant_state.y])
    
    dist_sq = np.sum((p_ego - p_i)**2)
    
    # RBF 1: omnidirectional proximity
    # Close objects are salient regardless of direction
    sigma_1 = ...  # narrow spread, needs tuning
    rbf_1 = math.exp(-dist_sq / (2 * sigma_1**2))
    
    # RBF 2: forward field-of-view, wider spread
    # Objects in front of the vehicle are noticeable at greater range
    sigma_2 = ...  # wider spread, sigma_2 > sigma_1
    rbf_2 = math.exp(-dist_sq / (2 * sigma_2**2))
    
    # Gate RBF 2 by 60-degree forward FOV (±30 degrees from heading)
    dx = participant_state.x - ego_state.x
    dy = participant_state.y - ego_state.y
    angle_to_participant = math.atan2(dy, dx)
    relative_angle = angle_to_participant - ego_state.phi
    # Wrap to [-pi, pi]
    relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
    
    in_fov = abs(relative_angle) <= math.radians(30)
    
    if not in_fov:
        rbf_2 = 0.0
    
    return rbf_1 + rbf_2
```

**Note:** f(s) does not depend on the action. Awareness changes are purely state-driven (proximity and field of view). Intervention effects on awareness are not modelled in this version.

### Kalman Prediction Implementation

```python
def kalman_predict(psi_hat, P_diag, ego_state, participants, A=1.0, b=1.0, q=0.05):
    """
    Propagate Kalman estimate forward one step.
    
    psi_hat: array of shape (n,) — current mean
    P_diag: array of shape (n,) — current variance (diagonal)
    ego_state: current ego vehicle state
    participants: list of participant states
    
    Returns: (psi_hat_new, P_diag_new)
    """
    n = len(participants)
    psi_hat_new = np.zeros(n)
    P_diag_new = np.zeros(n)
    
    for i in range(n):
        f_i = compute_feature_i(ego_state, participants[i])
        psi_hat_new[i] = A * psi_hat[i] + b * f_i
        P_diag_new[i] = A**2 * P_diag[i] + q
    
    return psi_hat_new, P_diag_new
```

## Integration into MCTS

### What Changes in the Simulation Loop

The existing per-belief MCTS simulation loop is modified as follows:

```python
def simulate(root, psi_hat_0, P_diag_0, beta, eta=0.15):
    """
    Run one MCTS simulation with Kalman-driven resampling.
    """
    # Compute initial b(theta) from Kalman state
    b_theta = compute_b_theta(psi_hat_0, P_diag_0)
    
    # Sample initial theta from belief distribution
    theta = sample_from(b_theta)
    
    # Initialise Kalman state for this simulation
    psi_hat = psi_hat_0.copy()
    P_diag = P_diag_0.copy()
    
    node = root
    path = []
    
    while node is not terminal:
        # Select action using UCB with current theta
        action = select_ucb(node, theta)
        path.append((node, action))
        
        # Advance to child
        child = node.children[action]
        next_state = child.state
        
        # Propagate Kalman prediction based on action taken
        psi_hat, P_diag = kalman_predict(psi_hat, P_diag, next_state, participants)
        
        # Compute local belief from propagated Kalman state
        b_local = compute_b_theta(psi_hat, P_diag)
        
        # Check if current theta is still plausible
        if b_local[theta] < eta:
            # Resample from local belief (NOT uniform)
            theta = sample_from(b_local)
        
        node = child
    
    # Rollout from leaf (freeze Kalman state, use b_local for theta sampling)
    rollout_trajectory = rollout(node.state, default_policy)
    
    # Evaluate rollout under ALL configurations
    returns = {}
    for theta_config in all_theta_configs:
        returns[theta_config] = evaluate_cost(rollout_trajectory, theta_config, participants)
    
    # Backup: update Q_theta for ALL configurations along path
    for (node, action) in reversed(path):
        node.N[action] += 1
        for theta_config in all_theta_configs:
            reward = step_cost(node.state, action, theta_config, participants)
            total_return = reward + gamma * returns[theta_config]
            node.Q[action][theta_config] += (total_return - node.Q[action][theta_config]) / node.N[action]
            returns[theta_config] = total_return
```

### What Does NOT Change

- Node data structure: still stores Q[action][theta] and N[action]
- Expansion: unchanged
- Rollout evaluation: unchanged — still evaluate under all θ simultaneously
- Backup: unchanged — still update all Q_θ values along the path
- Safety-filtered trajectory extraction: unchanged — still uses Q values at each node

### The Kalman State is NOT Stored at Nodes

The Kalman state (psi_hat, P_diag) is recomputed during each simulation by propagating from the root. Two simulations taking different action sequences will have different Kalman states at the same depth. This is correct — the awareness prediction depends on the path taken. Storing it would require history-dependent nodes which is unnecessary complexity since the propagation is cheap.

## Outer-Loop Observation Update

Between planning cycles, after the human takes a real action u_H:

### Step 1: Compute Boltzmann Likelihoods from Root Q Values

```python
def compute_likelihoods(root, u_H, all_theta_configs, beta):
    """
    Compute P(u_H | theta) for each configuration using root Q values.
    """
    likelihoods = {}
    for theta in all_theta_configs:
        Q_vals = {a: root.Q[a][theta] for a in actions}
        
        # Boltzmann: P(u_H | theta) = exp(beta * Q_theta(s, u_H)) / sum_a exp(beta * Q_theta(s, a))
        max_q = max(Q_vals.values())  # numerical stability
        exp_vals = {a: math.exp(beta * (q - max_q)) for a, q in Q_vals.items()}
        total = sum(exp_vals.values())
        
        likelihoods[theta] = exp_vals[u_H] / total
    
    return likelihoods
```

### Step 2: Compute Log Likelihood Ratio as Observation

For each participant i, aggregate the likelihoods across configurations where d_i = 1 versus d_i = 0:

```python
def compute_observation(likelihoods, all_theta_configs, n_participants):
    """
    Convert per-configuration likelihoods into per-participant
    log likelihood ratios for Kalman update.
    
    Returns: array of shape (n,) — one observation per participant
    """
    y = np.zeros(n_participants)
    
    for i in range(n_participants):
        L_aware = 0.0
        L_unaware = 0.0
        
        for theta in all_theta_configs:
            if theta[i] == 1:
                L_aware += likelihoods[theta]
            else:
                L_unaware += likelihoods[theta]
        
        # Avoid division by zero
        if L_aware < 1e-10:
            L_aware = 1e-10
        if L_unaware < 1e-10:
            L_unaware = 1e-10
        
        y[i] = math.log(L_aware / L_unaware)
    
    return y
```

### Step 3: Kalman Update

```python
def kalman_update(psi_hat, P_diag, y, R=1.0):
    """
    Update Kalman estimate with observation y.
    
    psi_hat: array of shape (n,) — predicted mean
    P_diag: array of shape (n,) — predicted variance
    y: array of shape (n,) — log likelihood ratio observations
    R: observation noise variance (scalar, same for all participants)
    
    Returns: (psi_hat_updated, P_diag_updated)
    """
    n = len(psi_hat)
    psi_hat_new = np.zeros(n)
    P_diag_new = np.zeros(n)
    
    for i in range(n):
        K = P_diag[i] / (P_diag[i] + R)      # Kalman gain
        psi_hat_new[i] = psi_hat[i] + K * (y[i] - psi_hat[i])  # update mean
        P_diag_new[i] = (1 - K) * P_diag[i]   # update variance
    
    return psi_hat_new, P_diag_new
```

**R = 1.0** (starting value): observation noise. Controls how much each human action shifts the estimate.
- Larger R = slower response to observations, more trust in dynamics model
- Smaller R = faster response, more trust in observations

## Full Planning Loop

```python
# Initialise
psi_hat = np.zeros(n_participants)  # neutral prior
P_diag = np.ones(n_participants)     # moderate uncertainty

for each planning cycle:
    # 1. Compute b(theta) from current Kalman state
    b_theta = compute_b_theta(psi_hat, P_diag)
    
    # 2. Run MCTS with Kalman-driven resampling
    root = build_tree(current_state)
    for i in range(n_iterations):
        simulate(root, psi_hat, P_diag, beta, eta)
    
    # 3. Safety-filtered trajectory extraction
    theta_star = argmax(b_theta)
    theta_R = ground_truth_config
    trajectory = extract_safe_trajectory(root, theta_star, theta_R, gamma, actions)
    
    # 4. Pass coarse trajectory to NLP for refinement
    smooth_trajectory = nlp_refine(trajectory)
    
    # 5. Execute first action
    execute(smooth_trajectory[0])
    
    # 6. Observe human's actual action
    u_H = observe_human_action()
    
    # 7. Kalman prediction step (advance one real timestep)
    psi_hat, P_diag = kalman_predict(psi_hat, P_diag, current_state, participants)
    
    # 8. Kalman observation update using tree Q values
    likelihoods = compute_likelihoods(root, u_H, all_theta_configs, beta)
    y = compute_observation(likelihoods, all_theta_configs, n_participants)
    psi_hat, P_diag = kalman_update(psi_hat, P_diag, y, R)
```

## Parameters Summary

| Parameter | Symbol | Starting Value | Role |
|-----------|--------|---------------|------|
| Persistence | A | 1.0 | Awareness decay (1 = no decay) |
| Feature weight | b | 1.0 | Strength of proximity → awareness |
| Process noise | q | 0.05 | Uncertainty growth per step |
| Observation noise | R | 1.0 | Trust in Boltzmann observations |
| Awareness threshold | phi_th | 0.5 | Continuous → discrete boundary |
| Resampling threshold | eta | 0.15 | Plausibility threshold for resampling |
| Rationality | beta | 1.0 | Boltzmann temperature |
| Safety parameter | gamma | 0.5 | Safety-agency trade-off |
| RBF narrow spread | sigma_1 | TBD | Omnidirectional proximity range |
| RBF wide spread | sigma_2 | TBD | Forward FOV detection range |
| Initial mean | psi_hat_0 | 0.0 | Neutral awareness prior |
| Initial variance | P_0 | 1.0 | Initial uncertainty |
