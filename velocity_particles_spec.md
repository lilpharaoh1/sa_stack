# Velocity Estimation Particles Specification

## Overview

This spec describes how to add velocity estimation error inference to the existing per-belief MCTS planner. The existing system uses a Kalman filter for visibility awareness (φ) and maintains per-configuration Q values. We now add a particle-based representation for the human's velocity scaling factor (κ) for each traffic participant, giving us a richer latent parameter space.

## Current System (What Exists)

- Kalman filter tracks awareness ψ̂_φ and P_φ per participant
- Discrete configurations: {invisible, visible} per participant (2 configs for n=1)
- Per-configuration Q values: Q_θ(s, a) at each tree node
- Boltzmann belief update using log likelihood ratio for awareness
- Awareness-guided resampling inside MCTS using Kalman propagation
- Safety-filtered trajectory extraction using relative regret

## What Changes

### 1. New Latent Parameter: Velocity Scaling Factor κ

Each participant i has a velocity scaling factor κ^i ∈ [κ_min, 1] representing the ratio of the human's estimated velocity to the true velocity. κ = 1 means the human has a correct estimate. κ < 1 means the human underestimates the speed.

### 2. Particle Representation for κ

For each participant i, maintain K weighted particles:

```python
class VelocityParticles:
    kappa_values: list[float]  # K particles, each in [kappa_min, 1.0]
    weights: list[float]       # K weights, sum to 1.0
    
    def __init__(self, K, kappa_min=0.3):
        # Initialise particles uniformly across the range
        self.kappa_values = [kappa_min + j * (1.0 - kappa_min) / (K - 1) for j in range(K)]
        self.weights = [1.0 / K] * K
    
    def weighted_mean(self):
        return sum(w * k for w, k in zip(self.weights, self.kappa_values))
    
    def effective_sample_size(self):
        return 1.0 / sum(w**2 for w in self.weights)
```

**Suggested K = 4 particles** for n=1. This gives 1 + 4 = 5 configurations per participant.
Example initial particles for κ_min = 0.3: {0.30, 0.53, 0.77, 1.00}

### 3. Configuration Space

For each participant i, the configurations are:

```python
configs_for_participant_i = [
    ("invisible", None),           # participant not seen
    ("visible", kappa_values[0]),   # visible, worst velocity estimate
    ("visible", kappa_values[1]),   # visible, poor velocity estimate
    ("visible", kappa_values[2]),   # visible, decent velocity estimate
    ("visible", kappa_values[3]),   # visible, correct velocity estimate
]
```

For n=1 participant: 5 total configurations.
For n=2 participants: 25 total configurations (product of per-participant configs).

The ground truth configuration is always ("visible", 1.0) for all participants — the assistive system knows the true velocity.

### 4. Configuration Belief

The probability of each configuration combines the Kalman-based visibility probability with the particle weights:

```python
def compute_config_belief(p_phi, velocity_particles):
    """
    p_phi: float, probability of visible from Kalman CDF computation
    velocity_particles: VelocityParticles
    
    Returns: dict mapping config -> probability
    """
    belief = {}
    belief[("invisible", None)] = 1.0 - p_phi
    
    for j in range(K):
        belief[("visible", velocity_particles.kappa_values[j])] = (
            p_phi * velocity_particles.weights[j]
        )
    
    return belief
```

### 5. Perceived Trajectory Computation

For each visible configuration with a specific κ, the perceived trajectory of participant i is:

```python
def perceived_trajectory(true_positions, current_position, kappa):
    """
    true_positions: list of (x, y) for each future timestep k+1, k+2, ..., k+N
    current_position: (x, y) at current timestep k
    kappa: velocity scaling factor
    
    Returns: list of perceived (x, y) positions
    """
    perceived = []
    for p_true in true_positions:
        displacement = (p_true[0] - current_position[0], 
                       p_true[1] - current_position[1])
        p_perceived = (
            current_position[0] + kappa * displacement[0],
            current_position[1] + kappa * displacement[1]
        )
        perceived.append(p_perceived)
    return perceived
```

For the invisible configuration: no collision avoidance constraint at all.
For visible with κ = 1.0: perceived trajectory = true trajectory.

### 6. Cost Evaluation Changes

The cost function now needs to evaluate collisions against the perceived trajectory for each configuration:

```python
def compute_cost(trajectory, config, participant, true_positions):
    cost = 0.0
    
    # Costs that are the SAME for all configs
    cost += progress_cost(trajectory)
    cost += comfort_cost(trajectory)
    cost += road_boundary_cost(trajectory)
    
    # Costs that DEPEND on config
    config_type, kappa = config
    
    if config_type == "invisible":
        # No collision cost for this participant
        pass
    elif config_type == "visible":
        # Compute perceived trajectory using this config's kappa
        perceived = perceived_trajectory(true_positions, current_pos, kappa)
        cost += collision_avoidance_cost(trajectory, perceived)
    
    return cost
```

### 7. Velocity Particle Dynamics (Between Planning Cycles)

After each real-world timestep, propagate each particle's κ value forward:

```python
def propagate_velocity_particles(particles, ego_state, participant, phi_i, params):
    """
    Propagate each particle's kappa toward 1.0 based on perceptual features.
    
    particles: VelocityParticles
    ego_state: current ego vehicle state
    participant: participant state
    phi_i: current awareness estimate for participant i (from Kalman)
    params: dict with b_kappa, sigma_3, alpha_fov, q_kappa
    """
    # Compute velocity feature
    f_kappa = compute_velocity_feature(ego_state, participant, phi_i, 
                                        params['sigma_3'], params['alpha_fov'])
    
    for j in range(len(particles.kappa_values)):
        kappa = particles.kappa_values[j]
        
        # Drift toward 1.0, modulated by feature
        kappa_new = kappa + params['b_kappa'] * f_kappa * (1.0 - kappa)
        
        # Add process noise
        kappa_new += random.gauss(0, math.sqrt(params['q_kappa']))
        
        # Clamp to valid range
        kappa_new = max(params['kappa_min'], min(1.0, kappa_new))
        
        particles.kappa_values[j] = kappa_new


def compute_velocity_feature(ego_state, participant, phi_i, sigma_3, alpha_fov):
    """
    Velocity estimation feature: RBF gated by FOV and modulated by awareness.
    """
    dx = ego_state.x - participant.x
    dy = ego_state.y - participant.y
    dist_sq = dx**2 + dy**2
    
    # RBF
    rbf = math.exp(-dist_sq / (2 * sigma_3**2))
    
    # FOV gating
    angle_to_participant = math.atan2(dy, dx)
    relative_angle = angle_to_participant - ego_state.heading
    relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
    in_fov = abs(relative_angle) <= alpha_fov
    
    if not in_fov:
        rbf = 0.0
    
    # Modulate by awareness
    return rbf * phi_i
```

### 8. Velocity Particle Observation Update (After MCTS, When Human Acts)

Reweight particles based on how consistent the human's action is with each particle's κ:

```python
def update_velocity_particles(particles, root_node, u_H, beta, actions):
    """
    Reweight velocity particles using Boltzmann likelihoods from tree Q values.
    
    particles: VelocityParticles for participant i
    root_node: MCTS root node with Q values
    u_H: human's observed action
    beta: Boltzmann rationality parameter
    actions: list of discrete actions
    """
    for j in range(len(particles.kappa_values)):
        config = ("visible", particles.kappa_values[j])
        
        # Compute Boltzmann likelihood of human's action under this particle's config
        Q_vals = {a: root_node.Q[a][config] for a in actions}
        
        # Numerical stability
        max_q = max(Q_vals.values())
        exp_vals = {a: math.exp(beta * (q - max_q)) for a, q in Q_vals.items()}
        total = sum(exp_vals.values())
        
        likelihood = exp_vals[u_H] / total
        
        # Reweight
        particles.weights[j] *= likelihood
    
    # Normalise
    total_weight = sum(particles.weights)
    if total_weight > 1e-10:
        particles.weights = [w / total_weight for w in particles.weights]
    else:
        # All weights collapsed, reset to uniform
        particles.weights = [1.0 / len(particles.weights)] * len(particles.weights)
    
    # Check effective sample size and resample if needed
    n_eff = particles.effective_sample_size()
    if n_eff < len(particles.kappa_values) / 2:
        resample_particles(particles)


def resample_particles(particles):
    """
    Systematic resampling to prevent weight degeneracy.
    Resample indices from weighted distribution, then add small noise.
    """
    K = len(particles.kappa_values)
    cumulative = []
    running = 0.0
    for w in particles.weights:
        running += w
        cumulative.append(running)
    
    # Systematic resampling
    new_kappas = []
    u = random.uniform(0, 1.0 / K)
    idx = 0
    for j in range(K):
        threshold = u + j / K
        while cumulative[idx] < threshold and idx < K - 1:
            idx += 1
        new_kappas.append(particles.kappa_values[idx])
    
    # Add small noise to prevent particle collapse
    noise_std = 0.05  # tuneable
    particles.kappa_values = [
        max(particles.kappa_min, min(1.0, k + random.gauss(0, noise_std)))
        for k in new_kappas
    ]
    particles.weights = [1.0 / K] * K
```

### 9. Changes to MCTS Simulation Loop

The simulation loop changes minimally. The main differences:

```python
def simulate(root, psi_hat_phi, P_phi, velocity_particles, beta, eta):
    """
    Run one MCTS simulation with awareness Kalman resampling.
    Velocity particles are fixed within the planning cycle.
    """
    # Compute belief from Kalman state + particle weights
    p_phi = compute_p_phi(psi_hat_phi, P_phi)
    config_belief = compute_config_belief(p_phi, velocity_particles)
    
    # Sample initial configuration from belief
    theta = sample_from(config_belief)
    
    # Initialise Kalman state for awareness (velocity particles don't propagate in tree)
    psi_phi = psi_hat_phi
    P_phi_local = P_phi
    
    node = root
    path = []
    
    while node is not terminal:
        # Select action using UCB with current theta
        action = select_ucb(node, theta)
        path.append((node, action))
        
        # Advance to child
        child = node.children[action]
        next_state = child.state
        
        # Propagate Kalman prediction for AWARENESS ONLY
        psi_phi, P_phi_local = kalman_predict_phi(psi_phi, P_phi_local, 
                                                    next_state, participants)
        
        # Compute local belief (awareness from Kalman, velocity from fixed particles)
        p_phi_local = compute_p_phi(psi_phi, P_phi_local)
        b_local = compute_config_belief(p_phi_local, velocity_particles)
        
        # Check if current theta is still plausible
        if b_local[theta] < eta:
            theta = sample_from(b_local)
        
        node = child
    
    # Rollout from leaf
    rollout_trajectory = rollout(node.state, default_policy)
    
    # Evaluate rollout under ALL configurations
    returns = {}
    for config in all_configs:
        returns[config] = evaluate_cost(rollout_trajectory, config, participants)
    
    # Backup: update Q for ALL configurations along path
    for (node, action) in reversed(path):
        node.N[action] += 1
        for config in all_configs:
            reward = step_cost(node.state, action, config, participants)
            total_return = reward + gamma_discount * returns[config]
            node.Q[action][config] += (total_return - node.Q[action][config]) / node.N[action]
            returns[config] = total_return
```

### 10. Full Planning Loop

```python
# Initialise
psi_hat_phi = np.zeros(n_participants)  # awareness: neutral prior
P_phi = np.ones(n_participants)          # awareness: moderate uncertainty

velocity_particles = {
    i: VelocityParticles(K=4, kappa_min=0.3) 
    for i in range(n_participants)
}

for each planning cycle:
    # 1. Compute configuration belief
    beliefs = {}
    for i in range(n_participants):
        p_phi_i = compute_p_phi(psi_hat_phi[i], P_phi[i])
        beliefs[i] = compute_config_belief(p_phi_i, velocity_particles[i])
    
    full_belief = compute_joint_belief(beliefs)  # product across participants
    
    # 2. Run MCTS
    root = build_tree(current_state)
    for iter in range(n_iterations):
        simulate(root, psi_hat_phi, P_phi, velocity_particles, beta, eta)
    
    # 3. Safety-filtered trajectory extraction
    theta_star = argmax(full_belief)
    theta_R = ground_truth_config  # ("visible", 1.0) for all participants
    trajectory = extract_safe_trajectory(root, theta_star, theta_R, gamma, actions)
    
    # 4. Pass coarse trajectory to NLP for refinement
    smooth_trajectory = nlp_refine(trajectory)
    
    # 5. Execute first action
    execute(smooth_trajectory[0])
    
    # 6. Observe human's actual action
    u_H = observe_human_action()
    
    # 7. Awareness: Kalman prediction + observation update
    for i in range(n_participants):
        psi_hat_phi[i], P_phi[i] = kalman_predict_phi(
            psi_hat_phi[i], P_phi[i], current_state, participants[i])
        
        y_phi = compute_awareness_log_likelihood_ratio(
            root, u_H, i, all_configs, beta)
        
        if abs(y_phi) > y_min:
            psi_hat_phi[i], P_phi[i] = kalman_update_phi(
                psi_hat_phi[i], P_phi[i], y_phi, R_phi)
    
    # 8. Velocity: particle propagation + reweighting
    for i in range(n_participants):
        phi_i = sigmoid(psi_hat_phi[i])
        
        propagate_velocity_particles(
            velocity_particles[i], current_state, participants[i], 
            phi_i, velocity_params)
        
        update_velocity_particles(
            velocity_particles[i], root, u_H, beta, actions)
```

### 11. Computing the Awareness Log Likelihood Ratio

This needs to aggregate across visible configurations (all κ particles) vs invisible:

```python
def compute_awareness_log_likelihood_ratio(root, u_H, participant_i, all_configs, beta):
    """
    Aggregate likelihoods across visible (all kappa particles) vs invisible.
    """
    L_visible = 0.0
    L_invisible = 0.0
    
    for config in all_configs:
        # Compute Boltzmann likelihood for this config
        Q_vals = {a: root.Q[a][config] for a in actions}
        max_q = max(Q_vals.values())
        exp_vals = {a: math.exp(beta * (q - max_q)) for a, q in Q_vals.items()}
        total = sum(exp_vals.values())
        likelihood = exp_vals[u_H] / total
        
        # Check if participant i is visible or invisible in this config
        config_for_i = config[participant_i]  # extract participant i's part
        if config_for_i[0] == "invisible":
            L_invisible += likelihood
        else:
            L_visible += likelihood
    
    # Avoid log(0)
    L_visible = max(L_visible, 1e-10)
    L_invisible = max(L_invisible, 1e-10)
    
    return math.log(L_visible / L_invisible)
```

## Parameters Summary

| Parameter | Symbol | Starting Value | Role |
|-----------|--------|---------------|------|
| Number of κ particles | K | 4 | Resolution of velocity belief |
| Minimum κ | κ_min | 0.3 | Worst possible velocity estimate |
| Velocity drift rate | b_κ | 0.1 | Speed of velocity estimate improvement |
| Velocity process noise | q_κ | 0.01 | Uncertainty growth in κ dynamics |
| Velocity feature spread | σ_3 | tuneable | RBF spread for velocity feature |
| Resample noise | - | 0.05 | Noise added after particle resampling |
| N_eff threshold | - | K/2 | When to resample particles |
| Awareness persistence | a_φ | 1.0 | Awareness decay (1 = no decay) |
| Awareness feature weight | b_φ | 1.0 | Strength of proximity → awareness |
| Awareness process noise | q_φ | 0.05 | Uncertainty growth per step |
| Awareness observation noise | R_φ | 1.0 | Trust in awareness observations |
| Awareness threshold | φ_th | 0.5 | Visible/invisible boundary |
| Resampling threshold | η | 0.15 | Plausibility threshold for tree resampling |
| Rationality | β | 1.0 | Boltzmann temperature |
| Safety parameter | γ | 0.5 | Safety-agency trade-off |
| Informativeness threshold | y_min | 0.1 | Gating for awareness Kalman update |

## What Does NOT Change

- Kalman filter for awareness: prediction, observation update, gating — all unchanged
- MCTS structure: UCB, expansion, rollout, backup — same logic, just more configs
- Awareness-guided resampling inside tree: only awareness Kalman propagates, velocity particles are fixed within planning cycle
- Safety-filtered trajectory extraction: same relative regret logic, θ_R is now ("visible", 1.0)
- NLP refinement: unchanged

## Running Velocity-Only Experiments

To run experiments with velocity error only (no visibility uncertainty):

- Remove the invisible configuration
- All participants are always visible
- Configurations are just the κ particles: {κ_1, κ_2, ..., κ_K}
- No Kalman filter needed — only particle reweighting
- Skip the awareness log likelihood ratio computation
- Resampling in tree is disabled (or based on velocity particle weight changes)
- θ_R = κ = 1.0 for all participants

This provides a clean ablation to isolate the velocity estimation component.
