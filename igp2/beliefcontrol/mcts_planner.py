"""MCTS trajectory planner with per-belief Q values.

Generates coarse trajectories via Monte Carlo Tree Search using
**longitudinal-only (1-D)** dynamics along the Frenet s-axis.
Each node stores per-belief Q values Q_θ(s, a) for every latent
configuration θ ∈ Θ, enabling Bayesian belief inference over the
human driver's latent visibility parameters.

Assumptions
-----------
* State is (s, v) — lateral offset d and heading phi are always 0.
* Actions are scalar accelerations a — no steering.
* Dynamics: s' = s + v·dt,  v' = clamp(v + a·dt, v_min, v_max).
* Collision checking is 1-D along s (longitudinal gap only).
* Road boundary checks are disabled (always on centreline).
* Output states/controls are zero-padded to (s, d=0, phi=0, v)
  and (a, delta=0) for downstream compatibility.

See ``per_belief_mcts_spec.md`` for the full specification.
"""

import itertools
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MCTSTrajectory:
    """A coarse trajectory produced by MCTS planning."""

    states: np.ndarray      # (K+1, 4) [s, d=0, phi=0, v]  coarse, padded for compat
    controls: np.ndarray    # (K, 2)   [a, delta=0]         coarse, padded for compat
    mcts_reward: float      # cumulative discounted reward
    interventions: Optional[List[bool]] = None  # per-step intervention flags (QCBF)


class BeliefState:
    """Belief distribution over latent visibility configurations.

    Each configuration θ = (d¹, d², ..., dⁿ) where dⁱ ∈ {0, 1}
    indicates whether the human believes participant i is present.
    """

    def __init__(self, agent_ids: List[int]):
        self._agent_ids = sorted(agent_ids)
        n = len(agent_ids)
        if n == 0:
            self._configs: List[tuple] = [()]
        else:
            self._configs = list(itertools.product((0, 1), repeat=n))
        self._probs: Dict[tuple, float] = {
            cfg: 1.0 / len(self._configs) for cfg in self._configs
        }

    @property
    def configs(self) -> List[tuple]:
        return self._configs

    @property
    def agent_ids(self) -> List[int]:
        return self._agent_ids

    @property
    def probs(self) -> Dict[tuple, float]:
        return self._probs

    def visible_aids(self, theta: tuple) -> Set[int]:
        """Return the set of agent IDs visible under configuration θ."""
        return {self._agent_ids[i] for i, v in enumerate(theta) if v == 1}

    def sample(self) -> tuple:
        """Sample a θ from the current belief distribution."""
        configs = list(self._probs.keys())
        probs = [self._probs[c] for c in configs]
        idx = np.random.choice(len(configs), p=probs)
        return configs[idx]

    def most_likely(self) -> tuple:
        """Return θ* = argmax_θ b(θ)."""
        return max(self._probs, key=self._probs.get)

    def update(self, likelihood: Dict[tuple, float]):
        """Bayesian update: b_new(θ) ∝ likelihood(θ) * b_old(θ)."""
        for cfg in self._configs:
            self._probs[cfg] *= likelihood.get(cfg, 1e-10)
        total = sum(self._probs.values())
        if total > 1e-30:
            for cfg in self._configs:
                self._probs[cfg] /= total
        else:
            # Collapsed — reset to uniform
            uniform = 1.0 / len(self._configs)
            for cfg in self._configs:
                self._probs[cfg] = uniform

    def marginals(self) -> Dict[int, float]:
        """P(agent i hidden) for each agent.

        Matches the convention used by naive inference: marginals
        represent P(hidden), not P(visible).
        """
        result = {}
        for i, aid in enumerate(self._agent_ids):
            p_visible = sum(prob for cfg, prob in self._probs.items() if cfg[i] == 1)
            result[aid] = 1.0 - p_visible
        return result


class MCTSNode:
    """Node in the per-belief MCTS tree.

    Stores per-action, per-θ Q values and shared per-action visit counts.
    Actions are scalar accelerations (float).
    """

    __slots__ = (
        'state', 'depth', 'parent', 'action', 'prev_action',
        'children', '_untried',
        'action_visits',   # Dict[float, int]
        'Q',               # Dict[float, Dict[theta, float]]
        'colliding',       # True if this state collides with an obstacle
        'jsd',             # JSD (policy divergence) at this node
        'resample_prob',   # sigmoid(gamma * jsd) — probability of resampling
    )

    def __init__(self,
                 state: np.ndarray,
                 depth: int,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[float] = None,
                 prev_action: Optional[float] = None,
                 colliding: bool = False):
        self.state = state          # (s, v)
        self.depth = depth
        self.parent = parent
        self.action = action        # scalar accel
        self.prev_action = prev_action
        self.children: Dict[float, 'MCTSNode'] = {}
        self._untried: Optional[List[float]] = None
        self.action_visits: Dict[float, int] = {}
        self.Q: Dict[float, Dict[tuple, float]] = {}
        self.colliding = colliding  # node is reachable but not expandable
        self.jsd = 0.0
        self.resample_prob = 0.0

    def is_terminal(self, horizon: int) -> bool:
        return self.depth >= horizon or self.colliding

    def total_visits(self) -> int:
        return sum(self.action_visits.values()) if self.action_visits else 0

    def untried_actions(self, all_actions: List[float],
                        planner: 'MCTSPlanner') -> List[float]:
        """Return actions not yet expanded, filtering by jerk limits."""
        if self._untried is None:
            self._untried = [
                a for a in all_actions
                if a not in self.children
                and planner._check_rate_limits(self.action, a)
            ]
        else:
            self._untried = [a for a in self._untried
                             if a not in self.children]
        return self._untried

    def is_fully_expanded(self, all_actions: List[float],
                          planner: 'MCTSPlanner') -> bool:
        return len(self.untried_actions(all_actions, planner)) == 0

    def best_action_ucb(self, theta: tuple, c: float) -> Optional[float]:
        """Select action maximising UCB1 with Q_θ values.

        UCB = Q_θ(s, a) + c * sqrt(ln(N_total) / N(s, a))
        """
        total_n = self.total_visits()
        if total_n == 0:
            return None

        log_n = math.log(total_n)
        best_action = None
        best_score = -float('inf')

        for action, child in self.children.items():
            if child is None:
                continue
            n = self.action_visits.get(action, 0)
            if n == 0:
                score = float('inf')
            else:
                q = self.Q.get(action, {}).get(theta, 0.0)
                explore = c * math.sqrt(log_n / n)
                score = q + explore
            score += random.uniform(0, 1e-6)  # tie-break
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


# ---------------------------------------------------------------------------
# MCTS Planner
# ---------------------------------------------------------------------------

class MCTSPlanner:
    """Longitudinal-only MCTS planner with per-belief Q values.

    Uses 1-D point-mass dynamics along the Frenet s-axis (d=0, phi=0).
    Actions are scalar accelerations.  Stores per-θ Q values at each
    node for Bayesian belief inference.

    Args:
        horizon: Planning steps (fine resolution).
        dt: Fine planning timestep (s).
        ego_length / ego_width / wheelbase: Vehicle geometry.
        collision_margin: Safety margin around obstacles (m).
        target_speed: Desired cruising speed (m/s).
        frenet: FrenetFrame for coordinate transforms.
        nlp_params: Dict overriding SecondStagePlanner.DEFAULTS.
        n_simulations: MCTS iterations.
        exploration_constant: UCB1 c parameter.
        n_accel_levels: Number of discrete acceleration levels.
        n_steer_levels: Ignored (kept for interface compatibility).
        max_trajectories: Trajectories to extract after search.
        gamma: Discount factor.
        collision_penalty: Penalty for longitudinal collision.
        clearance_threshold: Unused (kept for interface compatibility).
        beta: Rationality parameter for Boltzmann belief update.
        rollout_policy: Unused (kept for interface compatibility).
    """

    COARSENESS_FACTOR: int = 5

    def __init__(self,
                 horizon: int,
                 dt: float,
                 ego_length: float,
                 ego_width: float,
                 wheelbase: float,
                 collision_margin: float,
                 target_speed: float,
                 frenet,
                 nlp_params: dict,
                 n_simulations: int = 800,
                 exploration_constant: float = 10.0,
                 n_accel_levels: int = 19,
                 n_steer_levels: int = 21,
                 max_trajectories: int = 25,
                 gamma: float = 0.99,
                 collision_penalty: float = 100.0,
                 clearance_threshold: float = 4.0,
                 beta: float = 10.0,
                 rollout_policy: str = 'heuristic',
                 resample_gamma: Optional[float] = None,
                 resample_eta: Optional[float] = None,
                 kalman_eta: float = 0.15):

        self._coarseness = self.COARSENESS_FACTOR
        self._dt_fine = dt
        self._dt = dt * self._coarseness
        self._horizon = max(1, horizon // self._coarseness)
        self._horizon_fine = horizon

        self._ego_length = ego_length
        self._ego_width = ego_width
        self._half_L = ego_length / 2.0
        self._collision_margin = collision_margin
        self._target_speed = target_speed
        self._frenet = frenet
        self._gamma = gamma
        self._collision_penalty = collision_penalty
        self._beta = beta
        self._n_simulations = n_simulations
        self._exploration_constant = exploration_constant
        self._max_trajectories = max_trajectories

        # NLP parameters (bounds and weights)
        from igp2.beliefcontrol.second_stage import SecondStagePlanner
        self._params = dict(SecondStagePlanner.DEFAULTS)
        if nlp_params is not None:
            self._params.update(nlp_params)

        self._a_min = self._params['a_min']
        self._a_max = self._params['a_max']
        self._jerk_max = self._params['jerk_max']
        self._v_min = self._params['v_min']
        self._v_max = self._params['v_max']
        self._w_s = self._params['w_s']
        self._w_v = self._params['w_v']
        self._w_a = self._params['w_a']

        # Jerk limit per coarse step
        self._jerk_limit = self._jerk_max * self._dt

        # Discrete action set: 1-D acceleration grid
        self._actions = self._build_action_set(n_accel_levels)

        self._last_root: Optional[MCTSNode] = None

        # Information-guided resampling (None = disabled)
        self._resample_gamma = resample_gamma
        self._resample_eta = resample_eta
        self._resample_enabled = (resample_gamma is not None
                                  and resample_eta is not None)

        # Kalman-based resampling threshold
        self._kalman_eta = kalman_eta

        logger.info("MCTSPlanner (1-D): coarseness=%d, dt=%.3f/%.3f, "
                     "horizon=%d/%d, actions=%d, beta=%.2f, resample=%s",
                     self._coarseness, self._dt_fine, self._dt,
                     self._horizon, horizon, len(self._actions), beta,
                     f"gamma={resample_gamma},eta={resample_eta}"
                     if self._resample_enabled else "off")

    # ==================================================================
    # Action space (1-D: accelerations only)
    # ==================================================================

    def _build_action_set(self, n_accel: int) -> List[float]:
        accels = np.linspace(self._a_min, self._a_max, n_accel)
        if not any(abs(a) < 1e-12 for a in accels):
            accels = np.sort(np.append(accels, 0.0))
        return [round(float(a), 4) for a in accels]

    def _check_rate_limits(self, prev_action: Optional[float],
                           action: float) -> bool:
        if prev_action is None:
            prev_action = 0.0
        return abs(action - prev_action) <= self._jerk_limit + 1e-6

    def _nearest_action(self, continuous_action) -> float:
        """Map a continuous action to the nearest discrete acceleration.

        Accepts either a scalar, a tuple (a,), or a tuple (a, delta)
        for backward compatibility — only the first element is used.
        """
        if isinstance(continuous_action, (tuple, list)):
            a_cont = continuous_action[0]
        else:
            a_cont = float(continuous_action)
        best = self._actions[0]
        best_dist = float('inf')
        for act in self._actions:
            dist = abs(act - a_cont)
            if dist < best_dist:
                best_dist = dist
                best = act
        return best

    # ==================================================================
    # Information-guided resampling
    # ==================================================================

    def _compute_policy_divergence(self, node: MCTSNode,
                                   all_configs: List[tuple]) -> float:
        """Jensen-Shannon divergence of Boltzmann policies across configs.

        Measures how much the latent configurations disagree about the
        best action at this node.  Returns JSD >= 0.
        """
        if not node.Q or len(all_configs) <= 1:
            return 0.0

        actions = list(node.Q.keys())
        if not actions:
            return 0.0

        n_actions = len(actions)
        n_configs = len(all_configs)

        # Step 1: Boltzmann policy per configuration (normalised Q values)
        policies = {}
        for cfg in all_configs:
            raw_q = [node.Q[a].get(cfg, 0.0) for a in actions]
            max_q = max(raw_q)
            min_q = min(raw_q)
            q_range = max_q - min_q
            if q_range < 1e-10:
                policies[cfg] = [1.0 / n_actions] * n_actions
                continue
            logits = [self._beta * (q - max_q) / q_range for q in raw_q]
            exps = [math.exp(l) for l in logits]
            total = sum(exps)
            if total < 1e-30:
                policies[cfg] = [1.0 / n_actions] * n_actions
            else:
                policies[cfg] = [e / total for e in exps]

        # Step 2: mixture distribution (uniform weights over configs)
        mixture = [0.0] * n_actions
        for i in range(n_actions):
            for cfg in all_configs:
                mixture[i] += policies[cfg][i] / n_configs

        # Step 3: KL(pi_theta || mixture) for each config
        kl_values = []
        for cfg in all_configs:
            kl = 0.0
            for i in range(n_actions):
                if policies[cfg][i] > 1e-10:
                    kl += policies[cfg][i] * math.log(
                        policies[cfg][i] / max(mixture[i], 1e-30))
            kl_values.append(kl)

        # Step 4: JSD = average of KL divergences
        return sum(kl_values) / n_configs

    def _should_resample(self, info_value: float) -> bool:
        """Sigmoid decision: resample if p > eta."""
        p = 1.0 / (1.0 + math.exp(-self._resample_gamma * info_value))
        return p > self._resample_eta

    # ==================================================================
    # 1-D longitudinal dynamics
    # ==================================================================

    def _simulate_step(self, state: np.ndarray,
                       action: float,
                       road_left_k: float,
                       road_right_k: float,
                       obstacles: Optional[list] = None,
                       step_k: int = 0,
                       ) -> Tuple[np.ndarray, bool, bool, str]:
        """Forward one step of 1-D longitudinal dynamics.

        State is (s, v).  Propagates dynamics then checks the new state
        against all obstacles.  If the new state collides, the child
        node is still created but marked as colliding (terminal), so
        it receives the collision penalty but is never expanded further.

        Returns (new_state, feasible, colliding, reject_reason).
        """
        s, v = state
        dt = self._dt
        s_new = s + v * dt
        v_new = max(self._v_min, min(self._v_max, v + action * dt))

        new_state = np.array([s_new, v_new])

        # Check all obstacles against the new state (at the child's time)
        colliding = False
        if obstacles:
            fine_k = (step_k + 1) * self._coarseness
            for obs in obstacles:
                if self._check_longitudinal_collision(s_new, obs, fine_k):
                    colliding = True
                    break

        return new_state, True, colliding, ''

    # ==================================================================
    # Evaluation (per-θ cost function) — 1-D longitudinal
    # ==================================================================

    def _step_reward_base(self, state: np.ndarray,
                          action: float,
                          step_k: int,
                          s0: float) -> float:
        """Tracking + control cost (identical for all θ).

        Only longitudinal terms: s-tracking, v-tracking, acceleration.
        Evaluated once at the coarse timestep.
        """
        s, v = state

        s_max = self._frenet.total_length if self._frenet else 1e6
        ref_s = min(s0 + self._target_speed * step_k * self._dt, s_max)

        tracking = -(self._w_s * (s - ref_s) ** 2
                     + self._w_v * (v - self._target_speed) ** 2)

        control = -(self._w_a * action ** 2)
        return tracking + control

    def _collision_cost(self, state: np.ndarray, obstacles: list,
                        step_k: int,
                        visible_aids: Optional[Set[int]] = None) -> float:
        """1-D collision penalty (depends on θ via visible_aids).

        Static objects (aid < 0) are always penalised.
        Dynamic agents are only penalised if their aid is in visible_aids.
        """
        s = state[0]
        fine_k = step_k * self._coarseness
        collision = 0.0
        for obs in obstacles:
            aid = obs['agent_id']
            if aid >= 0:
                if visible_aids is not None and aid not in visible_aids:
                    continue
            if self._check_longitudinal_collision(s, obs, fine_k):
                collision = min(collision, -self._collision_penalty)
        return collision

    def _step_reward(self, state: np.ndarray,
                     action: float,
                     obstacles: list,
                     step_k: int,
                     s0: float,
                     visible_aids: Optional[Set[int]] = None) -> float:
        """Single-step reward under a specific visibility configuration."""
        base = self._step_reward_base(state, action, step_k, s0)
        collision = self._collision_cost(state, obstacles, step_k, visible_aids)
        return base + collision

    def _check_longitudinal_collision(self, s_ego: float, obs: dict,
                                      fine_k: int) -> bool:
        """Collision check: longitudinal gap along s + lateral check on d.

        Only triggers if the obstacle is close in both s (longitudinal)
        and d (lateral).  This prevents false collisions when a vehicle
        has crossed the ego's reference path and moved away laterally.
        """
        k = min(fine_k, len(obs['s']) - 1)
        s_obs = float(obs['s'][k])
        s_gap = self._half_L + obs['length'] / 2.0 + self._collision_margin
        if abs(s_ego - s_obs) >= s_gap:
            return False
        # Lateral check: obstacle must be within lane-width proximity
        d_obs = float(obs['d'][k])
        d_gap = self._ego_width / 2.0 + obs['width'] / 2.0 + self._collision_margin
        return abs(d_obs) < d_gap

    # ==================================================================
    # Core MCTS loop (per-belief)
    # ==================================================================

    def search(self, frenet_state: np.ndarray,
               road_left: np.ndarray, road_right: np.ndarray,
               obstacles: list,
               prev_action=None,
               belief: Optional[BeliefState] = None,
               human_action=None,
               kalman=None,
               ) -> Tuple[List[MCTSTrajectory], Optional[BeliefState]]:
        """Run per-belief MCTS search (1-D longitudinal).

        Args:
            frenet_state: Current ego state [s, d, phi, v] (d, phi ignored).
            road_left / road_right: Road boundary arrays (accepted but ignored).
            obstacles: Obstacle list (with agent_id, s, d, etc.).
            prev_action: Previous acceleration (scalar or tuple — first
                element used).  For rate-limit continuity.
            belief: Current belief state (created if None).
            human_action: Observed human action for belief update.
                When ``kalman`` is provided, pass None here — the
                outer loop handles the Kalman observation update.
            kalman: Optional KalmanAwareness instance for Kalman-driven
                resampling during simulation.  Each simulation copies
                this state and propagates it forward.

        Returns:
            (trajectories, updated_belief)
        """
        # Extract (s, v) from the full 4-element Frenet state
        s0 = float(frenet_state[0])
        v0 = float(frenet_state[3]) if len(frenet_state) > 3 else float(frenet_state[1])
        state_1d = np.array([s0, v0])

        # Normalise prev_action to scalar
        if isinstance(prev_action, (tuple, list)):
            prev_action = float(prev_action[0])

        root = MCTSNode(state=state_1d, depth=0, action=prev_action)

        # Debug: show root state and action filtering
        nearest_grid = self._nearest_action(prev_action) if prev_action is not None else None
        root_untried = [
            a for a in self._actions
            if self._check_rate_limits(prev_action, a)
        ]
        root_v_after = [max(self._v_min, min(self._v_max, v0 + a * self._dt))
                        for a in root_untried]
        logger.info("MCTS root: s=%.2f v=%.2f prev_action=%s (nearest_grid=%s) "
                     "jerk_limit=%.4f dt_coarse=%.3f | %d viable_actions=%s "
                     "→ v_after=%s | all_actions=[%.2f..%.2f] (%d total)",
                     s0, v0, prev_action, nearest_grid,
                     self._jerk_limit, self._dt,
                     len(root_untried),
                     [round(a, 3) for a in root_untried],
                     [round(v, 3) for v in root_v_after],
                     self._actions[0], self._actions[-1], len(self._actions))

        # Build belief state if not provided
        if belief is None:
            dynamic_aids = sorted({obs['agent_id'] for obs in obstacles
                                   if obs['agent_id'] >= 0})
            belief = BeliefState(dynamic_aids)

        all_configs = belief.configs

        # Pre-compute visible aid sets for each θ
        theta_visible: Dict[tuple, Set[int]] = {
            cfg: belief.visible_aids(cfg) for cfg in all_configs
        }

        # Debug counters
        n_expanded = 0
        n_terminal = 0
        n_no_untried = 0
        n_all_infeasible = 0
        self._reject_counts = {}

        import time as _time
        _t_select = 0.0
        _t_expand = 0.0
        _t_backup = 0.0

        n_resamples = 0  # debug counter
        n_kalman_resamples = 0  # Kalman resampling counter

        for _ in range(self._n_simulations):
            # 1. Sample θ from belief for this simulation
            theta_sampled = belief.sample()

            # Copy Kalman state for this simulation (if provided)
            kalman_sim = kalman.copy() if kalman is not None else None

            # 2. Selection: traverse using Q_θ for UCB
            _t0 = _time.perf_counter()
            path: List[Tuple[MCTSNode, Tuple[float, float]]] = []
            node = root

            while not node.is_terminal(self._horizon):
                if not node.is_fully_expanded(self._actions, self):
                    break

                # Information-guided resampling (JSD-based, when no Kalman)
                if kalman_sim is None and self._resample_enabled:
                    jsd = self._compute_policy_divergence(node, all_configs)
                    if self._should_resample(jsd):
                        theta_sampled = random.choice(all_configs)
                        n_resamples += 1

                action = node.best_action_ucb(theta_sampled,
                                              self._exploration_constant)
                if action is None:
                    break
                path.append((node, action))
                node = node.children[action]

                # Kalman-based resampling (after moving to child)
                if kalman_sim is not None:
                    fine_k = node.depth * self._coarseness
                    # Convert node Frenet state (s, v) to world position + heading
                    _w = self._frenet.frenet_to_world(
                        float(node.state[0]), 0.0,
                        heading=0.0)
                    _ego_xy = np.array([_w['x'], _w['y']], dtype=float)
                    _ego_heading = _w['heading']
                    kalman_sim.predict_from_obstacles(
                        _ego_xy, _ego_heading,
                        obstacles, fine_k)
                    b_local = kalman_sim.compute_b_theta(all_configs)
                    if b_local.get(theta_sampled, 0.0) < self._kalman_eta:
                        probs = [b_local.get(cfg, 1e-10) for cfg in all_configs]
                        total = sum(probs)
                        probs = [p / total for p in probs]
                        idx = np.random.choice(len(all_configs), p=probs)
                        theta_sampled = all_configs[idx]
                        n_kalman_resamples += 1
            _t_select += _time.perf_counter() - _t0

            # 3. Expansion
            _t0 = _time.perf_counter()
            if not node.is_terminal(self._horizon):
                child, action = self._expand(node, road_left, road_right,
                                             obstacles, s0)
                if child is not None:
                    n_expanded += 1
                    path.append((node, action))
                    # Leaf value estimate: 0 for terminal nodes (no future),
                    # step reward as crude V(leaf) for non-terminal nodes.
                    leaf_returns = {}
                    if child.colliding:
                        # Terminal — no future return
                        leaf_returns = {cfg: 0.0 for cfg in all_configs}
                    else:
                        for cfg in all_configs:
                            leaf_returns[cfg] = self._step_reward(
                                child.state, action, obstacles, child.depth, s0,
                                visible_aids=theta_visible[cfg])
                else:
                    untried = node.untried_actions(self._actions, self)
                    if not untried:
                        n_no_untried += 1
                    else:
                        n_all_infeasible += 1
                    leaf_returns = {cfg: 0.0 for cfg in all_configs}
            else:
                n_terminal += 1
                leaf_returns = {cfg: 0.0 for cfg in all_configs}

            _t_expand += _time.perf_counter() - _t0

            # 4. Backup: propagate per-θ returns up the path
            _t0 = _time.perf_counter()
            returns = leaf_returns

            for parent_node, act in reversed(path):
                child_node = parent_node.children[act]

                # Per-θ step reward for this transition
                step_rewards = {}
                for cfg in all_configs:
                    step_rewards[cfg] = self._step_reward(
                        child_node.state, act, obstacles,
                        child_node.depth, s0,
                        visible_aids=theta_visible[cfg])

                # Update visit count (shared across θ)
                parent_node.action_visits[act] = \
                    parent_node.action_visits.get(act, 0) + 1
                n = parent_node.action_visits[act]

                # Update Q for all θ (incremental mean)
                if act not in parent_node.Q:
                    parent_node.Q[act] = {}

                new_returns = {}
                for cfg in all_configs:
                    total_return = step_rewards[cfg] + self._gamma * returns[cfg]
                    old_q = parent_node.Q[act].get(cfg, 0.0)
                    parent_node.Q[act][cfg] = old_q + (total_return - old_q) / n
                    new_returns[cfg] = total_return

                returns = new_returns
            _t_backup += _time.perf_counter() - _t0

        self._last_root = root

        # Compute JSD and resample probability for every node (for plotting)
        gamma_r = self._resample_gamma if self._resample_gamma is not None else 5.0
        bfs_q = [root]
        while bfs_q:
            nd = bfs_q.pop(0)
            nd.jsd = self._compute_policy_divergence(nd, all_configs)
            nd.resample_prob = 1.0 / (1.0 + math.exp(-gamma_r * nd.jsd))
            for ch in nd.children.values():
                if ch is not None:
                    bfs_q.append(ch)

        _t0 = _time.perf_counter()
        # ----- Belief update from human's observed action -----
        if human_action is not None and root.Q:
            belief = self._update_belief(root, belief, human_action)

        # Debug logging
        reject_str = ", ".join(
            f"{r}={c}" for r, c in sorted(self._reject_counts.items()))
        n_nodes, depth_counts, visit_buckets = self._tree_stats(root)
        resample_str = ""
        if self._resample_enabled:
            resample_str = f" | resamples: {n_resamples}"
        if kalman is not None:
            resample_str += f" | kalman_resamples: {n_kalman_resamples}"
        logger.info(
            "MCTS debug: %d expanded, %d terminal, %d fully-expanded, "
            "%d all-infeasible | %d tree nodes | rejects: {%s}%s | "
            "road_left=[%.1f..%.1f] road_right=[%.1f..%.1f]",
            n_expanded, n_terminal, n_no_untried, n_all_infeasible,
            n_nodes, reject_str, resample_str,
            float(road_left.min()), float(road_left.max()),
            float(road_right.min()), float(road_right.max()))

        # Tree shape: nodes per depth
        depth_str = "  ".join(f"d{d}={c}" for d, c in sorted(depth_counts.items()))
        logger.info("MCTS tree shape: %s", depth_str)

        # Visit distribution (bucketed)
        bucket_str = "  ".join(f"{k}:{v}" for k, v in visit_buckets.items() if v > 0)
        logger.info("MCTS visit buckets: %s", bucket_str)

        # Log belief state
        if belief.agent_ids:
            marg = belief.marginals()
            marg_str = ", ".join(f"{aid}={p:.3f}" for aid, p in marg.items())
            logger.info("Belief marginals: {%s}", marg_str)

        _t_belief_update = _time.perf_counter() - _t0

        # ----- Trajectory extraction under θ* -----
        _t0 = _time.perf_counter()
        theta_star = belief.most_likely()
        trajectories = self._extract_trajectories(
            root, self._max_trajectories, road_left, road_right,
            obstacles, s0, theta_star, theta_visible)
        trajectories.sort(key=lambda t: -t.mcts_reward)
        _t_extract = _time.perf_counter() - _t0

        logger.info("MCTS search: %d simulations, %d trajectories",
                     self._n_simulations, len(trajectories))
        logger.info("MCTS timing: select=%.3fs  expand=%.3fs  backup=%.3fs  "
                     "belief_update=%.3fs  extract=%.3fs  total=%.3fs",
                     _t_select, _t_expand, _t_backup, _t_belief_update,
                     _t_extract,
                     _t_select + _t_expand + _t_backup + _t_belief_update + _t_extract)

        return trajectories, belief

    def _count_nodes(self, root: MCTSNode) -> int:
        count = 0
        queue = [root]
        while queue:
            nd = queue.pop(0)
            count += 1
            for child in nd.children.values():
                if child is not None:
                    queue.append(child)
        return count

    def _tree_stats(self, root: MCTSNode):
        """Collect tree statistics: node count, depth distribution, visit buckets.

        Returns (n_nodes, depth_counts, visit_buckets) where:
          depth_counts: {depth: node_count}
          visit_buckets: {bucket_label: count}
        """
        depth_counts: Dict[int, int] = {}
        all_visits = []

        queue = [root]
        n_nodes = 0
        while queue:
            nd = queue.pop(0)
            n_nodes += 1
            depth_counts[nd.depth] = depth_counts.get(nd.depth, 0) + 1
            all_visits.append(nd.total_visits())
            for child in nd.children.values():
                if child is not None:
                    queue.append(child)

        # Bucket visit counts
        buckets = {'0': 0, '1': 0, '2-5': 0, '6-20': 0,
                   '21-50': 0, '51-200': 0, '201+': 0}
        for v in all_visits:
            if v == 0:
                buckets['0'] += 1
            elif v == 1:
                buckets['1'] += 1
            elif v <= 5:
                buckets['2-5'] += 1
            elif v <= 20:
                buckets['6-20'] += 1
            elif v <= 50:
                buckets['21-50'] += 1
            elif v <= 200:
                buckets['51-200'] += 1
            else:
                buckets['201+'] += 1

        return n_nodes, depth_counts, buckets

    # ----- 1. Expansion -----

    def _expand(self, node: MCTSNode,
                road_left, road_right, obstacles, s0,
                ) -> Tuple[Optional[MCTSNode], Optional[Tuple[float, float]]]:
        """Expand one untried action.  Returns (child, action) or (None, None)."""
        untried = node.untried_actions(self._actions, self)
        if not untried:
            return None, None

        random.shuffle(untried)
        k = node.depth
        fine_k = (k + 1) * self._coarseness
        rl = road_left[min(fine_k, len(road_left) - 1)]
        rr = road_right[min(fine_k, len(road_right) - 1)]

        for action in untried:
            new_state, feasible, colliding, reason = self._simulate_step(
                node.state, action, rl, rr, obstacles, k)
            if not feasible:
                node.children[action] = None
                self._reject_counts[reason] = \
                    self._reject_counts.get(reason, 0) + 1
                continue

            child = MCTSNode(state=new_state, depth=k + 1,
                             parent=node, action=action,
                             prev_action=node.action,
                             colliding=colliding)
            node.children[action] = child
            node._untried = None
            return child, action

        node._untried = None
        return None, None

    # ----- 2. Belief update -----

    def _update_belief(self, root: MCTSNode, belief: BeliefState,
                       human_action) -> BeliefState:
        """Boltzmann belief update using root Q values and observed action.

        P(u_H | s_0, θ) = exp(β * Q_θ(s_0, u_H)) / Σ_a' exp(β * Q_θ(s_0, a'))
        b_new(θ) ∝ P(u_H | s_0, θ) * b_old(θ)
        """
        # Map continuous human action to nearest discrete acceleration
        u_h = self._nearest_action(human_action)

        likelihood = {}
        for cfg in belief.configs:
            # Collect Q_θ(s_0, a) for all visited actions
            q_vals = {}
            for act, q_dict in root.Q.items():
                if cfg in q_dict:
                    q_vals[act] = q_dict[cfg]

            if not q_vals:
                likelihood[cfg] = 1.0 / len(self._actions)
                continue

            # Boltzmann likelihood
            q_human = q_vals.get(u_h, None)
            if q_human is None:
                # Human action wasn't explored — use minimum Q as fallback
                q_human = min(q_vals.values()) - 1.0

            # Normalise Q values by range for scale-invariant β
            all_q = list(q_vals.values())
            if u_h not in q_vals:
                all_q.append(q_human)
            max_q = max(all_q)
            min_q = min(all_q)
            q_range = max_q - min_q
            if q_range < 1e-10:
                likelihood[cfg] = 1.0 / len(all_q)
                continue

            # Numerically stable softmax on normalised Q values
            numerator = math.exp(self._beta * (q_human - max_q) / q_range)
            denominator = sum(math.exp(self._beta * (q - max_q) / q_range)
                              for q in q_vals.values())
            if u_h not in q_vals:
                denominator += numerator

            likelihood[cfg] = numerator / max(denominator, 1e-30)

        belief.update(likelihood)
        return belief

    # ==================================================================
    # Trajectory extraction
    # ==================================================================

    def _extract_trajectories(self, root, k, road_left, road_right,
                              obstacles, s0, theta_star, theta_visible):
        """Extract top-k trajectories via greedy traversal under θ*."""
        results: List[MCTSTrajectory] = []

        # Primary trajectory: greedy under θ*
        primary = self._greedy_trajectory(
            root, road_left, road_right, obstacles, s0,
            theta_star, theta_visible)
        if primary is not None:
            results.append(primary)

        # Additional trajectories: best-first DFS on visit count
        stack: List[List[MCTSNode]] = [[root]]
        while stack and len(results) < k:
            path = stack.pop()
            node = path[-1]
            valid = {a: c for a, c in node.children.items() if c is not None}

            if node.is_terminal(self._horizon) or not valid:
                traj = self._path_to_trajectory(
                    path, road_left, road_right, obstacles, s0,
                    theta_star, theta_visible)
                if traj is not None:
                    results.append(traj)
                continue

            for child in sorted(valid.values(),
                                key=lambda c: c.parent.action_visits.get(
                                    c.action, 0) if c.parent else 0):
                n = node.action_visits.get(child.action, 0)
                if n > 0:
                    stack.append(path + [child])

        return results

    def _greedy_trajectory(self, root, road_left, road_right,
                           obstacles, s0, theta_star, theta_visible):
        """Greedy traversal: at each node, pick argmax_a Q_θ*(s, a)."""
        path = [root]
        node = root
        while not node.is_terminal(self._horizon):
            if not node.Q:
                break
            # Pick action with highest Q_θ*
            best_action = None
            best_q = -float('inf')
            for act, q_dict in node.Q.items():
                q = q_dict.get(theta_star, -float('inf'))
                if q > best_q:
                    best_q = q
                    best_action = act
            if best_action is None or best_action not in node.children:
                break
            child = node.children[best_action]
            if child is None:
                break
            path.append(child)
            node = child

        return self._path_to_trajectory(
            path, road_left, road_right, obstacles, s0,
            theta_star, theta_visible)

    def _path_to_trajectory(self, path, road_left, road_right,
                            obstacles, s0, theta_star, theta_visible):
        """Convert tree path to MCTSTrajectory, padding to horizon.

        Internal states are (s, v).  Output is zero-padded to (s, 0, 0, v)
        and (a, 0) for downstream compatibility.
        """
        H = self._horizon
        states_1d = [path[0].state.copy()]  # (s, v)
        controls_1d = []  # scalar accelerations
        reward = 0.0
        discount = 1.0
        vis = theta_visible.get(theta_star)

        for i in range(1, len(path)):
            n = path[i]
            states_1d.append(n.state.copy())
            controls_1d.append(n.action)
            reward += discount * self._step_reward(
                n.state, n.action, obstacles, n.depth, s0,
                visible_aids=vis)
            discount *= self._gamma

        # Pad to horizon with heuristic policy
        depth = len(path) - 1
        state = states_1d[-1].copy()
        prev_action = controls_1d[-1] if controls_1d else None

        while depth < H:
            k_next = depth + 1
            action = self._heuristic_action(state, prev_action)
            new_state, feasible, _, _ = self._simulate_step(
                state, action, 0.0, 0.0)
            if not feasible:
                # In 1-D this shouldn't happen, but pad with zero accel
                while depth < H:
                    reward += discount * self._step_reward(
                        state, 0.0, obstacles, depth + 1, s0,
                        visible_aids=vis)
                    discount *= self._gamma
                    states_1d.append(state.copy())
                    controls_1d.append(0.0)
                    depth += 1
                break

            reward += discount * self._step_reward(
                new_state, action, obstacles, k_next, s0,
                visible_aids=vis)
            discount *= self._gamma
            states_1d.append(new_state)
            controls_1d.append(action)
            state = new_state
            prev_action = action
            depth += 1

        # Pad (s, v) → (s, 0, 0, v) and accel → (accel, 0) for compat
        padded_states = np.array([
            [sv[0], 0.0, 0.0, sv[1]] for sv in states_1d
        ])
        padded_controls = np.array([
            [a, 0.0] for a in controls_1d
        ]) if controls_1d else np.empty((0, 2))

        return MCTSTrajectory(states=padded_states, controls=padded_controls,
                              mcts_reward=reward)

    # ==================================================================
    # Public trajectory extraction
    # ==================================================================

    def extract_trajectory_for_config(
            self, theta: tuple, belief: 'BeliefState',
            road_left: np.ndarray, road_right: np.ndarray,
            obstacles: list) -> Optional[MCTSTrajectory]:
        """Extract the greedy trajectory under a specific θ config.

        Uses the last search tree (``_last_root``).  Returns None if no
        tree is available or the config has no Q data at the root.

        Args:
            theta: Belief configuration tuple, e.g. (1, 0).
            belief: BeliefState (for visible_aids mapping).
            road_left / road_right: Road boundary arrays.
            obstacles: Obstacle list.

        Returns:
            MCTSTrajectory or None.
        """
        root = self._last_root
        if root is None or not root.Q:
            return None

        s0 = float(root.state[0])
        theta_visible = {
            cfg: belief.visible_aids(cfg) for cfg in belief.configs
        }
        return self._greedy_trajectory(
            root, road_left, road_right, obstacles, s0,
            theta, theta_visible)

    def extract_safe_trajectory(
            self, theta_star: tuple, theta_R: tuple,
            belief: 'BeliefState',
            road_left: np.ndarray, road_right: np.ndarray,
            obstacles: list,
            gamma: float = 0.8) -> Optional[MCTSTrajectory]:
        """Safety-filtered trajectory extraction (QCBF).

        At each node, follow the human's preferred action (argmax Q_{θ*})
        unless its relative regret under the true state θ_R exceeds
        (1 − γ), in which case override with argmax Q_{θ_R}.

        Args:
            theta_star: MAP belief config (human's estimated belief).
            theta_R: True config (all agents visible).
            belief: BeliefState for visible_aids mapping.
            road_left / road_right: Road boundary arrays.
            obstacles: Obstacle list.
            gamma: Safety strictness ∈ [0, 1].  Higher = stricter.

        Returns:
            MCTSTrajectory with per-step intervention flags, or None.
        """
        root = self._last_root
        if root is None or not root.Q:
            return None

        threshold = 1.0 - gamma

        path = [root]
        intervention_flags = []
        node = root

        while not node.is_terminal(self._horizon):
            if not node.Q:
                break

            # Human's preferred action: argmax_a Q_{θ*}(s, a)
            a_H = None
            best_q_star = -float('inf')
            for act, q_dict in node.Q.items():
                q = q_dict.get(theta_star, -float('inf'))
                if q > best_q_star:
                    best_q_star = q
                    a_H = act

            if a_H is None:
                break

            # Q values under true state θ_R
            q_R = {}
            for act, q_dict in node.Q.items():
                q_R[act] = q_dict.get(theta_R, -float('inf'))

            q_max = max(q_R.values())
            q_min = min(q_R.values())

            # Relative regret of human's action under true state
            if q_max == q_min:
                relative_regret = 0.0
            else:
                relative_regret = (q_max - q_R.get(a_H, -float('inf'))) / (q_max - q_min)

            # Safety filter
            if relative_regret <= threshold:
                chosen_action = a_H
                intervention = False
            else:
                chosen_action = max(q_R, key=q_R.get)
                intervention = True

            if chosen_action not in node.children or node.children[chosen_action] is None:
                break

            child = node.children[chosen_action]
            path.append(child)
            intervention_flags.append(intervention)
            node = child

        s0 = float(root.state[0])
        theta_visible = {
            cfg: belief.visible_aids(cfg) for cfg in belief.configs
        }
        traj = self._path_to_trajectory(
            path, road_left, road_right, obstacles, s0,
            theta_R, theta_visible)

        if traj is not None:
            # Pad intervention flags to match trajectory length (tree + padding)
            n_tree = len(path) - 1
            n_total = len(traj.controls)
            padded_flags = list(intervention_flags)
            # Padding steps beyond the tree: no intervention (heuristic policy)
            padded_flags.extend([False] * (n_total - n_tree))
            traj.interventions = padded_flags

        return traj

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _heuristic_action(self, state, prev_action):
        """Simple proportional acceleration towards target speed."""
        v = state[1]
        a_des = np.clip((self._target_speed - v) / self._dt,
                        self._a_min, self._a_max)
        if prev_action is not None:
            a_des = np.clip(a_des, prev_action - self._jerk_limit,
                            prev_action + self._jerk_limit)
        return float(np.clip(a_des, self._a_min, self._a_max))
