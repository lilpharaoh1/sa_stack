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
        """P(agent i visible) for each agent."""
        result = {}
        for i, aid in enumerate(self._agent_ids):
            p = sum(prob for cfg, prob in self._probs.items() if cfg[i] == 1)
            result[aid] = p
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
                 n_accel_levels: int = 13,
                 n_steer_levels: int = 21,
                 max_trajectories: int = 25,
                 gamma: float = 0.99,
                 collision_penalty: float = 1000000.0,
                 clearance_threshold: float = 4.0,
                 beta: float = 1.0,
                 rollout_policy: str = 'heuristic'):

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

        logger.info("MCTSPlanner (1-D): coarseness=%d, dt=%.3f/%.3f, "
                     "horizon=%d/%d, actions=%d, beta=%.2f",
                     self._coarseness, self._dt_fine, self._dt,
                     self._horizon, horizon, len(self._actions), beta)

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
            return True
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

        # Check all obstacles against the new state
        colliding = False
        if obstacles:
            fine_k = step_k * self._coarseness
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
        """1-D collision: check if ego s overlaps obstacle s (+ margins)."""
        k = min(fine_k, len(obs['s']) - 1)
        s_obs = float(obs['s'][k])
        gap = self._half_L + obs['length'] / 2.0 + self._collision_margin
        return abs(s_ego - s_obs) < gap

    # ==================================================================
    # Core MCTS loop (per-belief)
    # ==================================================================

    def search(self, frenet_state: np.ndarray,
               road_left: np.ndarray, road_right: np.ndarray,
               obstacles: list,
               prev_action=None,
               belief: Optional[BeliefState] = None,
               human_action=None,
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

        for _ in range(self._n_simulations):
            # 1. Sample θ for this simulation
            theta_sampled = belief.sample()

            # 2. Selection: traverse using Q_θ for UCB
            _t0 = _time.perf_counter()
            path: List[Tuple[MCTSNode, Tuple[float, float]]] = []
            node = root

            while not node.is_terminal(self._horizon):
                if not node.is_fully_expanded(self._actions, self):
                    break
                action = node.best_action_ucb(theta_sampled,
                                              self._exploration_constant)
                if action is None:
                    break
                path.append((node, action))
                node = node.children[action]
            _t_select += _time.perf_counter() - _t0

            # 3. Expansion
            _t0 = _time.perf_counter()
            if not node.is_terminal(self._horizon):
                child, action = self._expand(node, road_left, road_right,
                                             obstacles, s0)
                if child is not None:
                    n_expanded += 1
                    path.append((node, action))
                    # Leaf evaluation: per-θ step reward
                    leaf_returns = {}
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

        _t0 = _time.perf_counter()
        # ----- Belief update from human's observed action -----
        if human_action is not None and root.Q:
            belief = self._update_belief(root, belief, human_action)

        # Debug logging
        reject_str = ", ".join(
            f"{r}={c}" for r, c in sorted(self._reject_counts.items()))
        n_nodes = self._count_nodes(root)
        logger.info(
            "MCTS debug: %d expanded, %d terminal, %d fully-expanded, "
            "%d all-infeasible | %d tree nodes | rejects: {%s} | "
            "road_left=[%.1f..%.1f] road_right=[%.1f..%.1f]",
            n_expanded, n_terminal, n_no_untried, n_all_infeasible,
            n_nodes, reject_str,
            float(road_left.min()), float(road_left.max()),
            float(road_right.min()), float(road_right.max()))

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

            # Numerically stable softmax
            max_q = max(max(q_vals.values()), q_human)
            numerator = math.exp(self._beta * (q_human - max_q))
            denominator = sum(math.exp(self._beta * (q - max_q))
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
