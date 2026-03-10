"""MCTS trajectory planner in Frenet coordinates with bicycle dynamics.

Generates diverse coarse trajectories via Monte Carlo Tree Search, then
derives implied belief configurations from per-agent reward decomposition.
Each trajectory is linked to a belief about which agents the driver can
see, based on which agents' collision/clearance terms were active.

Adapted from TreeIRL (Tomov et al., 2025) for belief-driven assistive
driving with optimisation warm-starting.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryBeliefPair:
    """A coarse trajectory with its implied belief configuration."""

    states: np.ndarray                  # (H+1, 4) [s, d, phi, v]
    controls: np.ndarray                # (H, 2)   [a, delta]
    implied_belief: Dict[int, bool]     # {agent_id: visible}
    mcts_reward: float                  # cumulative discounted reward
    obstacle_rewards: Dict[int, float]  # per-agent cumulative reward


class MCTSNode:
    """Node in the MCTS search tree.

    Each node stores a Frenet state ``[s, d, phi, v]`` at a particular
    depth (timestep) and tracks visit statistics for UCB1 selection.
    Per-obstacle reward accumulators support belief derivation.
    """

    __slots__ = (
        'state', 'depth', 'parent', 'action', 'prev_action',
        'children', 'visit_count', 'total_reward',
        'obstacle_reward_totals', '_untried',
    )

    def __init__(self,
                 state: np.ndarray,
                 depth: int,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[Tuple[float, float]] = None,
                 prev_action: Optional[Tuple[float, float]] = None):
        self.state = state              # [s, d, phi, v]
        self.depth = depth
        self.parent = parent
        self.action = action            # (a, delta) that led here
        self.prev_action = prev_action  # parent's action (for rate limits)
        self.children: Dict[Tuple[float, float], 'MCTSNode'] = {}
        self.visit_count: int = 0
        self.total_reward: float = 0.0
        self.obstacle_reward_totals: Dict[int, float] = {}
        self._untried: Optional[List[Tuple[float, float]]] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def is_terminal(self, horizon: int) -> bool:
        return self.depth >= horizon

    def untried_actions(self, all_actions: List[Tuple[float, float]],
                        planner: 'MCTSPlanner') -> List[Tuple[float, float]]:
        """Return actions not yet expanded, filtering by rate limits.

        Actions mapped to ``None`` in ``self.children`` were tried but
        found infeasible — they are excluded.
        """
        if self._untried is None:
            valid = []
            for act in all_actions:
                if act not in self.children:
                    if planner._check_rate_limits(self.action, act):
                        valid.append(act)
            self._untried = valid
        else:
            # Remove any that have since been expanded (including None sentinels)
            self._untried = [a for a in self._untried
                             if a not in self.children]
        return self._untried

    def is_fully_expanded(self, all_actions: List[Tuple[float, float]],
                          planner: 'MCTSPlanner') -> bool:
        return len(self.untried_actions(all_actions, planner)) == 0

    def best_child_ucb(self, c: float) -> Optional['MCTSNode']:
        """Select child with highest UCB1 score.

        Skips ``None`` sentinels (infeasible actions).
        """
        log_n = math.log(self.visit_count + 1)
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            if child is None:
                continue
            if child.visit_count == 0:
                score = float('inf')
            else:
                exploit = child.total_reward / child.visit_count
                explore = c * math.sqrt(log_n / child.visit_count)
                score = exploit + explore
            # Tie-breaking
            score += random.uniform(0, 1e-6)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def best_child_visit(self) -> Optional['MCTSNode']:
        """Select child with highest visit count (for extraction).

        Skips ``None`` sentinels (infeasible actions).
        """
        valid = [c for c in self.children.values() if c is not None]
        if not valid:
            return None
        return max(valid, key=lambda c: c.visit_count)


# ---------------------------------------------------------------------------
# MCTS Planner
# ---------------------------------------------------------------------------

class MCTSPlanner:
    """MCTS trajectory planner in Frenet coordinates with bicycle dynamics.

    Uses the same bicycle kinematic model and constraints as
    :class:`SecondStagePlanner` but evaluates trajectories via a reward
    function rather than solving an NLP.  The reward decomposes per-agent
    collision/clearance contributions to derive implied belief
    configurations.

    Args:
        horizon: Number of planning steps.
        dt: Planning timestep (s).
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
        wheelbase: Bicycle model wheelbase (m).
        collision_margin: Safety margin around obstacles (m).
        target_speed: Desired cruising speed (m/s).
        frenet: FrenetFrame for coordinate transforms.
        nlp_params: Dict of NLP parameters (bounds and weights).
        n_simulations: Number of MCTS iterations.
        exploration_constant: UCB1 exploration parameter c.
        n_accel_levels: Discretisation levels for acceleration.
        n_steer_levels: Discretisation levels for steering angle.
        max_trajectories: Maximum distinct trajectories to extract (k).
        gamma: Discount factor for cumulative reward.
        collision_penalty: Penalty scale for ellipse violation.
        clearance_threshold: Ellipse g-value below which clearance
            reward is applied (g < threshold means "close").
        rollout_policy: Default policy for simulation phase
            (``'heuristic'`` or ``'random'``).
        activity_threshold: Minimum |cumulative obstacle reward| to
            consider an agent "visible" in the implied belief.
    """

    # How much coarser the MCTS timestep is compared to the NLP dt.
    # E.g. COARSENESS_FACTOR=5 with NLP dt=0.1s → MCTS dt=0.5s.
    # This gives larger per-step rate-limit budgets and fewer steps
    # to cover the same time horizon, making the discrete action grid
    # practical while still producing trajectories over the full
    # planning window.
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
                 n_simulations: int = 5000,
                 exploration_constant: float = 10.0,
                 n_accel_levels: int = 21,
                 n_steer_levels: int = 21,
                 max_trajectories: int = 25,
                 gamma: float = 0.99,
                 collision_penalty: float = 100.0,
                 clearance_threshold: float = 4.0,
                 rollout_policy: str = 'heuristic',
                 activity_threshold: float = 1e-3):

        # Apply coarseness: larger dt, fewer horizon steps, same time span
        self._coarseness = self.COARSENESS_FACTOR
        self._dt_fine = dt                          # original NLP dt
        self._dt = dt * self._coarseness            # coarse MCTS dt
        self._horizon = max(1, horizon // self._coarseness)

        self._ego_length = ego_length
        self._ego_width = ego_width
        self._half_L = ego_length / 2.0
        self._half_W = ego_width / 2.0
        self._wheelbase = wheelbase
        self._collision_margin = collision_margin
        self._target_speed = target_speed
        self._frenet = frenet
        self._gamma = gamma
        self._collision_penalty = collision_penalty
        self._clearance_threshold = clearance_threshold
        self._rollout_policy = rollout_policy
        self._activity_threshold = activity_threshold
        self._n_simulations = n_simulations
        self._exploration_constant = exploration_constant
        self._max_trajectories = max_trajectories

        # Store original (fine) horizon for obstacle indexing
        self._horizon_fine = horizon

        # NLP parameters (bounds and weights)
        from igp2.beliefcontrol.second_stage import SecondStagePlanner
        self._params = dict(SecondStagePlanner.DEFAULTS)
        if nlp_params is not None:
            self._params.update(nlp_params)

        self._a_min = self._params['a_min']
        self._a_max = self._params['a_max']
        self._delta_max = self._params['delta_max']
        self._delta_rate_max = self._params['delta_rate_max']
        self._jerk_max = self._params['jerk_max']
        self._v_min = self._params['v_min']
        self._v_max = self._params['v_max']
        self._w_s = self._params['w_s']
        self._w_d = self._params['w_d']
        self._w_v = self._params['w_v']
        self._w_a = self._params['w_a']
        self._w_delta = self._params['w_delta']
        self._w_phi = self._params['w_phi']

        # Rate limits per coarse step (proportionally larger)
        self._jerk_limit = self._jerk_max * self._dt
        self._delta_rate_limit = self._delta_rate_max * self._dt

        # Build discrete action set
        self._actions = self._build_action_set(n_accel_levels, n_steer_levels)

        logger.info("MCTSPlanner: coarseness=%d, dt_fine=%.3f, dt_coarse=%.3f, "
                     "horizon_fine=%d, horizon_coarse=%d, "
                     "jerk_limit=%.4f/step, delta_rate_limit=%.4f/step, "
                     "n_actions=%d",
                     self._coarseness, self._dt_fine, self._dt,
                     horizon, self._horizon,
                     self._jerk_limit, self._delta_rate_limit,
                     len(self._actions))

        # Stored after each search for debug visualisation
        self._last_root: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def _build_action_set(self, n_accel: int, n_steer: int
                          ) -> List[Tuple[float, float]]:
        """Build discrete (a, delta) action set."""
        accels = np.linspace(self._a_min, self._a_max, n_accel)
        steers = np.linspace(-self._delta_max, self._delta_max, n_steer)

        # Ensure (0, 0) is always on the grid
        has_zero_a = any(abs(a) < 1e-12 for a in accels)
        has_zero_d = any(abs(d) < 1e-12 for d in steers)
        if not has_zero_a:
            accels = np.sort(np.append(accels, 0.0))
            logger.debug("  action grid: 0.0 NOT in accel linspace — injected")
        if not has_zero_d:
            steers = np.sort(np.append(steers, 0.0))
            logger.debug("  action grid: 0.0 NOT in steer linspace — injected")

        logger.debug("  accel grid (%d): %s", len(accels),
                      np.array2string(accels, precision=4, separator=', '))
        logger.debug("  steer grid (%d): %s", len(steers),
                      np.array2string(steers, precision=4, separator=', '))

        actions = []
        for a in accels:
            for d in steers:
                actions.append((round(float(a), 4), round(float(d), 4)))
        return actions

    def _check_rate_limits(self, prev_action: Optional[Tuple[float, float]],
                           action: Tuple[float, float]) -> bool:
        """Check jerk and steering-rate constraints against previous action."""
        if prev_action is None:
            return True
        a_prev, delta_prev = prev_action
        a_new, delta_new = action
        if abs(a_new - a_prev) > self._jerk_limit + 1e-6:
            return False
        if abs(delta_new - delta_prev) > self._delta_rate_limit + 1e-6:
            return False
        return True

    # ------------------------------------------------------------------
    # Bicycle dynamics (matching SecondStagePlanner exactly)
    # ------------------------------------------------------------------

    def _simulate_step(self, state: np.ndarray,
                       action: Tuple[float, float],
                       road_left_k: float,
                       road_right_k: float,
                       obstacles: Optional[list] = None,
                       belief_vector: Optional[Dict[int, bool]] = None,
                       step_k: int = 0,
                       ) -> Tuple[np.ndarray, bool]:
        """Forward one step of bicycle dynamics in Frenet coordinates.

        Dynamics (matching second_stage.py):
            s' = s + v * cos(phi + delta) * dt
            d' = d + v * sin(phi + delta) * dt
            phi' = phi + (2*v/L) * sin(delta) * dt
            v' = clip(v + a * dt, v_min, v_max)

        Returns:
            (new_state, feasible) where feasible=False if road bounds
            or collision constraints are violated.
        """
        s, d, phi, v = state
        a, delta = action
        dt = self._dt
        L = self._wheelbase

        s_new = s + v * math.cos(phi + delta) * dt
        d_new = d + v * math.sin(phi + delta) * dt
        phi_new = phi + (2.0 * v / L) * math.sin(delta) * dt
        v_new = max(self._v_min, min(self._v_max, v + a * dt))

        # Wrap heading to [-pi, pi]
        phi_new = (phi_new + math.pi) % (2.0 * math.pi) - math.pi

        new_state = np.array([s_new, d_new, phi_new, v_new])

        # Check road boundaries (corner-based, matching NLP)
        feasible = self._check_road_bounds(d_new, phi_new,
                                           road_left_k, road_right_k)

        # Check collision constraints against believed obstacles
        if feasible and obstacles is not None:
            feasible = self._check_collision(new_state, obstacles,
                                             belief_vector, step_k)

        return new_state, feasible

    def _check_collision(self, state: np.ndarray, obstacles: list,
                         belief_vector: Optional[Dict[int, bool]],
                         step_k: int) -> bool:
        """Return True if the state is collision-free w.r.t. believed obstacles.

        Only obstacles whose agent_id maps to True in the belief_vector
        are checked.  If belief_vector is None, no obstacles are checked
        (all ignored).
        """
        if belief_vector is None:
            return True

        fine_k = self._coarse_to_fine(step_k)
        for obs in obstacles:
            aid = obs['agent_id']
            if not belief_vector.get(aid, False):
                continue
            g_min = self._min_ellipse_value(state, obs, fine_k)
            if g_min < 1.0:
                return False
        return True

    def _check_road_bounds(self, d: float, phi: float,
                           road_left: float, road_right: float) -> bool:
        """Check all 4 ego corners stay within road boundaries."""
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        half_L = self._half_L
        half_W = self._half_W

        for sl, sw in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            c_d = d + sl * half_L * sin_phi + sw * half_W * cos_phi
            if c_d < road_right - 1e-3 or c_d > road_left + 1e-3:
                return False
        return True

    # ------------------------------------------------------------------
    # Reward function with per-agent decomposition
    # ------------------------------------------------------------------

    def _coarse_to_fine(self, coarse_k: int) -> int:
        """Map a coarse MCTS step index to the fine (NLP) step index."""
        return coarse_k * self._coarseness

    def _step_reward(self, state: np.ndarray,
                     action: Tuple[float, float],
                     obstacles: list,
                     step_k: int,
                     s0: float
                     ) -> Tuple[float, Dict[int, float]]:
        """Compute step reward with per-obstacle decomposition.

        ``step_k`` is the coarse MCTS step index.

        Returns:
            (total_reward, obstacle_rewards_dict) where
            obstacle_rewards_dict maps agent_id → reward component.
        """
        s, d, phi, v = state
        a, delta = action

        # Map coarse step to fine for reference position and obstacles
        fine_k = self._coarse_to_fine(step_k)

        # Tracking cost (negated to form reward, matching NLP cost structure)
        s_max = self._frenet.total_length if self._frenet is not None else 1e6
        ref_s = min(s0 + self._target_speed * fine_k * self._dt_fine, s_max)
        tracking = -(self._w_s * (s - ref_s) ** 2
                     + self._w_d * d ** 2
                     + self._w_phi * phi ** 2
                     + self._w_v * (v - self._target_speed) ** 2)

        # Control regularisation
        control = -(self._w_a * a ** 2 + self._w_delta * delta ** 2)

        # Per-obstacle collision/clearance reward
        obstacle_rewards: Dict[int, float] = {}
        collision_total = 0.0

        for obs in obstacles:
            aid = obs['agent_id']
            g_min = self._min_ellipse_value(state, obs, fine_k)

            if g_min < 1.0:
                # Inside collision ellipse — large penalty
                r_obs = -self._collision_penalty * (1.0 - g_min)
            elif g_min < self._clearance_threshold:
                # Close to obstacle — clearance penalty (decays with distance)
                r_obs = -(self._clearance_threshold - g_min) / self._clearance_threshold
            else:
                r_obs = 0.0

            obstacle_rewards[aid] = r_obs
            collision_total += r_obs

        total = tracking + control + collision_total
        return total, obstacle_rewards

    def _min_ellipse_value(self, state: np.ndarray, obs: dict,
                           step_k: int) -> float:
        """Minimum ellipse g-value across all 4 ego corners.

        Uses the identical elliptical formulation from SecondStagePlanner
        (Eiras et al. Eq. 9-10) but with plain math instead of CasADi.

        g >= 1 means safe (corner outside ellipse).
        g < 1 means collision (corner inside ellipse).
        """
        s, d, phi, v = state
        half_L = self._half_L
        half_W = self._half_W

        # Ellipse semi-axes
        a_i = obs['length'] / 2.0 + self._collision_margin
        b_i = obs['width'] / 2.0 + self._collision_margin

        # Obstacle position at step_k
        s_arr = obs['s']
        d_arr = obs['d']
        k = min(step_k, len(s_arr) - 1)
        s_obs = float(s_arr[k])
        d_obs = float(d_arr[k])

        # Obstacle heading in Frenet frame
        if self._frenet is not None:
            obs_s0 = float(s_arr[0])
            _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
            phi_i = obs.get('heading', road_angle_obs) - road_angle_obs
        else:
            phi_i = 0.0

        cos_phi_i = math.cos(phi_i)
        sin_phi_i = math.sin(phi_i)

        # Ego heading rotation
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        g_min = float('inf')
        for alpha_l, alpha_w in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            # Corner position (Eq. 6)
            c_s = s + alpha_l * half_L * cos_phi - alpha_w * half_W * sin_phi
            c_d = d + alpha_l * half_L * sin_phi + alpha_w * half_W * cos_phi

            # Vector from obstacle centre to corner
            d_s = c_s - s_obs
            d_d = c_d - d_obs

            # Rotate to obstacle body frame: R(-phi_i) * d
            d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
            d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d

            # Ellipse g-value (Eq. 10)
            g = (d_body_x / a_i) ** 2 + (d_body_y / b_i) ** 2
            if g < g_min:
                g_min = g

        return g_min

    # ------------------------------------------------------------------
    # Core MCTS loop
    # ------------------------------------------------------------------

    def search(self, frenet_state: np.ndarray,
               road_left: np.ndarray, road_right: np.ndarray,
               obstacles: list,
               prev_action: Optional[Tuple[float, float]] = None,
               belief_vector: Optional[Dict[int, bool]] = None,
               ) -> List[TrajectoryBeliefPair]:
        """Run MCTS and return trajectory-belief pairs.

        Args:
            frenet_state: Initial [s, d, phi, v] state.
            road_left: (H+1,) left road boundary at each step.
            road_right: (H+1,) right road boundary at each step.
            obstacles: List of obstacle dicts (all agents visible).
            prev_action: (a, delta) action applied just before the
                planning window, for rate-limit continuity at the root.
            belief_vector: {agent_id: bool} — True means the obstacle
                is considered for hard collision avoidance, False means
                ignored.  None means ignore all obstacles.

        Returns:
            List of TrajectoryBeliefPair sorted by MCTS reward (best first).
        """
        s0 = float(frenet_state[0])
        root = MCTSNode(state=frenet_state.copy(), depth=0,
                        action=prev_action)

        logger.debug("MCTS search start: state=[s=%.2f d=%.2f phi=%.3f v=%.2f], "
                     "horizon=%d, dt=%.3f, n_actions=%d, "
                     "jerk_limit=%.4f, delta_rate_limit=%.4f",
                     frenet_state[0], frenet_state[1],
                     frenet_state[2], frenet_state[3],
                     self._horizon, self._dt, len(self._actions),
                     self._jerk_limit, self._delta_rate_limit)
        logger.debug("  road bounds at k=0: left=%.3f  right=%.3f  |  "
                     "at k=%d: left=%.3f  right=%.3f",
                     road_left[0], road_right[0],
                     self._horizon,
                     road_left[min(self._horizon, len(road_left) - 1)],
                     road_right[min(self._horizon, len(road_right) - 1)])

        expand_fail_count = 0
        rollout_early_count = 0


        for sim_i in range(self._n_simulations):
            # 1. Selection: traverse tree using UCB1, accumulating
            #    step rewards along the path
            node, path_reward, path_obs = self._select(
                root, road_left, road_right, obstacles, s0, belief_vector)

            # 2. Expansion: add a new child if not terminal
            expand_reward = 0.0
            expand_obs: Dict[int, float] = {}
            if not node.is_terminal(self._horizon):
                old_node = node
                node, expand_reward, expand_obs = self._expand(
                    node, road_left, road_right,
                    obstacles, s0, belief_vector)
                if node is old_node:
                    expand_fail_count += 1

            # 3. Simulation (rollout): play out to horizon
            # Trace first rollout from each root child
            rollout_reward, rollout_obs = self._rollout(
                node, road_left, road_right, obstacles, s0, belief_vector)

            # Combine: path (selection) + expansion + rollout
            # Discount the expansion and rollout relative to the path
            path_depth = node.depth - (1 if expand_reward != 0.0 else 0)
            gamma_path = self._gamma ** max(path_depth, 0)
            gamma_expand = self._gamma ** max(node.depth - 1, 0) if expand_reward != 0.0 else 1.0

            total_reward = (path_reward
                            + gamma_expand * expand_reward
                            + gamma_path * rollout_reward)

            # Merge per-agent obstacle rewards
            total_obs: Dict[int, float] = dict(path_obs)
            for aid, r in expand_obs.items():
                total_obs[aid] = total_obs.get(aid, 0.0) + r
            for aid, r in rollout_obs.items():
                total_obs[aid] = total_obs.get(aid, 0.0) + r

            # 4. Backpropagation
            self._backpropagate(node, total_reward, total_obs)

        self._last_root = root

        # Tree stats
        max_depth = 0
        total_nodes = 0
        queue = [root]
        while queue:
            n = queue.pop()
            total_nodes += 1
            if n.depth > max_depth:
                max_depth = n.depth
            for c in n.children.values():
                if c is not None:
                    queue.append(c)

        logger.info("MCTS tree stats: %d nodes, max_depth=%d/%d, "
                     "expand_fails=%d/%d",
                     total_nodes, max_depth, self._horizon,
                     expand_fail_count, self._n_simulations)

        # Per-layer visit count breakdown
        layer_visits: dict[int, list[int]] = {}
        queue2 = [root]
        while queue2:
            n = queue2.pop()
            layer_visits.setdefault(n.depth, []).append(n.visit_count)
            for c in n.children.values():
                if c is not None:
                    queue2.append(c)
        for depth in sorted(layer_visits.keys()):
            visits = sorted(layer_visits[depth], reverse=True)
            total_v = sum(visits)
            n_layer = len(visits)
            top5 = visits[:5]
            logger.debug("  layer %d: %d nodes, total_visits=%d, "
                          "top5_visits=%s",
                          depth, n_layer, total_v, top5)

        # Debug: dump root children sorted by average reward
        root_children = [(a, c) for a, c in root.children.items()
                         if c is not None]
        root_children.sort(key=lambda x: x[1].total_reward / max(x[1].visit_count, 1),
                           reverse=True)
        logger.debug("MCTS root children (%d valid / %d total):",
                      len(root_children), len(root.children))
        for i, (action, child) in enumerate(root_children[:10]):
            avg_r = child.total_reward / max(child.visit_count, 1)
            s, d, phi, v = child.state
            fine_k = self._coarse_to_fine(1)
            ref_s = min(s0 + self._target_speed * fine_k * self._dt_fine,
                        self._frenet.total_length if self._frenet else 1e6)
            r_track_s = -self._w_s * (s - ref_s) ** 2
            r_track_d = -self._w_d * d ** 2
            r_track_phi = -self._w_phi * phi ** 2
            r_track_v = -self._w_v * (v - self._target_speed) ** 2
            r_ctrl_a = -self._w_a * action[0] ** 2
            r_ctrl_d = -self._w_delta * action[1] ** 2
            r_obs_total = 0.0
            for obs in obstacles:
                g = self._min_ellipse_value(child.state, obs, fine_k)
                if g < 1.0:
                    r_obs_total += -self._collision_penalty * (1.0 - g)
                elif g < self._clearance_threshold:
                    r_obs_total += -(self._clearance_threshold - g) / self._clearance_threshold
            logger.debug(
                "  [%2d] a=(%.3f,%.3f) vis=%d avg_r=%.3f | "
                "s_err=%.3f(r=%.4f) d=%.3f(r=%.4f) phi=%.4f(r=%.4f) "
                "v_err=%.3f(r=%.4f) ctrl_a=%.4f ctrl_d=%.4f obs=%.4f",
                i, action[0], action[1], child.visit_count, avg_r,
                s - ref_s, r_track_s, d, r_track_d, phi, r_track_phi,
                v - self._target_speed, r_track_v,
                r_ctrl_a, r_ctrl_d, r_obs_total)

        # Extract top-k trajectories
        trajectories = self._extract_trajectories(
            root, self._max_trajectories, road_left, road_right,
            obstacles, s0, belief_vector=belief_vector)

        # Sort best first
        trajectories.sort(key=lambda t: -t.mcts_reward)

        # Debug: log each extracted trajectory's coverage
        for ti, tbp in enumerate(trajectories):
            s_range = float(tbp.states[-1, 0] - tbp.states[0, 0])
            d_range = float(np.max(tbp.states[:, 1]) - np.min(tbp.states[:, 1]))
            n_unique = len(set(tbp.states[:, 0].round(4)))

            # logger.info(f"    traj[{ti}]: {tbp.states}")
            # logger.info("  traj[%d]: s_range=%.2f  d_range=%.2f  "
            #              "n_unique_s=%d/%d  reward=%.2f  belief={%s}",
            #              ti, s_range, d_range,
            #              n_unique, len(tbp.states), tbp.mcts_reward,
            #              ", ".join(f"{a}:{'V' if v else 'H'}"
            #                        for a, v in sorted(tbp.implied_belief.items())))

        logger.info("MCTS search: %d simulations, %d trajectories extracted",
                     self._n_simulations, len(trajectories))
        return trajectories

    def _select(self, node: MCTSNode,
                road_left: np.ndarray, road_right: np.ndarray,
                obstacles: list, s0: float,
                belief_vector: Optional[Dict[int, bool]] = None,
                ) -> Tuple[MCTSNode, float, Dict[int, float]]:
        """Traverse tree by UCB1 until reaching an expandable or terminal node.

        Returns:
            (node, path_reward, path_obs_rewards) where path_reward is the
            cumulative reward along the selection path (steps traversed
            from root to the returned node).
        """
        path_reward = 0.0
        path_obs_rewards: Dict[int, float] = {}
        discount = 1.0

        while not node.is_terminal(self._horizon):
            if not node.is_fully_expanded(self._actions, self):
                return node, path_reward, path_obs_rewards
            child = node.best_child_ucb(self._exploration_constant)
            if child is None:
                # All children infeasible — treat as terminal
                return node, path_reward, path_obs_rewards

            # Accumulate step reward for this transition
            step_r, step_obs_r = self._step_reward(
                child.state, child.action, obstacles, child.depth, s0)
            path_reward += discount * step_r
            discount *= self._gamma
            for aid, r in step_obs_r.items():
                path_obs_rewards[aid] = path_obs_rewards.get(aid, 0.0) + r

            node = child
        return node, path_reward, path_obs_rewards

    def _expand(self, node: MCTSNode,
                road_left: np.ndarray, road_right: np.ndarray,
                obstacles: list, s0: float,
                belief_vector: Optional[Dict[int, bool]] = None,
                ) -> Tuple[MCTSNode, float, Dict[int, float]]:
        """Expand one untried action from the node.

        Only creates child nodes for feasible (road-boundary and
        collision-free) actions.  Infeasible actions are discarded from
        the untried list so they are never revisited.

        Returns:
            (child_node, step_reward, step_obs_rewards) for the
            expansion step.  If no expansion occurs, returns
            (node, 0.0, {}).
        """
        untried = node.untried_actions(self._actions, self)
        if not untried:
            return node, 0.0, {}

        random.shuffle(untried)

        k = node.depth
        fine_k = self._coarse_to_fine(k + 1)
        rl = road_left[min(fine_k, len(road_left) - 1)]
        rr = road_right[min(fine_k, len(road_right) - 1)]

        for action in untried:
            new_state, feasible = self._simulate_step(
                node.state, action, rl, rr,
                obstacles=obstacles, belief_vector=belief_vector,
                step_k=k + 1)

            if not feasible:
                # Mark as tried so it won't be retried
                node.children[action] = None  # sentinel
                continue

            child = MCTSNode(
                state=new_state,
                depth=k + 1,
                parent=node,
                action=action,
                prev_action=node.action,
            )
            node.children[action] = child

            # Invalidate untried cache
            node._untried = None

            # Compute step reward for the expansion transition
            step_r, step_obs_r = self._step_reward(
                new_state, action, obstacles, k + 1, s0)
            return child, step_r, step_obs_r

        # All untried actions were infeasible — return stuck-state penalty
        node._untried = None
        s, d, phi, v = node.state
        rl_val = road_left[min(node.depth + 1, len(road_left) - 1)]
        rr_val = road_right[min(node.depth + 1, len(road_right) - 1)]
        logger.debug("  expand: ALL %d actions infeasible at depth=%d "
                      "state=[s=%.2f d=%.2f phi=%.3f v=%.2f] "
                      "road=[%.3f, %.3f]",
                      len(untried), node.depth, s, d, phi, v,
                      rr_val, rl_val)
        stuck_r, stuck_obs_r = self._step_reward(
            node.state, (0.0, 0.0), obstacles, node.depth + 1, s0)
        return node, stuck_r, stuck_obs_r

    def _rollout(self, node: MCTSNode,
                 road_left: np.ndarray, road_right: np.ndarray,
                 obstacles: list, s0: float,
                 belief_vector: Optional[Dict[int, bool]] = None,
                 ) -> Tuple[float, Dict[int, float]]:
        """Simulate from node to horizon using default policy.

        Hard-enforces road boundaries and collision avoidance: if the
        heuristic action is infeasible, tries alternative actions.  If
        no action keeps the vehicle on the road and collision-free, the
        rollout terminates early.

        Returns:
            (cumulative_discounted_reward, per_agent_obstacle_rewards)
        """
        state = node.state.copy()
        depth = node.depth
        prev_action = node.action
        cumulative_reward = 0.0
        obs_rewards_total: Dict[int, float] = {}
        discount = 1.0

        while depth < self._horizon:
            k_next = depth + 1
            fine_k = self._coarse_to_fine(k_next)
            rl = road_left[min(fine_k, len(road_left) - 1)]
            rr = road_right[min(fine_k, len(road_right) - 1)]

            # Try heuristic action first
            action = self._rollout_action(state, prev_action, rl, rr)
            new_state, feasible = self._simulate_step(
                state, action, rl, rr,
                obstacles=obstacles, belief_vector=belief_vector,
                step_k=k_next)

            if not feasible:
                # Heuristic failed — try all discrete actions
                new_state, action, feasible = self._find_feasible_action(
                    state, prev_action, rl, rr,
                    obstacles=obstacles, belief_vector=belief_vector,
                    step_k=k_next)
                if not feasible:

                    # Penalise remaining horizon steps: the vehicle is
                    # stuck at `state` while ref_s keeps advancing each
                    # step, so compute reward per-step with advancing k.
                    remaining = self._horizon - depth
                    stuck_action = (0.0, 0.0)
                    for r_step in range(remaining):
                        stuck_k = depth + 1 + r_step
                        stuck_r, stuck_obs_r = self._step_reward(
                            state, stuck_action, obstacles, stuck_k, s0)
                        cumulative_reward += discount * stuck_r
                        discount *= self._gamma
                        for aid, r in stuck_obs_r.items():
                            obs_rewards_total[aid] = (
                                obs_rewards_total.get(aid, 0.0) + r
                            )
                    break

            # Reward
            step_r, step_obs_r = self._step_reward(
                new_state, action, obstacles, k_next, s0)

            cumulative_reward += discount * step_r
            discount *= self._gamma

            # Accumulate per-agent rewards
            for aid, r in step_obs_r.items():
                obs_rewards_total[aid] = obs_rewards_total.get(aid, 0.0) + r

            state = new_state
            prev_action = action
            depth += 1

        return cumulative_reward, obs_rewards_total

    def _find_feasible_action(self, state: np.ndarray,
                              prev_action: Optional[Tuple[float, float]],
                              road_left_k: float, road_right_k: float,
                              obstacles: Optional[list] = None,
                              belief_vector: Optional[Dict[int, bool]] = None,
                              step_k: int = 0,
                              ) -> Tuple[np.ndarray, Tuple[float, float], bool]:
        """Try all discrete actions to find one that stays on the road
        and is collision-free.

        Returns (new_state, action, feasible).  If no action is feasible
        returns a dummy state with feasible=False.
        """
        valid = [a for a in self._actions
                 if self._check_rate_limits(prev_action, a)]
        random.shuffle(valid)

        for action in valid:
            new_state, ok = self._simulate_step(
                state, action, road_left_k, road_right_k,
                obstacles=obstacles, belief_vector=belief_vector,
                step_k=step_k)
            if ok:
                return new_state, action, True

        return state, (0.0, 0.0), False

    def _rollout_action(self, state: np.ndarray,
                        prev_action: Optional[Tuple[float, float]],
                        road_left_k: float = 1e6,
                        road_right_k: float = -1e6,
                        ) -> Tuple[float, float]:
        """Select action during rollout phase."""
        if self._rollout_policy == 'heuristic':
            return self._heuristic_action(state, prev_action,
                                          road_left_k, road_right_k)
        else:
            return self._random_valid_action(prev_action)

    def _heuristic_action(self, state: np.ndarray,
                          prev_action: Optional[Tuple[float, float]],
                          road_left_k: float = 1e6,
                          road_right_k: float = -1e6,
                          ) -> Tuple[float, float]:
        """Heuristic: accelerate towards target speed, steer towards centre.

        Drives at target speed with minimal steering, respecting rate
        limits relative to the previous action.  The lateral controller
        is clamped to keep d within road boundaries.
        """
        _, d, phi, v = state

        # Longitudinal: accelerate towards target speed
        speed_err = self._target_speed - v
        a_desired = np.clip(speed_err / self._dt,
                            self._a_min, self._a_max)

        # Lateral: steer towards d=0, phi=0
        # Add boundary-aware correction — steer harder when near edges
        d_target = 0.0
        margin = self._half_W + 0.3  # keep some clearance from road edge
        if d > road_left_k - margin:
            # Too close to left boundary — steer right
            d_target = road_left_k - margin - 0.5
        elif d < road_right_k + margin:
            # Too close to right boundary — steer left
            d_target = road_right_k + margin + 0.5

        # Scale gains inversely with coarse dt to prevent oscillation.
        # At dt_fine=0.1 the base gains (2.0, 1.0) are stable; at
        # dt_coarse=0.5 (coarseness=5) they become (0.4, 0.2).
        k_d = 2.0 * (self._dt_fine / self._dt)
        k_phi = 1.0 * (self._dt_fine / self._dt)
        delta_desired = np.clip(-k_d * (d - d_target) - k_phi * phi,
                                -self._delta_max, self._delta_max)

        # Enforce rate limits
        if prev_action is not None:
            a_prev, delta_prev = prev_action
            a_desired = np.clip(a_desired,
                                a_prev - self._jerk_limit,
                                a_prev + self._jerk_limit)
            delta_desired = np.clip(delta_desired,
                                    delta_prev - self._delta_rate_limit,
                                    delta_prev + self._delta_rate_limit)

        # Clip to absolute bounds
        a_desired = np.clip(a_desired, self._a_min, self._a_max)
        delta_desired = np.clip(delta_desired,
                                -self._delta_max, self._delta_max)

        return (float(a_desired), float(delta_desired))

    def _random_valid_action(self, prev_action: Optional[Tuple[float, float]]
                             ) -> Tuple[float, float]:
        """Sample a random action that satisfies rate limits."""
        valid = [a for a in self._actions
                 if self._check_rate_limits(prev_action, a)]
        if not valid:
            return (0.0, 0.0)
        return random.choice(valid)

    def _backpropagate(self, node: MCTSNode,
                       reward: float,
                       obs_rewards: Dict[int, float]):
        """Propagate reward up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            for aid, r in obs_rewards.items():
                node.obstacle_reward_totals[aid] = (
                    node.obstacle_reward_totals.get(aid, 0.0) + r)
            node = node.parent

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    def _extract_trajectories(self, root: MCTSNode, k: int,
                              road_left: np.ndarray,
                              road_right: np.ndarray,
                              obstacles: list,
                              s0: float,
                              belief_vector: Optional[Dict[int, bool]] = None,
                              ) -> List[TrajectoryBeliefPair]:
        """Extract top-k distinct trajectories from the MCTS tree.

        Uses best-first DFS following highest visit count at each level.
        At branch points with multiple well-visited children, forks to
        collect diverse trajectories.
        """
        results: List[TrajectoryBeliefPair] = []
        agent_ids = set()
        for obs in obstacles:
            aid = obs['agent_id']
            if aid >= 0:
                agent_ids.add(aid)

        # Collect all leaf-to-root paths via DFS ordered by visit count
        stack: List[List[MCTSNode]] = [[root]]

        while stack and len(results) < k:
            path = stack.pop()
            node = path[-1]

            # Filter out None sentinels (infeasible actions)
            valid_children = {a: c for a, c in node.children.items()
                              if c is not None}

            if node.is_terminal(self._horizon) or not valid_children:
                # Build trajectory from this path
                traj = self._path_to_trajectory(
                    path, road_left, road_right, obstacles, s0, agent_ids,
                    belief_vector=belief_vector)
                if traj is not None:
                    results.append(traj)
                continue

            # Sort children by visit count (descending) for best-first
            sorted_children = sorted(
                valid_children.values(),
                key=lambda c: c.visit_count,
                reverse=True)

            # Push children onto stack in reverse order so highest-visit
            # is popped first
            for child in reversed(sorted_children):
                if child.visit_count > 0:
                    stack.append(path + [child])

        return results

    def _path_to_trajectory(self, path: List[MCTSNode],
                            road_left: np.ndarray,
                            road_right: np.ndarray,
                            obstacles: list,
                            s0: float,
                            agent_ids: set,
                            belief_vector: Optional[Dict[int, bool]] = None,
                            ) -> Optional[TrajectoryBeliefPair]:
        """Convert a tree path to a TrajectoryBeliefPair.

        If the path doesn't reach the horizon, extends it using the
        rollout policy (padding).
        """
        H = self._horizon

        # Collect states and actions from the path
        states = [path[0].state.copy()]
        controls = []
        obs_rewards_total: Dict[int, float] = {}
        cumulative_reward = 0.0
        discount = 1.0

        for i in range(1, len(path)):
            node = path[i]
            states.append(node.state.copy())
            controls.append(node.action)

            # Compute reward for this step
            step_r, step_obs_r = self._step_reward(
                node.state, node.action, obstacles, node.depth, s0)
            cumulative_reward += discount * step_r
            discount *= self._gamma
            for aid, r in step_obs_r.items():
                obs_rewards_total[aid] = obs_rewards_total.get(aid, 0.0) + r

        # Pad to horizon if needed
        depth = len(path) - 1
        state = states[-1].copy()
        prev_action = controls[-1] if controls else None

        while depth < H:
            k_next = depth + 1
            fine_k = self._coarse_to_fine(k_next)
            rl = road_left[min(fine_k, len(road_left) - 1)]
            rr = road_right[min(fine_k, len(road_right) - 1)]

            action = self._heuristic_action(state, prev_action, rl, rr)
            new_state, feasible = self._simulate_step(
                state, action, rl, rr,
                obstacles=obstacles, belief_vector=belief_vector,
                step_k=k_next)

            if not feasible:
                # Try all discrete actions
                new_state, action, feasible = self._find_feasible_action(
                    state, prev_action, rl, rr,
                    obstacles=obstacles, belief_vector=belief_vector,
                    step_k=k_next)
                if not feasible:
                    s, d, phi, v = state
                    logger.debug("  pad: stuck at depth=%d/%d, padding %d "
                                  "remaining steps. state=[s=%.2f d=%.2f "
                                  "phi=%.3f v=%.2f] road=[%.3f, %.3f]",
                                  depth, H, H - depth, s, d, phi, v, rr, rl)
                    # Pad remaining steps with current state and zero controls
                    while depth < H:
                        states.append(state.copy())
                        controls.append((0.0, 0.0))
                        depth += 1
                    break

            step_r, step_obs_r = self._step_reward(
                new_state, action, obstacles, k_next, s0)
            cumulative_reward += discount * step_r
            discount *= self._gamma
            for aid, r in step_obs_r.items():
                obs_rewards_total[aid] = obs_rewards_total.get(aid, 0.0) + r

            states.append(new_state)
            controls.append(action)
            state = new_state
            prev_action = action
            depth += 1

        # Derive belief from per-agent obstacle rewards
        implied_belief = self._derive_belief(obs_rewards_total, agent_ids)

        states_arr = np.array(states)        # (H+1, 4)
        controls_arr = np.array(controls)    # (H, 2)

        tree_depth = len(path) - 1
        pad_depth = len(states_arr) - 1 - tree_depth
        s_start = float(states_arr[0, 0])
        s_end = float(states_arr[-1, 0])
        logger.debug("  path_to_traj: tree_depth=%d  pad=%d  "
                      "s=[%.2f → %.2f] (Δs=%.2f)  "
                      "d=[%.3f → %.3f]  v=[%.2f → %.2f]",
                      tree_depth, pad_depth,
                      s_start, s_end, s_end - s_start,
                      float(states_arr[0, 1]), float(states_arr[-1, 1]),
                      float(states_arr[0, 3]), float(states_arr[-1, 3]))

        return TrajectoryBeliefPair(
            states=states_arr,
            controls=controls_arr,
            implied_belief=implied_belief,
            mcts_reward=cumulative_reward,
            obstacle_rewards=obs_rewards_total,
        )

    # ------------------------------------------------------------------
    # Belief derivation
    # ------------------------------------------------------------------

    def _derive_belief(self, obstacle_rewards: Dict[int, float],
                       agent_ids: set) -> Dict[int, bool]:
        """Derive implied belief from per-obstacle reward decomposition.

        Agents with significant collision/clearance influence are
        considered "visible" (the trajectory was shaped by avoiding them).
        Agents with negligible influence are considered "hidden".

        Args:
            obstacle_rewards: {agent_id: cumulative reward} from trajectory.
            agent_ids: Set of dynamic agent IDs to consider.

        Returns:
            {agent_id: visible} dict.
        """
        belief: Dict[int, bool] = {}
        for aid in agent_ids:
            total_influence = abs(obstacle_rewards.get(aid, 0.0))
            belief[aid] = total_influence > self._activity_threshold
        return belief
