"""Two-stage trajectory optimisation policy (MILP + NLP).

Orchestrates :class:`FirstStagePlanner` and :class:`SecondStagePlanner`
to perform receding-horizon MPC in Frenet coordinates.  All shared
state (Frenet frame, obstacle prediction, warm-starting, diagnostics)
lives here.
"""

import logging
import time
from typing import List, Optional, Dict, Tuple

import numpy as np

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.vehicle import Action
from igp2.opendrive.map import Map
from igp2.beliefcontrol.frenet import FrenetFrame
from igp2.beliefcontrol.first_stage import FirstStagePlanner
from igp2.beliefcontrol.second_stage import SecondStagePlanner

logger = logging.getLogger(__name__)


class TwoStagePolicy:
    """Two-stage Frenet-frame trajectory optimisation: MILP + CasADi NLP.

    All optimisation is performed in Frenet (reference-path) coordinates,
    where road boundaries become simple box constraints
    ``d_right <= d <= d_left``.

    **Stage 1 — FirstStagePlanner (coarse trajectory).**
    A smooth approximation over a point-mass model ``[s, d, vs, vd]``
    in Frenet coordinates.

    **Stage 2 — SecondStagePlanner (refined trajectory).**
    A CasADi + IPOPT nonlinear program over the bicycle kinematic model
    ``[s, d, phi, v]`` in Frenet coordinates.

    The planning timestep ``dt`` (default 0.1 s) is independent of the
    simulation framerate.  Each MPC call plans at ``dt`` resolution over
    ``horizon`` steps (default 40 = 4 s), but only the first action is
    applied for one simulation step (receding-horizon MPC).

    Args:
        fps: Simulation framerate.
        metadata: Agent physical metadata (wheelbase, limits, etc.).
        reference_waypoints: Concatenated A* reference path (N, 2).
        scenario_map: Road layout for boundary queries. May be None.
        horizon: Number of planning steps (default 40).
        dt: Planning timestep in seconds (default 0.1).
        target_speed: Desired cruising speed (m/s).
        collision_margin: Extra safety margin around obstacles (m).
        milp_params: Dict of first-stage parameters.
        nlp_params: Dict of second-stage parameters.
        use_prev_nlp_on_fail: Use previous NLP solution as fallback on failure.
    """

    DEFAULT_HORIZON = 40
    DEFAULT_DT = 0.1
    N_OBS_MAX = 10

    def __init__(self,
                 fps: int,
                 metadata: AgentMetadata,
                 reference_waypoints: np.ndarray,
                 scenario_map: Optional[Map] = None,
                 horizon: int = None,
                 dt: float = None,
                 target_speed: float = 5.0,
                 collision_margin: float = 0.9,
                 milp_params: Optional[Dict] = None,
                 nlp_params: Optional[Dict] = None,
                 use_prev_nlp_on_fail: bool = True,
                 # Legacy params (for back-compat, override milp/nlp_params)
                 big_m: float = None,
                 delta_max: float = None,
                 a_min: float = None,
                 a_max: float = None,
                 jerk_max: float = None,
                 v_max: float = None,
                 milp_rho: float = None,
                 w_x: float = None,
                 w_v: float = None,
                 w_y: float = None,
                 w_a: float = None,
                 w_delta: float = None,
                 **kwargs):
        self._fps = fps
        self._dt_sim = 1.0 / fps
        self._dt = dt if dt is not None else self.DEFAULT_DT
        self._metadata = metadata
        self._reference_waypoints = np.asarray(reference_waypoints, dtype=float)
        self._scenario_map = scenario_map
        self._horizon = horizon if horizon is not None else self.DEFAULT_HORIZON
        self._target_speed = target_speed
        self._use_prev_nlp_on_fail = use_prev_nlp_on_fail

        # Build parameter dicts with legacy overrides
        _milp = dict(FirstStagePlanner.DEFAULTS)
        if milp_params is not None:
            _milp.update(milp_params)
        _nlp = dict(SecondStagePlanner.DEFAULTS)
        if nlp_params is not None:
            _nlp.update(nlp_params)

        # Apply legacy parameter overrides
        if milp_rho is not None:
            _milp['rho'] = milp_rho
        if v_max is not None:
            _milp['vs_max'] = v_max
            _nlp['v_max'] = v_max
        if a_min is not None:
            _milp['a_s_min'] = a_min
            _milp['a_d_min'] = a_min
            _nlp['a_min'] = a_min
        if a_max is not None:
            _milp['a_s_max'] = a_max
            _milp['a_d_max'] = a_max
            _nlp['a_max'] = a_max
        if delta_max is not None:
            _nlp['delta_max'] = delta_max
        if jerk_max is not None:
            _nlp['jerk_max'] = jerk_max
        if w_x is not None:
            _milp['w_s'] = w_x
            _nlp['w_s'] = w_x
        if w_y is not None:
            _milp['w_d'] = w_y
            _nlp['w_d'] = w_y
        if w_v is not None:
            _milp['w_v'] = w_v
            _nlp['w_v'] = w_v
        if w_a is not None:
            _milp['w_a_s'] = w_a
            _milp['w_a_d'] = w_a
            _nlp['w_a'] = w_a
        if w_delta is not None:
            _nlp['w_delta'] = w_delta

        # Vehicle dimensions
        self._collision_margin = collision_margin
        self._ego_length = metadata.length
        self._ego_width = metadata.width
        self._wheelbase = metadata.wheelbase

        # Frenet frame
        self._frenet: Optional[FrenetFrame] = None
        if len(self._reference_waypoints) >= 2:
            self._frenet = FrenetFrame(self._reference_waypoints)

        # Create stage planners
        self._first_stage = FirstStagePlanner(
            horizon=self._horizon,
            dt=self._dt,
            ego_length=self._ego_length,
            ego_width=self._ego_width,
            collision_margin=collision_margin,
            target_speed=target_speed,
            frenet=self._frenet,
            params=_milp,
            n_obs_max=self.N_OBS_MAX,
        )
        self._second_stage = SecondStagePlanner(
            horizon=self._horizon,
            dt=self._dt,
            ego_length=self._ego_length,
            ego_width=self._ego_width,
            wheelbase=self._wheelbase,
            collision_margin=collision_margin,
            target_speed=target_speed,
            frenet=self._frenet,
            params=_nlp,
            n_obs_max=self.N_OBS_MAX,
        )

        # MPC state
        self._prev_milp_states: Optional[np.ndarray] = None
        self._prev_nlp_states: Optional[np.ndarray] = None
        self._prev_nlp_controls: Optional[np.ndarray] = None
        self._last_rollout: Optional[np.ndarray] = None
        self._last_milp_rollout: Optional[np.ndarray] = None
        self._ref_start_idx: int = 0
        self._step_count: int = 0

        # Per-step diagnostics
        self._last_diagnostics: Optional[Dict] = None

        # Obstacle data (stored for plotter access)
        self._last_obstacles: Optional[List] = None
        self._last_other_agents: Optional[Dict] = None

        # NLP solver cache
        self._nlp_solver = None
        self._nlp_built = False

        # Debug predicted trajectory
        self._predicted_next_state: Optional[np.ndarray] = None
        self._predicted_next_world: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: AgentState,
                      other_agents: Optional[Dict] = None,
                      agent_trajectories: Optional[Dict[int, np.ndarray]] = None) -> tuple:
        """MPC step: solve first stage + second stage in Frenet frame, return first action.

        Args:
            state: Current ego agent state.
            other_agents: Dict mapping agent_id -> AgentState for other
                vehicles (used for collision avoidance). May be None.
            agent_trajectories: Optional dict mapping agent_id -> (T, 2) array
                of planned world positions over the planning horizon.

        Returns:
            (action, [trajectory_positions], 0)
        """
        if self._frenet is None:
            action = Action(acceleration=0.0, steer_angle=0.0,
                            target_speed=self._target_speed)
            return action, [np.array([state.position])], 0

        H = self._horizon
        dt = self._dt

        self._advance_reference_window(state.position)

        frenet_state = self._state_to_frenet(state)

        s_values = np.array([frenet_state[0] + frenet_state[3] *
                             np.cos(frenet_state[2]) * k * dt
                             for k in range(H + 1)])
        s_values = np.clip(s_values, 0.0, self._frenet.total_length)
        road_left, road_right = self._sample_road_boundaries(s_values)

        obstacles = self._predict_obstacles(other_agents, self._frenet, agent_trajectories)
        self._last_obstacles = obstacles
        self._last_other_agents = other_agents

        self._step_count += 1

        # --- Stage 1: First Stage ---
        t_milp_start = time.time()
        milp_states = self._first_stage.solve(frenet_state, road_left, road_right,
                                              obstacles)
        t_milp = time.time() - t_milp_start

        if milp_states is None:
            print(f"[Step {self._step_count:4d}] MILP: FAILED ({t_milp*1000:.1f}ms)")
            action = Action(acceleration=0.0, steer_angle=0.0,
                            target_speed=self._target_speed)
            return action, [np.array([state.position])], 0

        warm_states, warm_controls = self._milp_to_nlp_warmstart(
            milp_states, frenet_state)
        self._prev_milp_states = milp_states.copy()

        self._last_milp_rollout = self._milp_states_to_world(milp_states)

        # --- Stage 2: Second Stage ---
        t_nlp_start = time.time()
        nlp_states, nlp_controls, nlp_ok, nlp_debug = self._second_stage.solve(
            frenet_state, warm_states, warm_controls,
            road_left, road_right, obstacles)
        t_nlp = time.time() - t_nlp_start

        if not nlp_ok:
            if (self._use_prev_nlp_on_fail and
                self._prev_nlp_states is not None and
                self._prev_nlp_controls is not None):
                final_states = self._prev_nlp_states.copy()
                final_controls = self._prev_nlp_controls.copy()
                nlp_status = "FAILED(prev)"
            else:
                final_states = warm_states
                final_controls = warm_controls
                nlp_status = "FAILED(milp)"
        else:
            final_states = nlp_states
            final_controls = nlp_controls
            nlp_status = "OK"

        print(f"[Step {self._step_count:4d}] MILP: OK ({t_milp*1000:.1f}ms) | NLP: {nlp_status} ({t_nlp*1000:.1f}ms)")

        # Constraint analysis
        if not nlp_ok and nlp_debug is not None and nlp_debug[0] is not None:
            diag_states, diag_controls = nlp_debug
        else:
            diag_states, diag_controls = final_states, final_controls
        diag = self._second_stage.analyse_constraints(
            diag_states, diag_controls, road_left, road_right, obstacles,
            milp_ok=True, nlp_ok=nlp_ok, nlp_status=nlp_status,
            t_milp=t_milp, t_nlp=t_nlp,
        )
        self._last_diagnostics = diag

        self._prev_nlp_states = final_states.copy()
        self._prev_nlp_controls = final_controls.copy()

        self._last_rollout = self._frenet_trajectory_to_world(final_states)
        trajectory = self._last_rollout[:, :2]

        self._predicted_next_state = final_states[:2].copy()
        self._predicted_next_world = self._last_rollout[:2].copy()

        accel = float(final_controls[0, 0])
        steer = float(final_controls[0, 1])
        action = Action(
            acceleration=accel,
            steer_angle=steer,
            target_speed=self._target_speed,
        )
        return action, [trajectory], 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_rollout(self) -> Optional[np.ndarray]:
        return self._last_rollout

    @property
    def last_milp_rollout(self) -> Optional[np.ndarray]:
        return self._last_milp_rollout

    @property
    def last_obstacles(self) -> Optional[List]:
        return self._last_obstacles

    @property
    def last_other_agents(self) -> Optional[Dict]:
        return self._last_other_agents

    @property
    def frenet_frame(self) -> Optional[FrenetFrame]:
        return self._frenet

    @property
    def collision_margin(self) -> float:
        return self._collision_margin

    @property
    def ego_length(self) -> float:
        return self._ego_length

    @property
    def ego_width(self) -> float:
        return self._ego_width

    @property
    def milp_params(self) -> Dict:
        return self._first_stage.params

    @property
    def nlp_params(self) -> Dict:
        return self._second_stage.params

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def last_diagnostics(self) -> Optional[Dict]:
        return self._last_diagnostics

    @property
    def first_stage(self) -> FirstStagePlanner:
        return self._first_stage

    @property
    def second_stage(self) -> SecondStagePlanner:
        return self._second_stage

    def reset(self):
        """Reset all MPC state."""
        self._prev_milp_states = None
        self._prev_nlp_states = None
        self._prev_nlp_controls = None
        self._last_rollout = None
        self._last_milp_rollout = None
        self._last_diagnostics = None
        self._ref_start_idx = 0
        self._nlp_built = False
        self._nlp_solver = None
        self._last_obstacles = None
        self._last_other_agents = None
        self._step_count = 0
        self._first_stage.reset()
        self._second_stage.reset()

    # ------------------------------------------------------------------
    # Frenet state conversion
    # ------------------------------------------------------------------

    def _state_to_frenet(self, state: AgentState) -> np.ndarray:
        """Convert AgentState to Frenet state [s, d, phi, v]."""
        f = self._frenet.world_to_frenet(
            float(state.position[0]), float(state.position[1]),
            heading=float(state.heading),
        )
        return np.array([f['s'], f['d'], f['heading'], float(state.speed)])

    def _frenet_trajectory_to_world(self, frenet_states: np.ndarray) -> np.ndarray:
        """Convert (K, 4) Frenet states [s, d, phi, v] to world [x, y, heading, speed]."""
        K = len(frenet_states)
        world = np.empty((K, 4))
        for i in range(K):
            s_i, d_i, phi_i, v_i = frenet_states[i]
            w = self._frenet.frenet_to_world(s_i, d_i, heading=phi_i)
            world[i] = [w['x'], w['y'], w['heading'], v_i]
        return world

    def _milp_states_to_world(self, milp_states: np.ndarray) -> np.ndarray:
        """Convert (K, 4) MILP states [s, d, vs, vd] to world [x, y] positions."""
        K = len(milp_states)
        world = np.empty((K, 2))
        for i in range(K):
            s_i, d_i = milp_states[i, 0], milp_states[i, 1]
            w = self._frenet.frenet_to_world(s_i, d_i)
            world[i] = [w['x'], w['y']]
        return world

    # ------------------------------------------------------------------
    # Road boundary sampling
    # ------------------------------------------------------------------

    def _sample_road_boundaries(self, s_values: np.ndarray):
        """Sample road boundaries at given s positions."""
        if self._scenario_map is not None and self._frenet is not None:
            return self._frenet.road_boundaries(s_values, self._scenario_map,
                                                search_radius=15.0)
        return (np.full(len(s_values), 3.5),
                np.full(len(s_values), -3.5))

    # ------------------------------------------------------------------
    # Obstacle prediction
    # ------------------------------------------------------------------

    def _predict_obstacles(self, other_agents, frenet, agent_trajectories=None,
                           trajectory_dt=None):
        """Predict obstacle positions over the planning horizon.

        Uses provided trajectories if available, otherwise falls back to
        constant-velocity propagation.
        """
        if other_agents is None or frenet is None:
            return []

        H = self._horizon
        dt_plan = self._dt
        dt_traj = trajectory_dt if trajectory_dt is not None else self._dt_sim

        resample_ratio = dt_plan / dt_traj if dt_traj > 0 else 1.0

        obstacles = []

        for aid, agent_state in other_agents.items():
            pos = np.array(agent_state.position, dtype=float)
            vel = np.array(agent_state.velocity, dtype=float)

            uses_planned = False

            if agent_trajectories is not None and aid in agent_trajectories:
                provided_traj = np.asarray(agent_trajectories[aid], dtype=float)
                n_provided = len(provided_traj)

                world_positions = np.empty((H + 1, 2))
                for k in range(H + 1):
                    traj_idx_float = k * resample_ratio
                    traj_idx = int(traj_idx_float)

                    if traj_idx + 1 < n_provided:
                        alpha = traj_idx_float - traj_idx
                        world_positions[k] = (1 - alpha) * provided_traj[traj_idx] + alpha * provided_traj[traj_idx + 1]
                    elif traj_idx < n_provided:
                        world_positions[k] = provided_traj[traj_idx]
                    else:
                        time_beyond = (k * dt_plan) - ((n_provided - 1) * dt_traj)
                        world_positions[k] = provided_traj[-1] + vel * time_beyond

                uses_planned = True
            else:
                world_positions = np.empty((H + 1, 2))
                for k in range(H + 1):
                    world_positions[k] = pos + vel * k * dt_plan

            headings = np.empty(H + 1)
            headings[0] = float(agent_state.heading)
            for k in range(1, H + 1):
                dx = world_positions[k, 0] - world_positions[k - 1, 0]
                dy = world_positions[k, 1] - world_positions[k - 1, 1]
                if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                    headings[k] = np.arctan2(dy, dx)
                else:
                    headings[k] = headings[k - 1]

            s_arr, d_arr = frenet.world_to_frenet_batch(world_positions)

            obs_meta = getattr(agent_state, 'metadata', None)
            obs_length = obs_meta.length if obs_meta is not None else 4.5
            obs_width = obs_meta.width if obs_meta is not None else 1.8

            obstacles.append({
                's': s_arr,
                'd': d_arr,
                'world_positions': world_positions,
                'headings': headings,
                'length': obs_length,
                'width': obs_width,
                'heading': float(agent_state.heading),
                'agent_id': aid,
                'uses_planned_trajectory': uses_planned,
            })

        return obstacles

    # ------------------------------------------------------------------
    # MPC helpers
    # ------------------------------------------------------------------

    def _advance_reference_window(self, position: np.ndarray):
        """Move ``_ref_start_idx`` to the nearest waypoint ahead."""
        if len(self._reference_waypoints) == 0:
            return
        dists = np.linalg.norm(
            self._reference_waypoints[self._ref_start_idx:] - position, axis=1)
        self._ref_start_idx += int(np.argmin(dists))

    # ------------------------------------------------------------------
    # MILP -> NLP warm-start conversion
    # ------------------------------------------------------------------

    def _milp_to_nlp_warmstart(self, milp_states, frenet_state):
        """Convert MILP [s, d, vs, vd] to NLP [s, d, phi, v] + controls."""
        H = self._horizon
        dt = self._dt
        L = self._wheelbase
        nlp = self._second_stage.params

        s_arr = milp_states[:, 0]
        d_arr = milp_states[:, 1]
        vs_arr = milp_states[:, 2]
        vd_arr = milp_states[:, 3]

        speeds = np.sqrt(vs_arr**2 + vd_arr**2)
        phis = np.arctan2(vd_arr, vs_arr)
        phis[0] = frenet_state[2]

        nlp_states = np.column_stack([s_arr, d_arr, phis, speeds])
        nlp_controls = np.zeros((H, 2))

        for k in range(H):
            a_k = np.clip((speeds[k + 1] - speeds[k]) / dt,
                          nlp['a_min'], nlp['a_max'])

            d_phi = (phis[k + 1] - phis[k] + np.pi) % (2 * np.pi) - np.pi
            spd = max(speeds[k], 0.5)
            sin_arg = np.clip(d_phi * L / (2.0 * spd * dt), -1.0, 1.0)
            delta_k = np.clip(np.arcsin(sin_arg),
                              -nlp['delta_max'], nlp['delta_max'])

            nlp_controls[k] = [a_k, delta_k]

        return nlp_states, nlp_controls
