"""Belief inference via inverse planning.

Observes the ego driver's executed trajectory and infers which belief
configuration (visibility of other agents) best explains the behaviour.
Uses the first-stage planner to evaluate candidate belief configurations
by comparing planned trajectories against observed history.
"""

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from igp2.core.agentstate import AgentState
from igp2.beliefcontrol.frenet import FrenetFrame
from igp2.beliefcontrol.first_stage import FirstStagePlanner
from igp2.beliefcontrol.second_stage import SecondStagePlanner
from igp2.beliefcontrol.plotting import InferencePlotter, InterventionPlotter
from igp2.beliefcontrol.planning_utils import (
    milp_to_nlp_warmstart as _milp_to_nlp_warmstart,
    sample_road_boundaries as _sample_road_boundaries_util,
    predict_obstacles_cv as _predict_obstacles_cv_util,
)

logger = logging.getLogger(__name__)

INFERENCE_TYPES = ('naive',)


@dataclass
class HistoryEntry:
    """Single timestep of recorded ego and environment state."""
    frenet_state: np.ndarray               # [s, d, phi, v]
    other_agent_states: Dict[int, AgentState]  # snapshot of other agents
    human_action: Optional[tuple] = None   # (acceleration, steer_angle) before intervention


@dataclass
class InferenceResult:
    """Evaluation of one belief configuration."""
    config: Dict[int, bool]                # {agent_id: visible}
    pos_cost: float                        # mean position L2 (metres)
    vel_cost: float                        # mean velocity L2 (m/s)
    planned_sd: Optional[np.ndarray]       # (K, 2) planned [s, d]
    planned_vel: Optional[np.ndarray]      # (K, 2) planned [vs, vd]
    milp_ok: bool                          # True if MILP succeeded
    nlp_ok: bool                           # True if NLP succeeded (False = MILP fallback)
    nlp_states: Optional[np.ndarray] = None    # (H+1, 4) [s, d, phi, v]
    nlp_controls: Optional[np.ndarray] = None  # (H, 2) [a, delta]


class BeliefInference:
    """Infer driver beliefs by comparing observed trajectory to planned ones.

    For each subset of visible agents (belief configuration), solves the
    first-stage planner from a historical ego state with only the visible
    agents as obstacles.  The configuration whose planned trajectory most
    closely matches the observed trajectory is the inferred belief.

    Args:
        policy: TwoStagePolicy instance (used to extract planning config).
        scenario_map: Road layout for boundary queries.
        warmup_fraction: Fraction of planning horizon needed before
            inference starts.
        relevance_s_margin: Longitudinal margin for relevant agent
            detection (m).
        relevance_d_threshold: Maximum lateral offset for relevance (m).
    """

    RELEVANCE_METHODS = ('corridor', 'dual')

    def __init__(self, policy, scenario_map,
                 warmup_fraction: float = 0.2,
                 relevance_s_margin: float = 10.0,
                 relevance_d_threshold: float = 7.0,
                 boltzmann_beta: float = 12.5,
                 vel_weight: float = 1.0,
                 hidden_threshold: float = 0.6,
                 intervention_type: str = 'none',
                 w_agency: float = 1.0,
                 inference_type: str = 'naive',
                 relevance_method: str = 'dual',
                 plot: bool = True):
        if relevance_method not in self.RELEVANCE_METHODS:
            raise ValueError(
                f"Unknown relevance_method {relevance_method!r}. "
                f"Choose from: {self.RELEVANCE_METHODS}")
        self._relevance_method = relevance_method
        if inference_type not in INFERENCE_TYPES:
            raise ValueError(
                f"Unknown inference_type {inference_type!r}. "
                f"Choose from: {INFERENCE_TYPES}")
        self._inference_type = inference_type
        self._scenario_map = scenario_map
        self._warmup_fraction = warmup_fraction
        self._boltzmann_beta = boltzmann_beta
        self._vel_weight = vel_weight
        self._hidden_threshold = hidden_threshold
        self._relevance_s_margin = relevance_s_margin
        self._relevance_d_threshold = relevance_d_threshold

        # Extract planning parameters from the policy
        self._frenet: Optional[FrenetFrame] = policy.frenet_frame
        self._horizon = policy.horizon
        self._dt = policy.dt
        self._target_speed = policy.target_speed
        self._ego_length = policy.ego_length
        self._ego_width = policy.ego_width
        self._collision_margin = policy.collision_margin
        self._fps = policy.fps
        self._dt_sim = 1.0 / self._fps

        # Warmup: number of planning steps worth of history needed
        self._warmup_steps = max(1, int(self._warmup_fraction * self._horizon))

        self._wheelbase = policy.wheelbase

        # Create own FirstStagePlanner to avoid corrupting the policy's
        self._first_stage = FirstStagePlanner(
            horizon=self._horizon,
            dt=self._dt,
            ego_length=self._ego_length,
            ego_width=self._ego_width,
            collision_margin=self._collision_margin,
            target_speed=self._target_speed,
            frenet=self._frenet,
            params=policy.milp_params,
            n_obs_max=policy.first_stage._n_obs_max,
        )

        # Create own SecondStagePlanner for two-stage inference
        self._second_stage = SecondStagePlanner(
            horizon=self._horizon,
            dt=self._dt,
            ego_length=self._ego_length,
            ego_width=self._ego_width,
            wheelbase=self._wheelbase,
            collision_margin=self._collision_margin,
            target_speed=self._target_speed,
            frenet=self._frenet,
            params=policy.nlp_params,
            n_obs_max=policy.first_stage._n_obs_max,
        )

        # Intervention configuration
        self._intervention_type = intervention_type
        self._w_agency = w_agency

        # History buffer
        self._history: List[HistoryEntry] = []
        self._last_results: Optional[List[InferenceResult]] = None
        self._last_observed_sd: Optional[np.ndarray] = None
        self._last_intervention = None  # dict or None
        self._last_marginals: Dict[int, float] = {}  # P(hidden | τ_obs) per agent
        self._last_config_probs: List[float] = []     # P(b | τ_obs) per config
        self._last_energies: List[float] = []          # E(b) per config

        # Debug plotters
        self._plotter: Optional[InferencePlotter] = None
        self._intervention_plotter: Optional[InterventionPlotter] = None
        if plot and self._frenet is not None:
            self._plotter = InferencePlotter(
                scenario_map, policy.reference_waypoints, self._frenet,
                dt=self._dt,
                ego_length=self._ego_length, ego_width=self._ego_width)
            if intervention_type != 'none':
                self._intervention_plotter = InterventionPlotter(
                    scenario_map, policy.reference_waypoints, self._frenet,
                    dt=self._dt,
                    ego_length=self._ego_length, ego_width=self._ego_width,
                    collision_margin=self._collision_margin)

    def reset(self):
        """Clear history and results."""
        self._history = []
        self._last_results = None
        self._last_marginals = {}
        self._last_config_probs = []
        self._last_energies = []
        self._last_intervention = None
        self._first_stage.reset()
        self._second_stage.reset()

    def step(self, frenet_state: np.ndarray,
             other_agent_states: Dict[int, AgentState],
             step_count: int,
             ego_position: np.ndarray = None,
             human_action: Optional[tuple] = None,
             true_obstacles: Optional[list] = None,
             true_policy_result: Optional[tuple] = None,
             human_policy_result: Optional[tuple] = None,
             active_agents: Optional[Dict[int, float]] = None):
        """Run one inference step.

        Args:
            frenet_state: Current ego Frenet state [s, d, phi, v].
            other_agent_states: Snapshot of all other agents' states.
            step_count: Current simulation step number (for display).
            ego_position: Current ego world position [x, y] (for plot centring).
            human_action: Raw (acceleration, steer_angle) from the human policy
                before any intervention override.
            true_obstacles: Pre-built obstacle list from the true policy
                (same prediction used for true planning).  If provided, the
                intervention uses these directly so obstacle prediction is
                identical to the true policy.
            true_policy_result: (states, controls) from the true policy's last
                solve.  For ``policy_only`` mode this is used directly as the
                intervention result, avoiding a redundant MILP→NLP.
            human_policy_result: (states, controls) from the human policy's
                last solve.  For ``policy_only`` mode this is shown as the
                believed trajectory in the intervention plotter.
            active_agents: Per-agent dual influence {agent_id: Σ|λ|} from
                the true policy's NLP solve.  Used for dual-based relevance
                detection.
        """
        # 1. Append to history
        self._history.append(HistoryEntry(
            frenet_state=frenet_state.copy(),
            other_agent_states={aid: s for aid, s in other_agent_states.items()},
            human_action=human_action,
        ))

        # 2. Check warmup
        # Convert warmup from planning steps to sim steps
        sim_steps_per_plan_step = max(1, int(round(self._dt / self._dt_sim)))
        warmup_sim_steps = self._warmup_steps * sim_steps_per_plan_step
        if len(self._history) < warmup_sim_steps:
            return

        if self._frenet is None:
            return

        # 3. Fixed observation window: always compare the first warmup_steps
        # of the planned trajectory, but plan the full horizon so forward
        # obstacles influence the trajectory shape even in the compared portion.
        window_sim_steps = self._warmup_steps * sim_steps_per_plan_step
        start_idx = len(self._history) - window_sim_steps

        # Historical ego state at the start of the window
        hist_frenet = self._history[start_idx].frenet_state
        hist_agents = self._history[start_idx].other_agent_states

        # Forward-simulate human-intended trajectory from window start
        observed_sd = self._simulate_human_trajectory(start_idx, len(self._history) - 1,
                                                      sim_steps_per_plan_step)

        # 4. Find relevant agents
        relevant_aids = self._find_relevant_agents(
            frenet_state, other_agent_states, active_agents=active_agents)

        # 5. Evaluate belief configs (only if there are relevant agents)
        results: List[InferenceResult] = []
        t_elapsed = 0.0

        if not relevant_aids:
            # No dynamic agents relevant — clear any stale intervention
            self._last_intervention = None

        if relevant_aids:
            if self._inference_type == 'naive':
                results, marginals, t_elapsed = self._run_naive_inference(
                    hist_frenet, hist_agents, observed_sd,
                    current_frenet=frenet_state,
                    relevant_aids=relevant_aids,
                    step_count=step_count)
            else:
                raise ValueError(f"Unknown inference type: {self._inference_type}")

            self._last_marginals = marginals

            # Compute ego world heading from Frenet state
            w = self._frenet.frenet_to_world(
                frenet_state[0], frenet_state[1], heading=frenet_state[2])
            ego_heading = w['heading']

            # Compute minimum-deviation intervention from current state
            self._compute_intervention(
                marginals, frenet_state, other_agent_states,
                relevant_aids,
                ego_position=ego_position, step_count=step_count,
                ego_heading=ego_heading,
                true_obstacles=true_obstacles,
                true_policy_result=true_policy_result,
                human_policy_result=human_policy_result)

        self._last_results = results
        self._last_observed_sd = observed_sd

        # 6. Update debug plot (always, even with no candidates)
        if self._plotter is not None and ego_position is not None:
            # Compute ego world heading from Frenet state (reuse if computed above)
            if not relevant_aids:
                w = self._frenet.frenet_to_world(
                    frenet_state[0], frenet_state[1], heading=frenet_state[2])
                ego_heading = w['heading']
            self._plotter.update(
                observed_sd, results, ego_position, step_count,
                other_agent_states=other_agent_states,
                marginals=marginals if relevant_aids else {},
                ego_heading=ego_heading)

    def _run_naive_inference(self, hist_frenet: np.ndarray,
                             hist_agents: Dict[int, AgentState],
                             observed_sd: np.ndarray,
                             current_frenet: np.ndarray,
                             relevant_aids: List[int],
                             step_count: int,
                             ) -> tuple:
        """Exhaustive 2^N enumeration of belief configs with Boltzmann posteriors.

        Returns:
            (results, marginals, elapsed_time) where *results* is a sorted list
            of :class:`InferenceResult`, *marginals* is ``{aid: P(hidden)}``,
            and *elapsed_time* is wall-clock seconds for the solve loop.
        """
        configs = self._enumerate_configs(relevant_aids)

        # Sample road boundaries from the historical start state
        s_values = np.array([
            hist_frenet[0] + hist_frenet[3] * np.cos(hist_frenet[2]) * k * self._dt
            for k in range(self._horizon + 1)
        ])
        s_values = np.clip(s_values, 0.0, self._frenet.total_length)
        road_left, road_right = self._sample_road_boundaries(s_values)

        results: List[InferenceResult] = []
        t_start = time.perf_counter()

        for config in configs:
            visible_aids = {aid for aid, vis in config.items() if vis}

            # Build obstacles with constant-velocity prediction
            obstacles = self._predict_obstacles_cv(
                hist_agents, visible_aids)

            # --- Stage 1: MILP ---
            self._first_stage.reset()
            milp_states = self._first_stage.solve(
                hist_frenet, road_left, road_right, obstacles)

            if milp_states is None:
                results.append(InferenceResult(
                    config=config,
                    pos_cost=float('inf'), vel_cost=float('inf'),
                    planned_sd=None, planned_vel=None,
                    milp_ok=False, nlp_ok=False,
                    nlp_states=None, nlp_controls=None,
                ))
                continue

            # --- Stage 2: NLP (bicycle model) ---
            warm_states, warm_controls = self._milp_to_nlp_warmstart(
                milp_states, hist_frenet)

            self._second_stage.reset()
            nlp_states, nlp_controls, nlp_ok, _ = self._second_stage.solve(
                hist_frenet, warm_states, warm_controls,
                road_left, road_right, obstacles)

            if nlp_ok:
                # NLP states are [s, d, phi, v] → convert to [s, d, vs, vd]
                planned_sdvv = np.column_stack([
                    nlp_states[:, 0],                                     # s
                    nlp_states[:, 1],                                     # d
                    nlp_states[:, 3] * np.cos(nlp_states[:, 2]),          # vs
                    nlp_states[:, 3] * np.sin(nlp_states[:, 2]),          # vd
                ])
            else:
                # Fallback to MILP-derived warmstart
                planned_sdvv = np.column_stack([
                    warm_states[:, 0],
                    warm_states[:, 1],
                    warm_states[:, 3] * np.cos(warm_states[:, 2]),
                    warm_states[:, 3] * np.sin(warm_states[:, 2]),
                ])

            pos_cost, vel_cost = self._trajectory_cost(
                observed_sd, planned_sdvv)
            results.append(InferenceResult(
                config=config,
                pos_cost=pos_cost, vel_cost=vel_cost,
                planned_sd=planned_sdvv[:, :2],
                planned_vel=planned_sdvv[:, 2:4],
                milp_ok=True, nlp_ok=nlp_ok,
                nlp_states=nlp_states if nlp_ok else warm_states,
                nlp_controls=nlp_controls if nlp_ok else warm_controls,
            ))

        t_elapsed = time.perf_counter() - t_start

        # Sort by position cost
        results.sort(key=lambda r: r.pos_cost)

        # Print results and get marginal posteriors
        marginals = self._print_results(results, step_count, relevant_aids,
                                        t_elapsed, self._warmup_steps)

        return results, marginals, t_elapsed

    def _find_relevant_agents(self, frenet_state: np.ndarray,
                              other_agent_states: Dict[int, AgentState],
                              active_agents: Optional[Dict[int, float]] = None,
                              ) -> List[int]:
        """Find agents that are relevant for belief inference enumeration.

        Dispatches to either dual-based or corridor-based detection depending
        on ``self._relevance_method``.

        Args:
            frenet_state: Current ego Frenet state [s, d, phi, v].
            other_agent_states: Snapshot of all other agents' states.
            active_agents: Per-agent dual influence {agent_id: Σ|λ|} from
                the true policy's NLP solve (only used for 'dual' method).
        """
        if self._relevance_method == 'dual':
            if active_agents is not None:
                return sorted([
                    aid for aid, influence in active_agents.items()
                    if influence > 1e-4 and aid >= 0
                ])
            # Fall back to corridor if dual info not yet available
            # (e.g. first steps before the true policy has solved)
            logger.debug("Dual relevance requested but active_agents is None, "
                         "falling back to corridor method")

        return self._find_relevant_agents_corridor(frenet_state, other_agent_states)

    def _find_relevant_agents_corridor(self, frenet_state: np.ndarray,
                                        other_agent_states: Dict[int, AgentState],
                                        ) -> List[int]:
        """Find agents whose predicted trajectories enter the ego corridor.

        Forward-simulates each agent via constant-velocity prediction,
        converts to Frenet, and checks overlap with the ego's driving
        corridor (road boundaries + lateral margin).
        """
        if self._frenet is None:
            return []

        # Ego corridor: forward-simulate ego position along the reference
        ego_s = frenet_state[0]
        vs = frenet_state[3] * np.cos(frenet_state[2])
        s_ego = np.array([ego_s + vs * k * self._dt
                          for k in range(self._horizon + 1)])
        s_ego = np.clip(s_ego, 0.0, self._frenet.total_length)
        s_min = s_ego[0] - self._relevance_s_margin
        s_max = s_ego[-1] + self._relevance_s_margin

        # Road boundaries at ego s-positions define the lateral corridor
        d_left, d_right = self._sample_road_boundaries(s_ego)
        d_margin = self._relevance_d_threshold

        relevant = []
        times = np.arange(self._horizon + 1).reshape(-1, 1) * self._dt

        for aid, state in other_agent_states.items():
            # Skip static objects (negative IDs are parked vehicles/barriers)
            if aid < 0:
                continue

            # Constant-velocity prediction in world frame
            pos = np.array(state.position[:2], dtype=float)
            vel = np.array(state.velocity[:2], dtype=float)
            world_traj = pos + vel * times  # (H+1, 2)

            # Convert to Frenet
            s_agent, d_agent = self._frenet.world_to_frenet_batch(world_traj)

            # Check if any predicted point falls inside the corridor
            for k in range(len(s_agent)):
                sk, dk = s_agent[k], d_agent[k]
                if not (s_min <= sk <= s_max):
                    continue
                # Interpolate road boundaries at this agent s-position
                dl = np.interp(sk, s_ego, d_left)
                dr = np.interp(sk, s_ego, d_right)
                if dr - d_margin <= dk <= dl + d_margin:
                    relevant.append(aid)
                    break

        return sorted(relevant)

    def _enumerate_configs(self, agent_ids: List[int]
                           ) -> List[Dict[int, bool]]:
        """Generate all 2^N visibility combinations."""
        configs = []
        for combo in itertools.product([True, False], repeat=len(agent_ids)):
            configs.append(dict(zip(agent_ids, combo)))
        return configs

    def _predict_obstacles_cv(self, agent_states: Dict[int, AgentState],
                              visible_aids: set) -> list:
        """Constant-velocity obstacle prediction for visible agents only."""
        return _predict_obstacles_cv_util(
            agent_states, visible_aids, self._frenet,
            self._horizon, self._dt)

    def _subsample_history(self, start_idx: int, end_idx: int,
                           sim_steps_per_plan_step: int) -> np.ndarray:
        """Subsample history from sim dt to planning dt.

        Returns (K, 4) array of [s, d, vs, vd] at planning timesteps.
        """
        indices = list(range(start_idx, end_idx + 1,
                             sim_steps_per_plan_step))
        observed = np.empty((len(indices), 4))
        for i, idx in enumerate(indices):
            fs = self._history[idx].frenet_state  # [s, d, phi, v]
            observed[i, 0] = fs[0]  # s
            observed[i, 1] = fs[1]  # d
            observed[i, 2] = fs[3] * np.cos(fs[2])  # vs = v * cos(phi)
            observed[i, 3] = fs[3] * np.sin(fs[2])  # vd = v * sin(phi)
        return observed

    def _simulate_human_trajectory(self, start_idx: int, end_idx: int,
                                    sim_steps_per_plan_step: int) -> np.ndarray:
        """Forward-simulate from window start using stored human actions.

        Uses the NLP bicycle model in Frenet coordinates to replay what the
        human *intended* (before intervention override), then subsamples at
        planning dt.

        Dynamics (matching second_stage.py and carla_client._bicycle_step):
            s_{k+1}   = s_k   + v_k * cos(phi_k + delta_k) * dt_sim
            d_{k+1}   = d_k   + v_k * sin(phi_k + delta_k) * dt_sim
            phi_{k+1} = phi_k + (2*v_k/L) * sin(delta_k) * dt_sim
            v_{k+1}   = max(0, v_k + a_k * dt_sim)

        Returns (K, 4) array of [s, d, vs, vd] at planning timesteps.
        """
        # Initial Frenet state at window start
        fs0 = self._history[start_idx].frenet_state  # [s, d, phi, v]
        s, d, phi, v = float(fs0[0]), float(fs0[1]), float(fs0[2]), float(fs0[3])

        dt_sim = self._dt_sim
        L = self._wheelbase

        # Collect sim-step states: initial + one per sim step
        n_sim_steps = end_idx - start_idx
        # Store [s, d, phi, v] at each sim step (including initial)
        states = np.empty((n_sim_steps + 1, 4))
        states[0] = [s, d, phi, v]

        for k in range(n_sim_steps):
            hist_entry = self._history[start_idx + k]
            ha = hist_entry.human_action
            if ha is not None:
                a_k, delta_k = float(ha[0]), float(ha[1])
            else:
                a_k, delta_k = 0.0, 0.0

            s_new = s + v * np.cos(phi + delta_k) * dt_sim
            d_new = d + v * np.sin(phi + delta_k) * dt_sim
            phi_new = phi + (2.0 * v / L) * np.sin(delta_k) * dt_sim
            v_new = max(0.0, v + a_k * dt_sim)

            # Wrap heading to [-pi, pi]
            phi_new = (phi_new + np.pi) % (2.0 * np.pi) - np.pi

            s, d, phi, v = s_new, d_new, phi_new, v_new
            states[k + 1] = [s, d, phi, v]

        # Subsample at planning dt intervals
        indices = list(range(0, n_sim_steps + 1, sim_steps_per_plan_step))
        observed = np.empty((len(indices), 4))
        for i, idx in enumerate(indices):
            st = states[idx]
            observed[i, 0] = st[0]                          # s
            observed[i, 1] = st[1]                          # d
            observed[i, 2] = st[3] * np.cos(st[2])          # vs = v * cos(phi)
            observed[i, 3] = st[3] * np.sin(st[2])          # vd = v * sin(phi)
        return observed

    def _milp_to_nlp_warmstart(self, milp_states, frenet_state):
        """Convert MILP [s, d, vs, vd] to NLP [s, d, phi, v] + controls."""
        return _milp_to_nlp_warmstart(
            milp_states, frenet_state,
            self._horizon, self._dt, self._wheelbase,
            self._second_stage.params)

    def _trajectory_cost(self, observed_sdvv: np.ndarray,
                         planned_sdvv: np.ndarray
                         ) -> tuple:
        """Mean L2 distance in (s, d) and (vs, vd) space.

        Compares min(len(observed), len(planned)) steps.

        Returns:
            (pos_cost, vel_cost) tuple.
        """
        n = min(len(observed_sdvv), len(planned_sdvv))
        if n == 0:
            return float('inf'), float('inf')

        # Position error
        pos_diffs = observed_sdvv[:n, :2] - planned_sdvv[:n, :2]
        pos_dists = np.sqrt(pos_diffs[:, 0] ** 2 + pos_diffs[:, 1] ** 2)

        # Velocity error
        vel_diffs = observed_sdvv[:n, 2:4] - planned_sdvv[:n, 2:4]
        vel_dists = np.sqrt(vel_diffs[:, 0] ** 2 + vel_diffs[:, 1] ** 2)

        return float(np.mean(pos_dists)), float(np.mean(vel_dists))

    def _sample_road_boundaries(self, s_values: np.ndarray):
        """Sample road boundaries at given s positions."""
        return _sample_road_boundaries_util(s_values, self._scenario_map, self._frenet)

    def _print_results(self, results: List[InferenceResult],
                       step_count: int, relevant_aids: List[int],
                       elapsed: float, window_steps: int) -> Dict[int, float]:
        """Pretty-print inference results table sorted by cost.

        Returns:
            Marginal posteriors {agent_id: P(h_i | τ_obs)} for each agent.
        """
        n_configs = len(results)
        aids_str = ", ".join(str(a) for a in relevant_aids)

        logger.info("[Step %4d] BELIEF INFERENCE "
                    "(%d relevant agents [%s], %d configs, window=%d steps, %.2fs):",
                    step_count, len(relevant_aids), aids_str,
                    n_configs, window_steps, elapsed)

        # Boltzmann rational-action model for belief inference:
        #
        # b     = full belief config (e.g. {1:V, 2:H})
        # τ_obs = observed driver trajectory
        # τ*(b) = optimal trajectory under belief b (from two-stage planner)
        #
        # Energy:    E(b) = cost(τ_obs, τ*(b))
        #                  = pos_error + w_vel * vel_error
        # Likelihood (Boltzmann): P(τ_obs | b) ∝ exp(-β · E(b))
        # Posterior (uniform prior): P(b | τ_obs) = P(τ_obs | b) / Z
        #   where Z = Σ_b exp(-β · E(b))   (partition function)
        #
        # Marginal posterior that agent i is hidden:
        #   P(h_i | τ_obs) = Σ_{b : i hidden in b} P(b | τ_obs)
        #
        # β (inverse temperature) controls rationality: β→∞ assumes perfectly
        # rational driver, β→0 gives uniform distribution over beliefs.
        beta = self._boltzmann_beta
        w_vel = self._vel_weight

        energies = np.array([
            r.pos_cost + w_vel * r.vel_cost
            if r.milp_ok and np.isfinite(r.pos_cost) and np.isfinite(r.vel_cost)
            else 1e10
            for r in results
        ])
        log_probs = -beta * energies
        log_probs -= np.max(log_probs)  # numerical stability (shift by log Z')
        boltzmann_probs = np.exp(log_probs)
        Z = np.sum(boltzmann_probs)
        boltzmann_probs = boltzmann_probs / Z if Z > 0 else np.zeros(len(results))

        # Store for external access
        self._last_energies = energies.tolist()
        self._last_config_probs = boltzmann_probs.tolist()

        best_energy = np.min(energies) if len(energies) > 0 else float('inf')
        for i, r in enumerate(results):
            cfg_str = ", ".join(
                f"{aid}:{'V' if vis else 'H'}"
                for aid, vis in sorted(r.config.items())
            )
            if not r.milp_ok:
                status = "MILP_FAIL"
            elif not r.nlp_ok:
                status = "NLP_FAIL"
            else:
                status = "OK"
            marker = "  <-- best" if energies[i] == best_energy and r.milp_ok else ""
            logger.info("  {%s}: E=%.4f (pos=%.4f vel=%.4f)  P(b|tau)=%.3f  [%s]%s",
                        cfg_str, energies[i], r.pos_cost, r.vel_cost,
                        boltzmann_probs[i], status, marker)

        # Marginal posterior: P(h_i | τ_obs) = Σ_{b : i hidden in b} P(b | τ_obs)
        marginals = {}
        if results and len(boltzmann_probs) > 0:
            for i, r in enumerate(results):
                for aid, vis in r.config.items():
                    if aid not in marginals:
                        marginals[aid] = 0.0
                    if not vis:
                        marginals[aid] += boltzmann_probs[i]
            logger.info("  P(h_i|tau): %s", ", ".join(
                f"{aid}={marginals[aid]:.3f}" for aid in sorted(marginals)))
        return marginals

    def _compute_intervention(self, marginals: Dict[int, float],
                               frenet_state: np.ndarray,
                               other_agent_states: Dict[int, AgentState],
                               relevant_aids: List[int],
                               ego_position: np.ndarray = None,
                               step_count: int = 0,
                               ego_heading: float = 0.0,
                               true_obstacles: Optional[list] = None,
                               true_policy_result: Optional[tuple] = None,
                               human_policy_result: Optional[tuple] = None):
        """Compute minimum-deviation intervention from the CURRENT state.

        Re-plans the believed trajectory from the current ego state (not the
        historical window start), then solves the intervention NLP from the
        same current state so both trajectories stay anchored to the ego.

        The ``true_obstacles`` list should come from the true policy so that
        the intervention sees exactly the same obstacle predictions.

        1. Threshold marginals → believed config
        2. Sample road boundaries from current state
        3. (non-policy_only) Two-stage plan for believed trajectory
        4. Solve intervention (or replan for policy_only) with true obstacles
        5. Store results and update plotter
        6. Print diagnostics
        """
        if self._intervention_type == 'none':
            self._last_intervention = None
            return

        # 1. Threshold marginals to get believed config
        believed_config = {}
        for aid in relevant_aids:
            p_hidden = marginals.get(aid, 0.0)
            believed_config[aid] = p_hidden <= self._hidden_threshold  # visible

        # Skip if all agents are believed visible — no intervention needed
        if all(believed_config.values()):
            self._last_intervention = None
            return

        # 2. Sample road boundaries from current state
        s_values = np.array([
            frenet_state[0] + frenet_state[3] * np.cos(frenet_state[2]) * k * self._dt
            for k in range(self._horizon + 1)
        ])
        s_values = np.clip(s_values, 0.0, self._frenet.total_length)
        road_left, road_right = self._sample_road_boundaries(s_values)

        # Use pre-built obstacles from the true policy so prediction is
        # identical.  Fall back to constant-velocity if not provided.
        if true_obstacles is None:
            all_dynamic_aids = {aid for aid in other_agent_states if aid >= 0}
            true_obstacles = self._predict_obstacles_cv(
                other_agent_states, all_dynamic_aids)

        cfg_str = ", ".join(
            f"{aid}:{'V' if vis else 'H'}"
            for aid, vis in sorted(believed_config.items()))

        if self._intervention_type == 'policy_only':
            # Reuse the true policy's last NLP result directly — no redundant
            # MILP→NLP solve.  The true policy already planned with all agents
            # and identical conditions, so its result is the optimal
            # intervention for the "policy_only" scheme.
            if true_policy_result is not None:
                opt_states, opt_controls = true_policy_result
                success = True
            else:
                logger.warning("  policy_only: no true_policy_result provided, skipping")
                self._last_intervention = None
                return

            # Use the human policy's result as the believed reference (red)
            # so the plotter shows the meaningful comparison:
            #   red  = what the human planned (with missing agents)
            #   blue = what the true policy planned (with all agents)
            if human_policy_result is not None:
                ref_states, ref_controls = human_policy_result
            else:
                # Fallback: both are the true policy result
                ref_states, ref_controls = opt_states, opt_controls

            intervention = opt_controls - ref_controls

            self._last_intervention = {
                'believed_config': believed_config,
                'ref_states': ref_states,
                'ref_controls': ref_controls,
                'opt_states': opt_states,
                'opt_controls': opt_controls,
                'intervention': intervention,
                'success': success,
                'true_obstacles': true_obstacles,
            }

            if self._intervention_plotter is not None and ego_position is not None:
                self._intervention_plotter.update(
                    ref_states=ref_states,
                    ref_controls=ref_controls,
                    opt_states=opt_states,
                    opt_controls=opt_controls,
                    intervention=intervention,
                    success=success,
                    believed_config=believed_config,
                    true_obstacles=true_obstacles,
                    ego_position=ego_position,
                    ego_heading=ego_heading,
                    other_agent_states=other_agent_states or {},
                    step=step_count,
                    dt=self._dt)

            logger.info("  Believed config: {%s}", cfg_str)
            logger.info("  Intervention (policy_only): reused true policy result")
            return

        # --- agency_only / combined: full MILP → NLP with true obstacles ---

        # 3. Plan believed trajectory (reference for agency term)
        visible_aids = {aid for aid, vis in believed_config.items() if vis}
        believed_obstacles = self._predict_obstacles_cv(
            other_agent_states, visible_aids)

        self._first_stage.reset()
        milp_states = self._first_stage.solve(
            frenet_state, road_left, road_right, believed_obstacles)

        ref_controls = None
        nlp_states = None

        if milp_states is not None:
            warm_states, warm_controls = self._milp_to_nlp_warmstart(
                milp_states, frenet_state)

            self._second_stage.reset()
            nlp_states, nlp_controls, nlp_ok, _ = self._second_stage.solve(
                frenet_state, warm_states, warm_controls,
                road_left, road_right, believed_obstacles)

            if not nlp_ok:
                nlp_states = warm_states
                nlp_controls = warm_controls

            ref_controls = nlp_controls
        else:
            logger.info("  Believed config: {%s} -- believed MILP failed, "
                        "falling back to pure tracking", cfg_str)

        # 4. Full MILP → NLP with true obstacles + agency term
        logger.info("  INTERVENTION DEBUG: type=%s, n_true_obs=%d, "
                     "ref_controls=%s, w_agency=%.2f, frenet=[s=%.2f d=%.2f phi=%.3f v=%.2f]",
                     self._intervention_type, len(true_obstacles),
                     "provided" if ref_controls is not None else "None",
                     self._w_agency,
                     frenet_state[0], frenet_state[1],
                     frenet_state[2], frenet_state[3])
        if ref_controls is not None:
            logger.info("  INTERVENTION DEBUG: ref_controls[0]=[a=%.4f, δ=%.4f], "
                         "max|ref_a|=%.4f, max|ref_δ|=%.4f",
                         ref_controls[0, 0], ref_controls[0, 1],
                         np.max(np.abs(ref_controls[:, 0])),
                         np.max(np.abs(ref_controls[:, 1])))

        self._first_stage.reset()
        true_milp_states = self._first_stage.solve(
            frenet_state, road_left, road_right, true_obstacles)

        if true_milp_states is None:
            logger.info("  Believed config: {%s} -- true MILP failed, skipping intervention", cfg_str)
            self._last_intervention = None
            return

        logger.info("  INTERVENTION DEBUG: true MILP OK, "
                     "milp d range=[%.3f, %.3f]",
                     float(np.min(true_milp_states[:, 1])),
                     float(np.max(true_milp_states[:, 1])))

        true_warm_states, true_warm_controls = self._milp_to_nlp_warmstart(
            true_milp_states, frenet_state)

        logger.info("  INTERVENTION DEBUG: warm_controls[0]=[a=%.4f, δ=%.4f], "
                     "max|warm_δ|=%.4f",
                     true_warm_controls[0, 0], true_warm_controls[0, 1],
                     np.max(np.abs(true_warm_controls[:, 1])))

        self._second_stage.reset()
        opt_states, opt_controls, success, _ = self._second_stage.solve(
            frenet_state, true_warm_states, true_warm_controls,
            road_left, road_right, true_obstacles,
            ref_controls=ref_controls,
            w_agency=self._w_agency,
            agency_only=(self._intervention_type == 'agency_only'))

        logger.info("  INTERVENTION DEBUG: NLP success=%s", success)
        if success:
            logger.info("  INTERVENTION DEBUG: opt_controls[0]=[a=%.4f, δ=%.4f], "
                         "max|opt_a|=%.4f, max|opt_δ|=%.4f, "
                         "opt d range=[%.3f, %.3f]",
                         opt_controls[0, 0], opt_controls[0, 1],
                         np.max(np.abs(opt_controls[:, 0])),
                         np.max(np.abs(opt_controls[:, 1])),
                         float(np.min(opt_states[:, 1])),
                         float(np.max(opt_states[:, 1])))

        if not success:
            intervention = np.zeros_like(true_warm_controls)
            opt_states = true_warm_states
            opt_controls = true_warm_controls
            ref_controls = ref_controls if ref_controls is not None else true_warm_controls
            nlp_states = nlp_states if nlp_states is not None else true_warm_states
        else:
            if ref_controls is None:
                ref_controls = opt_controls
                nlp_states = opt_states
            intervention = opt_controls - ref_controls

        # 6. Store results
        self._last_intervention = {
            'believed_config': believed_config,
            'ref_states': nlp_states,
            'ref_controls': ref_controls,
            'opt_states': opt_states,
            'opt_controls': opt_controls,
            'intervention': intervention,
            'success': success,
            'true_obstacles': true_obstacles,
        }

        # Update plotter
        if self._intervention_plotter is not None and ego_position is not None:
            self._intervention_plotter.update(
                ref_states=nlp_states,
                ref_controls=ref_controls,
                opt_states=opt_states,
                opt_controls=opt_controls,
                intervention=intervention,
                success=success,
                believed_config=believed_config,
                true_obstacles=true_obstacles,
                ego_position=ego_position,
                ego_heading=ego_heading,
                other_agent_states=other_agent_states or {},
                step=step_count,
                dt=self._dt)

        # 7. Print diagnostics
        if success:
            du_norm = float(np.linalg.norm(intervention))
            max_da = float(np.max(np.abs(intervention[:, 0])))
            max_dd = float(np.max(np.abs(intervention[:, 1])))
            logger.info("  Believed config: {%s}", cfg_str)
            logger.info("  Intervention: OK, ||du||=%.4f, max|da|=%.4f, max|dd|=%.4f",
                        du_norm, max_da, max_dd)
        else:
            logger.info("  Believed config: {%s}", cfg_str)
            logger.info("  Intervention: FAILED (returning reference trajectory)")

    @property
    def last_results(self) -> Optional[List[InferenceResult]]:
        """Most recent inference results, sorted by cost."""
        return self._last_results

    @property
    def last_intervention(self):
        """Most recent intervention results dict, or None."""
        return self._last_intervention

    @property
    def last_marginals(self) -> Dict[int, float]:
        """Most recent marginal posteriors {agent_id: P(hidden | τ_obs)}."""
        return self._last_marginals

    @property
    def last_config_probs(self) -> List[float]:
        """P(b | τ_obs) for each config, aligned with ``last_results``."""
        return self._last_config_probs

    @property
    def last_energies(self) -> List[float]:
        """Energy E(b) for each config, aligned with ``last_results``."""
        return self._last_energies

    @property
    def history_length(self) -> int:
        """Number of history entries stored."""
        return len(self._history)
