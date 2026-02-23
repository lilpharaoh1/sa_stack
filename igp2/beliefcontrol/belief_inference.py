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
from igp2.beliefcontrol.secondstage_intervention import SecondStageIntervention
from igp2.beliefcontrol.plotting import InferencePlotter, InterventionPlotter
from igp2.beliefcontrol.planning_utils import (
    milp_to_nlp_warmstart as _milp_to_nlp_warmstart,
    sample_road_boundaries as _sample_road_boundaries_util,
    predict_obstacles_cv as _predict_obstacles_cv_util,
)

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """Single timestep of recorded ego and environment state."""
    frenet_state: np.ndarray               # [s, d, phi, v]
    other_agent_states: Dict[int, AgentState]  # snapshot of other agents


@dataclass
class InferenceResult:
    """Evaluation of one belief configuration."""
    config: Dict[int, bool]                # {agent_id: visible}
    pos_cost: float                        # mean position L2 (metres)
    vel_cost: float                        # mean velocity L2 (m/s)
    planned_sd: Optional[np.ndarray]       # (K, 2) planned [s, d]
    planned_vel: Optional[np.ndarray]      # (K, 2) planned [vs, vd]
    solver_ok: bool
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
        max_relevant: Cap on number of agents to enumerate (limits 2^N).
    """

    def __init__(self, policy, scenario_map,
                 warmup_fraction: float = 0.2,
                 relevance_s_margin: float = 10.0,
                 relevance_d_threshold: float = 7.0,
                 max_relevant: int = 5,
                 boltzmann_beta: float = 12.5,
                 vel_weight: float = 1.0,
                 hidden_threshold: float = 0.6,
                 intervention_type: str = 'none',
                 w_agency: float = 1.0,
                 plot: bool = True):
        self._scenario_map = scenario_map
        self._warmup_fraction = warmup_fraction
        self._boltzmann_beta = boltzmann_beta
        self._vel_weight = vel_weight
        self._hidden_threshold = hidden_threshold
        self._relevance_s_margin = relevance_s_margin
        self._relevance_d_threshold = relevance_d_threshold
        self._max_relevant = max_relevant

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

        # Minimum-intervention optimizer (same params as second stage)
        self._intervention_type = intervention_type
        self._intervention = None
        if intervention_type != 'none':
            self._intervention = SecondStageIntervention(
                horizon=self._horizon,
                dt=self._dt,
                ego_length=self._ego_length,
                ego_width=self._ego_width,
                wheelbase=self._wheelbase,
                collision_margin=self._collision_margin,
                frenet=self._frenet,
                params=policy.nlp_params,
                n_obs_max=policy.first_stage._n_obs_max,
                target_speed=self._target_speed,
                mode=intervention_type,
                w_agency=w_agency,
            )

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
             ego_position: np.ndarray = None):
        """Run one inference step.

        Args:
            frenet_state: Current ego Frenet state [s, d, phi, v].
            other_agent_states: Snapshot of all other agents' states.
            step_count: Current simulation step number (for display).
            ego_position: Current ego world position [x, y] (for plot centring).
        """
        # 1. Append to history
        self._history.append(HistoryEntry(
            frenet_state=frenet_state.copy(),
            other_agent_states={aid: s for aid, s in other_agent_states.items()},
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

        # Subsample observed trajectory at planning dt (warmup_steps + 1 points)
        observed_sd = self._subsample_history(start_idx, len(self._history) - 1,
                                              sim_steps_per_plan_step)

        # 4. Find relevant agents
        ego_s = frenet_state[0]
        relevant_aids = self._find_relevant_agents(ego_s, other_agent_states)

        # 5. Evaluate belief configs (only if there are relevant agents)
        results: List[InferenceResult] = []
        t_elapsed = 0.0

        if relevant_aids:
            configs = self._enumerate_configs(relevant_aids)

            # Sample road boundaries from the historical start state
            s_values = np.array([
                hist_frenet[0] + hist_frenet[3] * np.cos(hist_frenet[2]) * k * self._dt
                for k in range(self._horizon + 1)
            ])
            s_values = np.clip(s_values, 0.0, self._frenet.total_length)
            road_left, road_right = self._sample_road_boundaries(s_values)

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
                        solver_ok=False,
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
                    solver_ok=True,
                    nlp_states=nlp_states if nlp_ok else warm_states,
                    nlp_controls=nlp_controls if nlp_ok else warm_controls,
                ))

            t_elapsed = time.perf_counter() - t_start

            # Sort by position cost
            results.sort(key=lambda r: r.pos_cost)

            # Print results and get marginal posteriors
            marginals = self._print_results(results, step_count, relevant_aids,
                                            t_elapsed, self._warmup_steps)
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
                ego_heading=ego_heading)

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

    def _find_relevant_agents(self, ego_s: float,
                              other_agent_states: Dict[int, AgentState]
                              ) -> List[int]:
        """Find agents within the planning horizon's reach."""
        if self._frenet is None:
            return []

        margin = self._relevance_s_margin
        v_target = self._target_speed
        horizon_reach = self._horizon * self._dt * v_target
        s_min = ego_s - margin
        s_max = ego_s + horizon_reach + margin

        relevant = []
        for aid, state in other_agent_states.items():
            # Skip static objects (negative IDs are parked vehicles/barriers)
            if aid < 0:
                continue

            f = self._frenet.world_to_frenet(
                float(state.position[0]), float(state.position[1]))
            agent_s, agent_d = f['s'], f['d']

            if s_min <= agent_s <= s_max and abs(agent_d) < self._relevance_d_threshold:
                relevant.append(aid)

        # Cap at max_relevant (keep the closest by s-distance)
        if len(relevant) > self._max_relevant:
            relevant.sort(key=lambda aid: abs(
                self._frenet.world_to_frenet(
                    float(other_agent_states[aid].position[0]),
                    float(other_agent_states[aid].position[1]))['s'] - ego_s
            ))
            relevant = relevant[:self._max_relevant]

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
            if r.solver_ok and np.isfinite(r.pos_cost) and np.isfinite(r.vel_cost)
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
            status = "OK" if r.solver_ok else "FAILED"
            marker = "  <-- best" if energies[i] == best_energy and r.solver_ok else ""
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
                               ego_heading: float = 0.0):
        """Compute minimum-deviation intervention from the CURRENT state.

        Re-plans the believed trajectory from the current ego state (not the
        historical window start), then solves the intervention NLP from the
        same current state so both trajectories stay anchored to the ego.

        1. Threshold marginals → believed config
        2. Sample road boundaries & build believed obstacles from current state
        3. Two-stage plan (MILP → NLP) for believed trajectory
        4. Build true obstacles (all agents) from current state
        5. Solve intervention NLP
        6. Store results and update plotter
        7. Print diagnostics
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

        # Build believed obstacles (only visible agents)
        visible_aids = {aid for aid, vis in believed_config.items() if vis}
        believed_obstacles = self._predict_obstacles_cv(
            other_agent_states, visible_aids)

        # 3. Two-stage plan for believed trajectory from current state
        self._first_stage.reset()
        milp_states = self._first_stage.solve(
            frenet_state, road_left, road_right, believed_obstacles)

        if milp_states is None:
            cfg_str = ", ".join(
                f"{aid}:{'V' if vis else 'H'}"
                for aid, vis in sorted(believed_config.items()))
            logger.info("  Believed config: {%s} -- MILP failed, skipping intervention", cfg_str)
            self._last_intervention = None
            return

        warm_states, warm_controls = self._milp_to_nlp_warmstart(
            milp_states, frenet_state)

        self._second_stage.reset()
        nlp_states, nlp_controls, nlp_ok, _ = self._second_stage.solve(
            frenet_state, warm_states, warm_controls,
            road_left, road_right, believed_obstacles)

        if not nlp_ok:
            # Fall back to MILP warmstart as reference
            nlp_states = warm_states
            nlp_controls = warm_controls

        # 4. Build TRUE obstacles: all dynamic agents visible
        all_dynamic_aids = {aid for aid in other_agent_states if aid >= 0}
        true_obstacles = self._predict_obstacles_cv(
            other_agent_states, all_dynamic_aids)

        # 5. Solve intervention NLP from current state
        opt_states, opt_controls, success, intervention = self._intervention.solve(
            frenet_state, nlp_states, nlp_controls,
            road_left, road_right, true_obstacles)

        # 6. Store results
        self._last_intervention = {
            'believed_config': believed_config,
            'ref_states': nlp_states,
            'ref_controls': nlp_controls,
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
                ref_controls=nlp_controls,
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
        cfg_str = ", ".join(
            f"{aid}:{'V' if vis else 'H'}"
            for aid, vis in sorted(believed_config.items()))
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
