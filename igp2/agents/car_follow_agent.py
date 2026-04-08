"""Car-following agent with CBF safety filter and belief inference.

Thin orchestrator that delegates to ``igp2.carfollow`` subpackage for
the human model, inference, and intervention logic.
"""

import logging
from typing import Dict, Optional

import numpy as np

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.vehicle import Action, Observation, KinematicVehicle

from igp2.carfollow.beliefs import VelocityErrorBelief, CarFollowBeliefState
from igp2.carfollow import human_model
from igp2.carfollow import inference as _inference
from igp2.carfollow import intervention as _intervention
from igp2.recognition.astar import AStar

logger = logging.getLogger(__name__)


class CarFollowAgent(Agent):
    """Ego agent with proportional controller, belief inference, and
    CBF safety filter.

    All experiment parameters can be set via the constructor (populated
    from config) or overridden by CLI arguments in the runner script.
    """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal=None,
                 fps: int = 20,
                 scenario_map=None,
                 agent_beliefs: Optional[Dict] = None,
                 target_distance: float = 12.0,
                 d_safe: float = 8.0,
                 beta: float = 1.0,
                 gamma: float = 0.99,
                 inference: str = "none",
                 intervention: str = "none",
                 human: str = "static",
                 b_kappa: float = 0.01,
                 sigma_kappa: float = 25.0,
                 **kwargs):
        super().__init__(agent_id, initial_state, goal, fps)
        self._scenario_map = scenario_map
        self._target_distance = target_distance
        self._d_safe = d_safe
        self._beta = beta
        self._gamma = gamma
        self._inference = inference
        self._intervention = intervention
        self._human_type = human
        self._b_kappa = b_kappa
        self._sigma_kappa = sigma_kappa

        # Vehicle model
        self._vehicle = KinematicVehicle(initial_state,
                                         initial_state.metadata, fps)

        # Human's true velocity-error per agent (from config)
        self._agent_beliefs_config = agent_beliefs or {}
        self._human_vel_errors: Dict[int, float] = {}

        # Assistive system's inferred belief
        self._inferred_belief = CarFollowBeliefState()

        # Other agents (populated by set_agents)
        self._other_agents: Dict[int, Agent] = {}

        # Diagnostics
        self._step = 0
        self.last_step_info: Dict = {}

        # Lookahead warm-start
        self._prev_lookahead_sol: Optional[np.ndarray] = None

        # Kalman filter state (for boltzmann_kalman inference)
        self._kf_kappa: Dict[int, float] = {}
        self._kf_P: Dict[int, float] = {}
        self._kf_Q = 0.001
        self._kf_R = 0.01
        self._kf_B = 0.01

        # Reference path from A* (lane sequence + waypoints)
        self._reference_waypoints: Optional[np.ndarray] = None
        self._reference_path = []  # list of (Lane, waypoints)
        if scenario_map is not None and goal is not None:
            self._compute_reference_path()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_agents(self, agents: Dict[int, Agent]):
        """Register other agents and initialise belief profiles."""
        self._other_agents = {aid: a for aid, a in agents.items()
                              if aid != self.agent_id}
        for aid in self._other_agents:
            cfg = self._agent_beliefs_config.get(str(aid), {})
            vel_err = float(cfg.get("velocity_error", 0.0))
            self._human_vel_errors[aid] = vel_err
            self._inferred_belief.velocity_errors[aid] = VelocityErrorBelief()
            self._kf_kappa[aid] = 0.0
            self._kf_P[aid] = 0.5

        logger.info("CarFollowAgent %d: tracking %d agents, true_vel_errors=%s",
                     self.agent_id, len(self._other_agents),
                     self._human_vel_errors)

    # ------------------------------------------------------------------
    # Reference path (A*)
    # ------------------------------------------------------------------

    def _compute_reference_path(self):
        """Run A* to find a lane sequence to the goal, then extract waypoints."""
        astar = AStar(max_iter=1000)
        frame = {self.agent_id: self._initial_state}
        _, solutions = astar.search(
            self.agent_id, frame, self.goal, self._scenario_map,
            open_loop=True, fps=self.fps)

        if not solutions:
            logger.warning("CarFollowAgent %d: A* found no path to goal %s",
                           self.agent_id, self.goal)
            return

        macro_actions = solutions[0]
        all_waypoints = []
        for ma in macro_actions:
            for maneuver in ma._maneuvers:
                lane = (maneuver.lane_sequence[0]
                        if maneuver.lane_sequence else None)
                waypoints = maneuver.trajectory.path.copy()
                self._reference_path.append((lane, waypoints))
                all_waypoints.append(waypoints)

        if all_waypoints:
            combined = [all_waypoints[0]]
            for wp in all_waypoints[1:]:
                combined.append(wp[1:])  # skip duplicate start point
            self._reference_waypoints = np.concatenate(combined, axis=0)

        logger.info("CarFollowAgent %d: reference path has %d segments "
                     "(%d waypoints)",
                     self.agent_id, len(self._reference_path),
                     len(self._reference_waypoints)
                     if self._reference_waypoints is not None else 0)

    def _get_reference_heading(self, position: np.ndarray) -> Optional[float]:
        """Get heading from the reference path at the nearest waypoint."""
        if self._reference_waypoints is None or len(self._reference_waypoints) < 2:
            return None

        # Find nearest waypoint
        diffs = self._reference_waypoints - position
        dists = np.sum(diffs ** 2, axis=1)
        idx = np.argmin(dists)

        # Heading from this waypoint toward the next
        if idx < len(self._reference_waypoints) - 1:
            p0 = self._reference_waypoints[idx]
            p1 = self._reference_waypoints[idx + 1]
        else:
            p0 = self._reference_waypoints[idx - 1]
            p1 = self._reference_waypoints[idx]

        dx, dy = p1 - p0
        return float(np.arctan2(dy, dx))

    # ------------------------------------------------------------------
    # Human belief evolution
    # ------------------------------------------------------------------

    def _update_human_beliefs(self, frame):
        if self._human_type != "rbf":
            return
        lead_aid, lead_state, distance = human_model.get_lead_vehicle(
            self.agent_id, frame, self._other_agents)
        if lead_state is None or distance == float("inf"):
            return
        old = self._human_vel_errors.get(lead_aid, 0.0)
        new = human_model.evolve_vel_err(
            old, distance, self._b_kappa, self._sigma_kappa)
        self._human_vel_errors[lead_aid] = new
        self.last_step_info["f_kappa"] = float(
            human_model.rbf_feature(distance, self._sigma_kappa))
        self.last_step_info["human_vel_err"] = new

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _run_controller(self, frame):
        my_state = frame[self.agent_id]
        ego_speed = my_state.speed
        lead_aid, lead_state, distance = human_model.get_lead_vehicle(
            self.agent_id, frame, self._other_agents)

        if lead_state is None:
            accel = 0.0
            perceived_lead_speed = None
            desired_speed = ego_speed
        else:
            vel_err = self._human_vel_errors.get(lead_aid, 0.0)
            accel = human_model.compute_accel(
                ego_speed, lead_state.speed, distance, vel_err,
                self._target_distance)
            perceived_lead_speed = lead_state.speed * (1.0 + vel_err)
            desired_speed = max(
                0.0, perceived_lead_speed
                + human_model.K_DIST * (distance - self._target_distance))

        # Steering: follow reference path (A*), fall back to nearest lane
        steer = 0.0
        ref_heading = self._get_reference_heading(my_state.position)
        if ref_heading is not None:
            heading_error = ref_heading - my_state.heading
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            steer = float(np.clip(1.5 * heading_error, -0.5, 0.5))
        elif self._scenario_map is not None:
            lane = self._scenario_map.best_lane_at(
                my_state.position, max_distance=500.0)
            if lane is not None:
                ds = lane.distance_at(my_state.position)
                lane_heading = lane.get_heading_at(ds)
                heading_error = lane_heading - my_state.heading
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                steer = float(np.clip(1.5 * heading_error, -0.5, 0.5))

        self.last_step_info = {
            "lead_aid": lead_aid,
            "distance": distance if distance != float("inf") else None,
            "ego_speed": ego_speed,
            "lead_speed": lead_state.speed if lead_state else None,
            "v_perceived": perceived_lead_speed,
            "v_desired": desired_speed,
            "accel": accel,
            "human_vel_err": (self._human_vel_errors.get(lead_aid, 0.0)
                              if lead_aid is not None else None),
        }
        return Action(acceleration=accel, steer_angle=steer,
                      target_speed=max(0.0, desired_speed))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, observation, action):
        frame = observation.frame
        lead_aid, lead_state, distance = human_model.get_lead_vehicle(
            self.agent_id, frame, self._other_agents)
        if lead_state is None:
            return

        ego_speed = frame[self.agent_id].speed
        belief = self._inferred_belief.velocity_errors.get(lead_aid)
        if belief is None:
            return

        common = dict(ego_speed=ego_speed, lead_speed=lead_state.speed,
                      distance=distance, human_accel=action.acceleration,
                      target_distance=self._target_distance)

        if self._inference == "boltzmann_reactive":
            diag = _inference.infer_boltzmann_reactive(
                belief, beta=self._beta, **common)
        elif self._inference == "boltzmann_reactive_noprior":
            diag = _inference.infer_boltzmann_reactive(
                belief, beta=self._beta, use_prior=False, **common)
        elif self._inference == "boltzmann_reactive_floormix":
            diag = _inference.infer_boltzmann_reactive(
                belief, beta=self._beta, floor_mix=True, **common)
        elif self._inference == "boltzmann_kalman":
            diag = _inference.infer_boltzmann_kalman(
                self._kf_kappa[lead_aid], self._kf_P[lead_aid],
                belief=belief,
                sigma_kappa=self._sigma_kappa,
                kf_B=self._kf_B, kf_Q=self._kf_Q, kf_R=self._kf_R,
                **common)
            self._kf_kappa[lead_aid] = diag["kf_kappa"]
            self._kf_P[lead_aid] = diag["kf_P"]
        elif self._inference == "oracle":
            diag = _inference.infer_oracle(
                belief, self._human_vel_errors.get(lead_aid, 0.0))
        else:
            return

        self.last_step_info.update(diag)

    # ------------------------------------------------------------------
    # Intervention (CBF safety filter)
    # ------------------------------------------------------------------

    def _run_intervention(self, observation, action):
        frame = observation.frame
        lead_aid, lead_state, distance = human_model.get_lead_vehicle(
            self.agent_id, frame, self._other_agents)

        intervened = False
        a_cbf_max = None

        if lead_state is not None and distance != float("inf"):
            ego_speed = frame[self.agent_id].speed
            lead_speed = lead_state.speed

            a_cbf_max = _intervention.cbf_max_accel(
                distance, ego_speed, lead_speed,
                self._d_safe, self._gamma, self.fps)

            belief = self._inferred_belief.velocity_errors.get(lead_aid)

            if self._intervention == "cbf_single":
                safe_a, intervened, a_cbf_max = _intervention.intervene_cbf_single(
                    action.acceleration, distance, ego_speed, lead_speed,
                    self._d_safe, self._gamma, self.fps)
                if intervened:
                    action = Action(acceleration=safe_a,
                                   steer_angle=action.steer_angle,
                                   target_speed=action.target_speed)

            elif self._intervention in ("cbf_mode", "cbf_wmean",
                                        "cbf_contmean", "cbf_kalman",
                                        "cbf_chance"):
                # Determine kappa_hat
                if self._intervention == "cbf_mode":
                    kappa_hat = belief.mode if belief else 0.0
                elif self._intervention == "cbf_wmean":
                    kappa_hat = belief.mean if belief else 0.0
                elif self._intervention in ("cbf_contmean", "cbf_kalman"):
                    kappa_hat = self._kf_kappa.get(lead_aid, 0.0)
                else:
                    kappa_hat = belief.mode if belief else 0.0

                # Robust kappas
                robust_kappas = None
                if self._intervention == "cbf_chance" and belief is not None:
                    mask = belief.probabilities >= 0.1
                    rk = belief.candidates[mask]
                    robust_kappas = rk if len(rk) > 0 else None

                evolving = self._intervention == "cbf_kalman"

                safe_a, intervened, self._prev_lookahead_sol = \
                    _intervention.intervene_lookahead(
                        action.acceleration, distance, ego_speed, lead_speed,
                        kappa_hat, self._d_safe, self._gamma, self.fps,
                        self._target_distance,
                        prev_sol=self._prev_lookahead_sol,
                        robust_kappas=robust_kappas,
                        evolving=evolving,
                        b_kappa=self._b_kappa,
                        sigma_kappa=self._sigma_kappa)
                if intervened:
                    action = Action(acceleration=safe_a,
                                   steer_angle=action.steer_angle,
                                   target_speed=action.target_speed)

            elif self._intervention == "always_policy":
                perfect_accel = human_model.compute_accel(
                    ego_speed, lead_speed, distance, 0.0,
                    self._target_distance)
                action = Action(acceleration=perfect_accel,
                                steer_angle=action.steer_angle,
                                target_speed=action.target_speed)
                intervened = True

        self.last_step_info["intervened"] = intervened
        self.last_step_info["a_max_safe"] = (
            float(a_cbf_max) if a_cbf_max is not None else None)
        self.last_step_info["executed_accel"] = action.acceleration
        return action

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def next_action(self, observation, prediction=None):
        frame = observation.frame
        self._step += 1
        self._update_human_beliefs(frame)
        action = self._run_controller(frame)
        self._run_inference(observation, action)
        action = self._run_intervention(observation, action)
        return action

    def next_state(self, observation, return_action=False):
        action = self.next_action(observation)
        self._vehicle.execute_action(action)
        state = self._vehicle.get_state(time=self._step / self.fps)
        if return_action:
            return state, action
        return state

    def done(self, observation):
        if self.goal is None:
            return False
        state = observation.frame.get(self.agent_id)
        if state is None:
            return False
        return self.goal.reached(state.position)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_beliefs(self):
        return self._agent_beliefs_config

    @property
    def inferred_belief(self):
        return self._inferred_belief

    @property
    def target_distance(self):
        return self._target_distance
