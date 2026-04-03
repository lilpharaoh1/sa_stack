"""
Simplified belief agent with basic distance-tracking controller.

This agent maintains a belief about other agents' velocity errors and uses
a simple proportional controller to track a desired following distance behind
a lead vehicle.  The assistive system infers the human's velocity-error via
a normalised Boltzmann rationality model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.vehicle import Action, Observation, KinematicVehicle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Belief data structures
# ---------------------------------------------------------------------------

@dataclass
class VelocityErrorBelief:
    """Discrete posterior over velocity-error candidates for one agent.

    Attributes:
        candidates:    Array of velocity-error values, e.g. [0, 0.1, ..., 1.0].
        probabilities: Posterior probabilities (sums to 1).
    """
    candidates: np.ndarray = field(
        default_factory=lambda: np.arange(-1.0, 0.1, 0.1))
    probabilities: np.ndarray = None

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = np.ones(len(self.candidates)) / len(self.candidates)

    @property
    def mode(self) -> float:
        return float(self.candidates[np.argmax(self.probabilities)])

    @property
    def mean(self) -> float:
        return float(np.sum(self.candidates * self.probabilities))


@dataclass
class SimpleBeliefState:
    """Belief state the assistive system maintains about the human driver.

    For now this only tracks the human's perceived velocity-error for each
    traffic agent.  More belief variables can be added later.
    """
    velocity_errors: Dict[int, VelocityErrorBelief] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SimpleBeliefAgent(Agent):
    """Ego agent with a basic distance-tracking controller and belief state.

    The human driver tries to maintain ``target_distance`` metres behind the
    nearest lead vehicle.  Their perception of the lead vehicle's speed is
    biased by a ``velocity_error`` fraction read from the scenario config.

    The assistive system maintains a discrete posterior over candidate
    velocity-error values and updates it each step using a normalised
    Boltzmann rationality model.

    Args:
        agent_id:        Unique agent identifier.
        initial_state:   Starting state.
        goal:            Destination goal.
        fps:             Simulation frame rate.
        scenario_map:    Parsed road map (``ip.Map``).
        agent_beliefs:   Per-agent belief config, e.g.
                         ``{"1": {"visible": True, "velocity_error": 0.7}}``.
        target_distance: Desired following distance in metres.
        beta:            Boltzmann rationality coefficient.  Higher values
                         make the posterior more peaked around the best-
                         matching candidate.
    """

    # Controller gains (class-level so they're shared with _compute_accel)
    K_DIST = 0.3    # distance-error gain
    K_SPEED = 1.0   # speed-error gain
    MAX_ACCEL = 5.0  # acceleration clamp (m/s^2)

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal=None,
                 fps: int = 20,
                 scenario_map=None,
                 agent_beliefs: Optional[Dict] = None,
                 target_distance: float = 20.0,
                 beta: float = 1.0,
                 **kwargs):
        super().__init__(agent_id, initial_state, goal, fps)
        self._scenario_map = scenario_map
        self._target_distance = target_distance
        self._beta = beta

        # Vehicle model for state propagation
        self._vehicle = KinematicVehicle(initial_state, initial_state.metadata, fps)

        # --- Human's *true* velocity-error per agent (from config) --------
        self._agent_beliefs_config = agent_beliefs or {}
        self._human_vel_errors: Dict[int, float] = {}   # populated in set_agents

        # --- Assistive system's *inferred* belief about the human ---------
        self._inferred_belief = SimpleBeliefState()

        # Other agents (populated by set_agents)
        self._other_agents: Dict[int, Agent] = {}

        # Logging / diagnostics
        self._step = 0
        self.last_step_info: Dict = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_agents(self, agents: Dict[int, Agent]):
        """Register other agents and initialise belief profiles."""
        self._other_agents = {aid: agent for aid, agent in agents.items()
                              if aid != self.agent_id}

        for aid in self._other_agents:
            aid_str = str(aid)
            cfg = self._agent_beliefs_config.get(aid_str, {})
            vel_err = float(cfg.get("velocity_error", 0.0))

            # True velocity-error (drives the human controller)
            self._human_vel_errors[aid] = vel_err

            # Inferred belief: uniform prior over candidates
            self._inferred_belief.velocity_errors[aid] = VelocityErrorBelief()

        logger.info("SimpleBeliefAgent %d: tracking %d other agents, "
                     "true_vel_errors=%s",
                     self.agent_id, len(self._other_agents),
                     self._human_vel_errors)

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _get_lead_vehicle(self, frame: Dict[int, AgentState]):
        """Find the closest agent ahead of the ego along the heading direction.

        Returns:
            (agent_id, AgentState, along_distance) or (None, None, None).
        """
        my_state = frame.get(self.agent_id)
        if my_state is None:
            return None, None, None

        heading = my_state.heading
        fwd = np.array([np.cos(heading), np.sin(heading)])

        best_aid = None
        best_state = None
        best_dist = float("inf")

        for aid in self._other_agents:
            if aid not in frame:
                continue
            diff = frame[aid].position - my_state.position
            along = diff @ fwd  # projection onto heading direction
            if 0 < along < best_dist:
                best_aid = aid
                best_state = frame[aid]
                best_dist = along

        return best_aid, best_state, best_dist

    def _get_lane_heading(self, position: np.ndarray) -> Optional[float]:
        """Return the lane heading at the given position."""
        if self._scenario_map is None:
            return None
        lane = self._scenario_map.best_lane_at(position, max_distance=500.0)
        if lane is None:
            return None
        ds = lane.distance_at(position)
        return lane.get_heading_at(ds)

    def _compute_accel_for_vel_err(self, ego_speed: float,
                                   lead_speed: float,
                                   distance: float,
                                   vel_err: float) -> float:
        """Pure function: compute the acceleration the controller would output
        given a particular velocity-error assumption.

        This is used both by the controller (with the true vel_err) and by
        inference (to evaluate each candidate).
        """
        perceived_lead_speed = lead_speed * (1.0 + vel_err)
        dist_error = distance - self._target_distance
        desired_speed = perceived_lead_speed + self.K_DIST * dist_error
        desired_speed = max(0.0, desired_speed)
        accel = self.K_SPEED * (desired_speed - ego_speed)
        return float(np.clip(accel, -self.MAX_ACCEL, self.MAX_ACCEL))

    def _run_controller(self, frame: Dict[int, AgentState]) -> Action:
        """Simple proportional controller for distance + speed tracking.

        The human perceives the lead vehicle's speed as
        ``lead_speed * (1 + velocity_error)``.  They try to match that
        perceived speed while also correcting for any distance error.
        """
        my_state = frame[self.agent_id]
        ego_speed = my_state.speed

        lead_aid, lead_state, distance = self._get_lead_vehicle(frame)

        if lead_state is None:
            accel = 0.0
            perceived_lead_speed = None
            desired_speed = ego_speed
        else:
            # Human uses their TRUE velocity-error
            vel_err = self._human_vel_errors.get(lead_aid, 0.0)
            accel = self._compute_accel_for_vel_err(
                ego_speed, lead_state.speed, distance, vel_err)

            # Diagnostics
            perceived_lead_speed = lead_state.speed * (1.0 + vel_err)
            dist_error = distance - self._target_distance
            desired_speed = max(0.0, perceived_lead_speed
                                + self.K_DIST * dist_error)

        # --- Steering: follow lane heading ---
        lane_heading = self._get_lane_heading(my_state.position)
        if lane_heading is not None:
            heading_error = lane_heading - my_state.heading
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            k_steer = 1.5
            steer = float(np.clip(k_steer * heading_error, -0.5, 0.5))
        else:
            steer = 0.0

        # Store diagnostics
        self.last_step_info = {
            "lead_aid": lead_aid,
            "distance": distance if distance != float("inf") else None,
            "ego_speed": ego_speed,
            "lead_speed": lead_state.speed if lead_state else None,
            "v_perceived": perceived_lead_speed,
            "v_desired": desired_speed,
            "accel": accel,
        }

        return Action(acceleration=accel,
                      steer_angle=steer,
                      target_speed=max(0.0, desired_speed))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, observation: Observation, action: Action):
        """Boltzmann rationality update over velocity-error candidates.

        For each candidate kappa, compute the acceleration the controller
        would output, then weight by:

            P(a_human | state, kappa) = exp(-beta * (a_human - a_kappa)^2)

        The posterior is updated via Bayes' rule:

            P(kappa | a_human, state) ∝ P(a_human | state, kappa) * P_prev(kappa)
        """
        frame = observation.frame
        lead_aid, lead_state, distance = self._get_lead_vehicle(frame)
        if lead_state is None:
            return

        ego_speed = frame[self.agent_id].speed
        human_accel = action.acceleration

        belief = self._inferred_belief.velocity_errors.get(lead_aid)
        if belief is None:
            return

        # Likelihood for each candidate
        likelihoods = np.empty(len(belief.candidates))
        for i, kappa in enumerate(belief.candidates):
            candidate_accel = self._compute_accel_for_vel_err(
                ego_speed, lead_state.speed, distance, kappa)
            likelihoods[i] = np.exp(
                -self._beta * (human_accel - candidate_accel) ** 2)

        # Bayesian update: posterior ∝ prior * likelihood
        posterior = belief.probabilities * likelihoods
        total = posterior.sum()
        if total > 0:
            belief.probabilities = posterior / total
        else:
            belief.probabilities = np.ones_like(posterior) / len(posterior)

        # Add inference diagnostics
        self.last_step_info["inferred_mode"] = belief.mode
        self.last_step_info["inferred_mean"] = belief.mean
        self.last_step_info["inferred_dist"] = dict(
            zip(np.round(belief.candidates, 2), np.round(belief.probabilities, 4)))

    # ------------------------------------------------------------------
    # Intervention (stub)
    # ------------------------------------------------------------------

    def _run_intervention(self, observation: Observation,
                          action: Action) -> Action:
        """Decide whether to override the human's action.
        Currently returns the action unchanged."""
        return action

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def next_action(self, observation: Observation,
                    prediction=None) -> Action:
        frame = observation.frame
        self._step += 1

        # 1. Controller (human drives with true velocity-error)
        action = self._run_controller(frame)

        # 2. Inference (update belief about human's velocity-error)
        self._run_inference(observation, action)

        # 3. Intervention (stub -- passes through)
        action = self._run_intervention(observation, action)

        return action

    def next_state(self, observation: Observation,
                   return_action: bool = False):
        action = self.next_action(observation)
        self._vehicle.execute_action(action)
        state = self._vehicle.get_state(time=self._step / self.fps)
        if return_action:
            return state, action
        return state

    def done(self, observation: Observation) -> bool:
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
    def agent_beliefs(self) -> Dict:
        """Backwards-compatible dict view of human's configured beliefs."""
        return self._agent_beliefs_config

    @property
    def inferred_belief(self) -> SimpleBeliefState:
        return self._inferred_belief

    @property
    def target_distance(self) -> float:
        return self._target_distance
