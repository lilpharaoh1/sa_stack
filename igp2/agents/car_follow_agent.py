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
        default_factory=lambda: np.arange(-0.5, 0.55, 0.1))
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
class CarFollowBeliefState:
    """Belief state the assistive system maintains about the human driver.

    For now this only tracks the human's perceived velocity-error for each
    traffic agent.  More belief variables can be added later.
    """
    velocity_errors: Dict[int, VelocityErrorBelief] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CarFollowAgent(Agent):
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
        d_safe:          Minimum safe following distance in metres.
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

        # Vehicle model for state propagation
        self._vehicle = KinematicVehicle(initial_state, initial_state.metadata, fps)

        # --- Human's *true* velocity-error per agent (from config) --------
        self._agent_beliefs_config = agent_beliefs or {}
        self._human_vel_errors: Dict[int, float] = {}   # populated in set_agents

        # --- Assistive system's *inferred* belief about the human ---------
        self._inferred_belief = CarFollowBeliefState()

        # Other agents (populated by set_agents)
        self._other_agents: Dict[int, Agent] = {}

        # Logging / diagnostics
        self._step = 0
        self.last_step_info: Dict = {}

        # Warm-start: previous lookahead solution
        self._prev_lookahead_sol: Optional[np.ndarray] = None

        # Kalman filter state for boltzmann_kalman inference
        # Initialised at perfect perception (kappa=0)
        self._kf_kappa: Dict[int, float] = {}      # per-agent estimate
        self._kf_P: Dict[int, float] = {}           # per-agent variance
        self._kf_Q = 0.001   # process noise
        self._kf_R = 0.1     # observation noise
        self._kf_B = 0.01    # process input gain

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

            # Kalman filter: initialise at perfect perception
            self._kf_kappa[aid] = 0.0
            self._kf_P[aid] = 0.5

        logger.info("CarFollowAgent %d: tracking %d other agents, "
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
            "human_vel_err": self._human_vel_errors.get(lead_aid, 0.0) if lead_aid is not None else None,
        }

        return Action(acceleration=accel,
                      steer_angle=steer,
                      target_speed=max(0.0, desired_speed))

    # ------------------------------------------------------------------
    # Human belief evolution
    # ------------------------------------------------------------------

    def _update_human_beliefs(self, frame: Dict[int, AgentState]):
        """Evolve the human's velocity-error using an RBF kernel on distance.

        When human='rbf', the velocity-error drifts toward 0 (perfect
        perception) at a rate governed by an RBF feature of the distance
        to the lead vehicle:

            f_kappa = exp(-d^2 / (2 * sigma^2))
            vel_err_new = vel_err * (1 - b_kappa * f_kappa)

        Closer to the lead -> higher f_kappa -> faster correction.
        This matches the velocity particle propagation from the belief
        experiments (deterministic, no noise).
        """
        if self._human_type != "rbf":
            return

        lead_aid, lead_state, distance = self._get_lead_vehicle(frame)
        if lead_state is None or distance == float("inf"):
            return

        f_kappa = np.exp(-distance ** 2 / (2.0 * self._sigma_kappa ** 2))

        old_vel_err = self._human_vel_errors.get(lead_aid, 0.0)
        new_vel_err = old_vel_err * (1.0 - self._b_kappa * f_kappa)
        self._human_vel_errors[lead_aid] = new_vel_err

        self.last_step_info["f_kappa"] = float(f_kappa)
        self.last_step_info["human_vel_err"] = new_vel_err

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, observation: Observation, action: Action):
        """Dispatch to the selected inference method."""
        if self._inference == "boltzmann_reactive":
            self._infer_boltzmann_reactive(observation, action)
        elif self._inference == "boltzmann_reactive_noprior":
            self._infer_boltzmann_reactive(observation, action, use_prior=False)
        elif self._inference == "boltzmann_reactive_floormix":
            self._infer_boltzmann_reactive(observation, action,
                                           floor_mix=True)
        elif self._inference == "boltzmann_kalman":
            self._infer_boltzmann_kalman(observation, action)
        elif self._inference == "oracle":
            self._infer_oracle()

    FLOOR_MIX_EPSILON = 0.05  # uniform mixing weight for floormix inference

    def _infer_boltzmann_reactive(self, observation: Observation,
                                  action: Action,
                                  use_prior: bool = True,
                                  floor_mix: bool = False):
        """Boltzmann rationality update using the reactive human model.

        For each candidate kappa, compute the acceleration the reactive
        controller would output, then weight by:

            P(a_human | state, kappa) = exp(-beta * (a_human - a_kappa)^2)

        If use_prior=True (default):
            posterior proportional to likelihood * prior

        If use_prior=False:
            posterior = normalised likelihood (uniform prior each step)

        If floor_mix=True:
            Before multiplying by the likelihood, the prior is mixed with
            a uniform distribution to prevent lock-in:
            prior_k = (1 - eps) * posterior_{k-1} + eps * uniform
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

        # Update belief
        if use_prior or floor_mix:
            prior = belief.probabilities
            if floor_mix:
                eps = self.FLOOR_MIX_EPSILON
                uniform = np.ones_like(prior) / len(prior)
                prior = (1.0 - eps) * prior + eps * uniform
            posterior = prior * likelihoods
        else:
            # No prior: normalised likelihood (uniform prior each step)
            posterior = likelihoods
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

    def _infer_boltzmann_kalman(self, observation: Observation,
                               action: Action):
        """Kalman filter inference over velocity-error.

        State:  x = kappa  (scalar per agent)

        Prediction (human model as process input):
            x_pred = A * x_hat + B * (-f_kappa * x_hat)
                   = x_hat * (1 - B * f_kappa)
            P_pred = A^2 * P + Q

        where A=1, B=0.01, f_kappa = RBF feature on distance.

        Observation (linearised human controller):
            a_H = a_H(kappa=0) + C * kappa + noise
            C = K_SPEED * v_lead   (da_H / dkappa)

        Kalman update:
            innovation = a_observed - a_predicted(x_pred)
            S = C^2 * P_pred + R
            K = P_pred * C / S
            x_hat = x_pred + K * innovation
            P = (1 - K * C) * P_pred

        The Gaussian (x_hat, P) is projected onto the discrete candidate
        grid for compatibility with the rest of the system.
        """
        frame = observation.frame
        lead_aid, lead_state, distance = self._get_lead_vehicle(frame)
        if lead_state is None:
            return

        ego_speed = frame[self.agent_id].speed
        lead_speed = lead_state.speed
        human_accel = action.acceleration

        if lead_aid not in self._kf_kappa:
            return

        x_hat = self._kf_kappa[lead_aid]
        P = self._kf_P[lead_aid]

        # --- Prediction ---
        f_kappa = np.exp(-distance ** 2 / (2.0 * self._sigma_kappa ** 2))
        x_pred = x_hat * (1.0 - self._kf_B * f_kappa)
        P_pred = P + self._kf_Q

        # --- Observation ---
        # Linearisation: C = da_H / dkappa = K_SPEED * v_lead
        C = self.K_SPEED * lead_speed
        a_pred = self._compute_accel_for_vel_err(
            ego_speed, lead_speed, distance, x_pred)
        innovation = human_accel - a_pred

        S = C ** 2 * P_pred + self._kf_R
        K_gain = P_pred * C / S
        x_hat_new = x_pred + K_gain * innovation
        P_new = (1.0 - K_gain * C) * P_pred

        self._kf_kappa[lead_aid] = x_hat_new
        self._kf_P[lead_aid] = max(P_new, 1e-8)

        # --- Project Gaussian onto discrete candidates ---
        belief = self._inferred_belief.velocity_errors.get(lead_aid)
        if belief is not None:
            probs = np.exp(-(belief.candidates - x_hat_new) ** 2
                           / (2.0 * max(P_new, 1e-8)))
            total = probs.sum()
            if total > 0:
                belief.probabilities = probs / total
            else:
                belief.probabilities = np.ones(len(probs)) / len(probs)

            self.last_step_info["inferred_mode"] = belief.mode
            self.last_step_info["inferred_mean"] = belief.mean
            self.last_step_info["inferred_dist"] = dict(
                zip(np.round(belief.candidates, 2),
                    np.round(belief.probabilities, 4)))
            self.last_step_info["kf_kappa"] = x_hat_new
            self.last_step_info["kf_P"] = P_new

    def _infer_oracle(self):
        """Set the belief to the ground-truth velocity-error from config."""
        for aid, vel_err in self._human_vel_errors.items():
            belief = self._inferred_belief.velocity_errors.get(aid)
            if belief is None:
                continue
            # Put all probability on the nearest candidate
            dists = np.abs(belief.candidates - vel_err)
            belief.probabilities = np.zeros_like(belief.probabilities)
            belief.probabilities[np.argmin(dists)] = 1.0

            self.last_step_info["inferred_mode"] = belief.mode
            self.last_step_info["inferred_mean"] = belief.mean
            self.last_step_info["inferred_dist"] = dict(
                zip(np.round(belief.candidates, 2),
                    np.round(belief.probabilities, 4)))

    # ------------------------------------------------------------------
    # Safety filter (Control Barrier Function)
    # ------------------------------------------------------------------
    #
    # Safe set:   C = { x : h(x) >= 0 }
    # Barrier:    h(x) = d - d_safe
    #
    # Discrete-time CBF condition (per step):
    #   h(x_{t+1}) >= (1 - gamma) * h(x_t)
    #
    # gamma = 1.0  ->  h(x_{t+1}) >= 0            (hard constraint)
    # gamma < 1.0  ->  margin decays gradually     (smoother filter)
    #
    # The safety filter solves:
    #   min_u  ||u - u_des||^2
    #   s.t.   Delta_h(x, u) + gamma * h(x) >= 0
    #
    # where Delta_h = h(f(x, u)) - h(x).
    # ------------------------------------------------------------------

    LOOKAHEAD_N = 25  # horizon steps for predictive safety filter

    def _barrier(self, distance: float) -> float:
        """Barrier function  h(x) = d - d_safe."""
        return distance - self._d_safe

    def _barrier_vec(self, distances: np.ndarray) -> np.ndarray:
        """Vectorised barrier function."""
        return distances - self._d_safe

    def _cbf_max_accel(self, distance: float, ego_speed: float,
                       lead_speed: float) -> float:
        """Maximum acceleration satisfying the discrete-time CBF condition.

        The bicycle model uses forward-Euler:
            d_{k+1} = d_k + (v_lead - v_ego) * dt
            v_{k+1} = v_ego + a * dt

        The CBF condition  h(x_{t+1}) >= (1 - gamma) * h(x_t)  gives:

            (v_lead - v_ego) * dt  >=  -gamma * h(x_t)

        Since v_ego doesn't change until next step (Euler), the distance
        constraint is independent of a at step k.  The velocity constraint
        v_{k+1} >= 0  limits acceleration from below.

        We enforce safety at step k+2 (two-step lookahead) to account for
        the delayed effect of acceleration on distance:

            d_{k+2} = d_{k+1} + (v_lead - v_{k+1}) * dt
                    = d_k + (v_lead - v_ego)*dt + (v_lead - v_ego - a*dt)*dt

            h(x_{k+2}) >= (1-gamma)^2 * h(x_k)  gives:

            a  <=  [ 2*(v_lead - v_ego)*dt + gamma*(2-gamma)*h(x_k) ] / dt^2
        """
        dt = 1.0 / self.fps
        h = self._barrier(distance)
        g = self._gamma
        return (2.0 * (lead_speed - ego_speed) * dt
                + g * (2.0 - g) * h) / (dt ** 2)

    def _simulate_forward(self, a_exec, lead_speed, s0, kappa, dt, N,
                          ceiling=False):
        """Forward-simulate N steps of the 1-D car-following dynamics.

        Args:
            a_exec:     (N,) executed accelerations (decision variables).
            lead_speed: Constant lead vehicle speed.
            s0:         Initial state [distance, ego_speed].
            kappa:      Velocity-error used to predict the human's action.
            dt:         Timestep.
            N:          Horizon length.
            ceiling:    If True the effective action at each step is
                        ``min(a_human, a_exec)`` — the filter can only
                        *reduce* acceleration.

        Returns:
            states:   (N+1, 2) array of [distance, ego_speed].
            a_H_pred: (N,) human's predicted accelerations.
        """
        states = np.zeros((N + 1, 2))
        states[0] = s0
        a_H_pred = np.zeros(N)

        for j in range(N):
            d_j, v_j = states[j]
            # Human's intended action under this kappa
            perceived = lead_speed * (1.0 + kappa)
            desired = max(0.0, perceived
                          + self.K_DIST * (d_j - self._target_distance))
            a_H = float(np.clip(self.K_SPEED * (desired - v_j),
                                -self.MAX_ACCEL, self.MAX_ACCEL))
            a_H_pred[j] = a_H

            # Effective action (ceiling = filter can only reduce accel)
            a = min(a_H, a_exec[j]) if ceiling else a_exec[j]

            # Dynamics (forward Euler, matching bicycle model):
            #   d_{k+1} = d_k + (v_lead - v_ego) * dt
            #   v_{k+1} = max(0, v_ego + a * dt)
            v_next = max(0.0, v_j + a * dt)
            d_next = d_j + (lead_speed - v_j) * dt
            states[j + 1] = [d_next, v_next]

        return states, a_H_pred

    # --- cbf_single: one-step CBF safety filter -----------------------------

    def _intervene_cbf_single(self, action, distance, ego_speed, lead_speed):
        """One-step CBF safety filter:  min ||u - u_H||^2  s.t. CBF condition.

        For a scalar acceleration this reduces to clipping:
            u = min(u_H, a_cbf_max)
        """
        a_cbf_max = self._cbf_max_accel(distance, ego_speed, lead_speed)
        intervened = False

        if action.acceleration > a_cbf_max:
            safe_accel = float(np.clip(a_cbf_max,
                                       -self.MAX_ACCEL, self.MAX_ACCEL))
            action = Action(acceleration=safe_accel,
                            steer_angle=action.steer_angle,
                            target_speed=action.target_speed)
            intervened = True

        return action, intervened, a_cbf_max

    # --- predictive CBF safety filter (lookahead) -------------------------

    def _intervene_lookahead(self, action, distance, ego_speed, lead_speed,
                             lead_aid, robust=False,
                             kappa_override=None):
        """Predictive safety filter (CBF-MPC).

        Solves a receding-horizon optimisation:

            min_{u_{0:N-1}}  sum_j || u_j - a_H(x_j, kappa_hat) ||^2

            s.t.  h(x_{j+1}) >= (1 - gamma) * h(x_j)   (CBF condition)
                  v_j >= 0                                (speed bound)

        cbf_mode:     kappa_hat from discrete belief mode.
        cbf_contmean:   kappa_hat from continuous Kalman estimate.
        cbf_chance:   constraints over all likely kappa candidates,
                      with ceiling model.

        Only u_0 is applied (receding horizon).
        """
        from scipy.optimize import minimize as scipy_minimize

        dt = 1.0 / self.fps
        N = self.LOOKAHEAD_N
        s0 = np.array([distance, ego_speed])

        belief = self._inferred_belief.velocity_errors.get(lead_aid)
        if kappa_override is not None:
            kappa_hat = kappa_override
        else:
            kappa_hat = belief.mode if belief is not None else 0.0

        # -- objective: minimise deviation from human ----------------------
        def objective(a_exec):
            _, a_H = self._simulate_forward(
                a_exec, lead_speed, s0, kappa_hat, dt, N)
            return float(np.sum((a_exec - a_H) ** 2))

        # -- CBF + speed constraints ---------------------------------------
        constraints = []

        if robust and belief is not None:
            # Only check candidates with non-negligible probability
            mask = belief.probabilities >= 0.1
            kappas_check = belief.candidates[mask]
            if len(kappas_check) == 0:
                kappas_check = [kappa_hat]
        else:
            kappas_check = [kappa_hat]

        for kappa_k in kappas_check:
            use_ceiling = robust

            def _make_cbf_con(k, ceil, gamma=self._gamma):
                def con(a_exec):
                    st, _ = self._simulate_forward(
                        a_exec, lead_speed, s0, k, dt, N, ceiling=ceil)
                    # CBF: h(x_{j+1}) - (1-gamma)*h(x_j) >= 0
                    h_j = self._barrier_vec(st[:-1, 0])
                    h_jp1 = self._barrier_vec(st[1:, 0])
                    return h_jp1 - (1.0 - gamma) * h_j
                return con

            def _make_v_con(k, ceil):
                def con(a_exec):
                    st, _ = self._simulate_forward(
                        a_exec, lead_speed, s0, k, dt, N, ceiling=ceil)
                    return st[1:, 1]
                return con

            constraints.append(
                {"type": "ineq", "fun": _make_cbf_con(kappa_k, use_ceiling)})
            constraints.append(
                {"type": "ineq", "fun": _make_v_con(kappa_k, use_ceiling)})

        # -- initial guess: warm-start from previous solution ---------------
        if (self._prev_lookahead_sol is not None
                and len(self._prev_lookahead_sol) == N):
            # Shift left by one: drop a_exec[0] (already applied),
            # append the human's current acceleration as the new last step
            x0 = np.empty(N)
            x0[:-1] = self._prev_lookahead_sol[1:]
            x0[-1] = action.acceleration
        else:
            x0 = np.full(N, action.acceleration)

        result = scipy_minimize(
            objective, x0,
            method="SLSQP",
            bounds=[(-self.MAX_ACCEL, self.MAX_ACCEL)] * N,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        if result.success:
            self._prev_lookahead_sol = result.x.copy()
            safe_accel = float(np.clip(result.x[0],
                                       -self.MAX_ACCEL, self.MAX_ACCEL))
        else:
            self._prev_lookahead_sol = None
            # Fallback to single-step CBF filter
            a_max_safe = self._cbf_max_accel(
                distance, ego_speed, lead_speed)
            safe_accel = float(np.clip(
                min(action.acceleration, a_max_safe),
                -self.MAX_ACCEL, self.MAX_ACCEL))

        intervened = abs(safe_accel - action.acceleration) > 1e-3
        new_action = Action(acceleration=safe_accel,
                            steer_angle=action.steer_angle,
                            target_speed=action.target_speed)
        return new_action, intervened

    # --- dispatcher -------------------------------------------------------

    def _run_intervention(self, observation: Observation,
                          action: Action) -> Action:
        """Dispatch to the selected safety filter / intervention."""
        frame = observation.frame
        lead_aid, lead_state, distance = self._get_lead_vehicle(frame)

        intervened = False
        a_cbf_max = None

        if lead_state is not None and distance != float("inf"):
            ego_speed = frame[self.agent_id].speed
            lead_speed = lead_state.speed

            # Always compute one-step CBF bound for diagnostics
            a_cbf_max = self._cbf_max_accel(
                distance, ego_speed, lead_speed)

            if self._intervention == "cbf_single":
                action, intervened, a_cbf_max = self._intervene_cbf_single(
                    action, distance, ego_speed, lead_speed)

            elif self._intervention in ("cbf_mode", "cbf_chance"):
                robust = self._intervention == "cbf_chance"
                action, intervened = self._intervene_lookahead(
                    action, distance, ego_speed, lead_speed,
                    lead_aid, robust=robust)

            elif self._intervention == "cbf_wmean":
                # Weighted mean of discrete belief
                belief = self._inferred_belief.velocity_errors.get(lead_aid)
                wmean = belief.mean if belief is not None else 0.0
                action, intervened = self._intervene_lookahead(
                    action, distance, ego_speed, lead_speed,
                    lead_aid, kappa_override=wmean)

            elif self._intervention == "cbf_contmean":
                # Use continuous Kalman estimate directly (no discretisation)
                kf_est = self._kf_kappa.get(lead_aid, 0.0)
                action, intervened = self._intervene_lookahead(
                    action, distance, ego_speed, lead_speed,
                    lead_aid, kappa_override=kf_est)

            elif self._intervention == "always_policy":
                # Perfect-perception policy: kappa=0 (no velocity error)
                perfect_accel = self._compute_accel_for_vel_err(
                    ego_speed, lead_speed, distance, vel_err=0.0)
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

    def next_action(self, observation: Observation,
                    prediction=None) -> Action:
        frame = observation.frame
        self._step += 1

        # 0. Evolve human's velocity-error belief (if kalman)
        self._update_human_beliefs(frame)

        # 1. Controller (human drives with current velocity-error)
        action = self._run_controller(frame)

        # 2. Inference (update assistive system's belief about vel-error)
        self._run_inference(observation, action)

        # 3. Safety filter (project onto safe set)
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
    def inferred_belief(self) -> CarFollowBeliefState:
        return self._inferred_belief

    @property
    def target_distance(self) -> float:
        return self._target_distance
