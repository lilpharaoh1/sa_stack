"""
Car-following controller with velocity-error beliefs, inference, and
CBF safety filtering — self-contained port of igp2/carfollow for the
CommonRoad experiment harness.

The controller is a callable that maps Observation → Action and can be
plugged directly into the Simulation / SimVehicle framework.

Mirrors the logic in:
    igp2/agents/car_follow_agent.py
    igp2/carfollow/human_model.py
    igp2/carfollow/beliefs.py
    igp2/carfollow/inference.py
    igp2/carfollow/intervention.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from run_experiment import VehicleState, Action, Observation, LANE_WIDTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Human model constants (matching igp2/carfollow/human_model.py)
# ---------------------------------------------------------------------------
K_DIST = 0.3       # distance-error gain
K_SPEED = 1.0      # speed-error gain
MAX_ACCEL = 5.0     # acceleration clamp (m/s^2)


# ---------------------------------------------------------------------------
#  Belief data structures
# ---------------------------------------------------------------------------
@dataclass
class VelocityErrorBelief:
    """Discrete posterior over velocity-error candidates kappa."""
    candidates: np.ndarray = field(
        default_factory=lambda: np.arange(-0.5, 0.55, 0.1))
    probabilities: np.ndarray = None

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = (
                np.ones(len(self.candidates)) / len(self.candidates))

    @property
    def mode(self) -> float:
        return float(self.candidates[np.argmax(self.probabilities)])

    @property
    def mean(self) -> float:
        return float(np.sum(self.candidates * self.probabilities))

    @property
    def std(self) -> float:
        mu = self.mean
        return float(np.sqrt(np.sum(
            self.probabilities * (self.candidates - mu) ** 2)))


# ---------------------------------------------------------------------------
#  Human model helpers
# ---------------------------------------------------------------------------

def compute_accel(ego_speed: float, lead_speed: float,
                  distance: float, vel_err: float,
                  target_distance: float) -> float:
    """Proportional controller with velocity-error bias.

    v_desired = v_lead * (1 + kappa) + k_d * (d - d*)
    a = k_v * (v_desired - v_ego)
    """
    perceived = lead_speed * (1.0 + vel_err)
    dist_error = distance - target_distance
    desired = max(0.0, perceived + K_DIST * dist_error)
    accel = K_SPEED * (desired - ego_speed)
    return float(np.clip(accel, -MAX_ACCEL, MAX_ACCEL))


def rbf_feature(distance: float, sigma: float = 25.0) -> float:
    """RBF kernel on following distance (attention kernel)."""
    return float(np.exp(-distance ** 2 / (2.0 * sigma ** 2)))


def evolve_vel_err(vel_err: float, distance: float,
                   b_kappa: float = 0.01, sigma: float = 25.0) -> float:
    """One step of RBF drift toward perfect perception."""
    f = rbf_feature(distance, sigma)
    return vel_err * (1.0 - b_kappa * f)


def get_lead_vehicle(ego: VehicleState,
                     others: Dict[int, VehicleState]):
    """Find closest vehicle ahead in the same lane."""
    fwd = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    best_id, best_state, best_dist = None, None, float("inf")

    for vid, vs in others.items():
        diff = np.array([vs.x - ego.x, vs.y - ego.y])
        along = diff @ fwd
        lateral = abs(diff[0] * (-fwd[1]) + diff[1] * fwd[0])
        if 0 < along < best_dist and lateral < LANE_WIDTH * 0.8:
            best_id = vid
            best_state = vs
            best_dist = along

    return best_id, best_state, best_dist


def get_all_vehicles_ahead(ego: VehicleState,
                           others: Dict[int, VehicleState],
                           max_range: float = 100.0):
    """Find all vehicles ahead of ego within max_range.

    Returns dict of vid → (state, along_distance, lateral_offset).
    Includes vehicles in adjacent lanes that may merge.
    """
    fwd = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    right = np.array([-fwd[1], fwd[0]])
    result = {}

    for vid, vs in others.items():
        diff = np.array([vs.x - ego.x, vs.y - ego.y])
        along = diff @ fwd
        lateral = diff @ right
        if 0 < along < max_range:
            result[vid] = (vs, along, lateral)

    return result


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------

FLOOR_MIX_EPSILON = 0.05


def infer_boltzmann_reactive(belief: VelocityErrorBelief,
                             ego_speed, lead_speed, distance,
                             human_accel, target_distance, beta,
                             use_prior=True, floor_mix=False) -> dict:
    likelihoods = np.empty(len(belief.candidates))
    for i, kappa in enumerate(belief.candidates):
        a_cand = compute_accel(ego_speed, lead_speed, distance,
                               kappa, target_distance)
        likelihoods[i] = np.exp(-beta * (human_accel - a_cand) ** 2)

    if use_prior or floor_mix:
        prior = belief.probabilities
        if floor_mix:
            uniform = np.ones_like(prior) / len(prior)
            prior = ((1.0 - FLOOR_MIX_EPSILON) * prior
                     + FLOOR_MIX_EPSILON * uniform)
        posterior = prior * likelihoods
    else:
        posterior = likelihoods

    total = posterior.sum()
    if total > 0:
        belief.probabilities = posterior / total
    else:
        belief.probabilities = np.ones_like(posterior) / len(posterior)

    return {
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_std": belief.std,
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


def infer_boltzmann_kalman(kf_kappa, kf_P,
                           ego_speed, lead_speed, distance,
                           human_accel, target_distance,
                           belief: VelocityErrorBelief,
                           sigma_kappa=25.0,
                           kf_B=0.01, kf_Q=0.001,
                           kf_R=0.0001) -> dict:
    f_kappa = rbf_feature(distance, sigma_kappa)
    x_pred = kf_kappa * (1.0 - kf_B * f_kappa)
    P_pred = kf_P + kf_Q

    C = K_SPEED * lead_speed
    a_pred = compute_accel(ego_speed, lead_speed, distance,
                           x_pred, target_distance)
    innovation = human_accel - a_pred

    S = C ** 2 * P_pred + kf_R
    K_gain = P_pred * C / S
    x_hat_new = x_pred + K_gain * innovation
    P_new = max((1.0 - K_gain * C) * P_pred, 1e-8)

    log_probs = -(belief.candidates - x_hat_new) ** 2 / (2.0 * P_new)
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    belief.probabilities = probs / probs.sum()

    return {
        "kf_kappa": x_hat_new,
        "kf_P": P_new,
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_std": belief.std,
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


def infer_oracle(belief: VelocityErrorBelief, true_vel_err: float) -> dict:
    dists = np.abs(belief.candidates - true_vel_err)
    belief.probabilities = np.zeros_like(belief.probabilities)
    belief.probabilities[np.argmin(dists)] = 1.0
    return {
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_std": belief.std,
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


# ---------------------------------------------------------------------------
#  CBF intervention
# ---------------------------------------------------------------------------
LOOKAHEAD_N = 25


def barrier(distance, d_safe):
    return distance - d_safe


def cbf_max_accel(distance, ego_speed, lead_speed, d_safe, gamma, dt):
    h = barrier(distance, d_safe)
    return (2.0 * (lead_speed - ego_speed) * dt
            + gamma * (2.0 - gamma) * h) / (dt ** 2)


def intervene_cbf_single(human_accel, distance, ego_speed, lead_speed,
                         d_safe, gamma, dt):
    """One-step CBF safety filter.

    Returns the closest acceleration to the human's intended action
    that still satisfies the CBF safety constraint.
    """
    a_max = cbf_max_accel(distance, ego_speed, lead_speed,
                          d_safe, gamma, dt)
    a_max_clipped = float(np.clip(a_max, -MAX_ACCEL, MAX_ACCEL))
    if human_accel > a_max_clipped:
        # Project to nearest safe action: clamp to a_max
        return a_max_clipped, True, a_max
    return human_accel, False, a_max


def _sim_forward_static(a_exec, lead_speed, s0, kappa, dt, N,
                        target_distance):
    """Forward-simulate N steps with kappa held constant.

    Returns:
        states:   (N+1, 2) array of [distance, ego_speed] at each step.
        a_H_pred: (N,) array of predicted human accelerations under kappa.
    """
    states = np.zeros((N + 1, 2))
    states[0] = s0
    a_H_pred = np.zeros(N)

    for j in range(N):
        d_j, v_j = states[j]
        perceived = lead_speed * (1.0 + kappa)
        desired = max(0.0, perceived + K_DIST * (d_j - target_distance))
        a_H = float(np.clip(K_SPEED * (desired - v_j),
                            -MAX_ACCEL, MAX_ACCEL))
        a_H_pred[j] = a_H

        a = a_exec[j]
        v_next = max(0.0, v_j + a * dt)
        d_next = d_j + (lead_speed - v_j) * dt
        states[j + 1] = [d_next, v_next]

    return states, a_H_pred


def _sim_forward_evolving(a_exec, lead_speed, s0, kappa_init, dt, N,
                          target_distance, b_kappa=0.01, sigma_kappa=25.0):
    """Forward-simulate N steps with kappa evolving via RBF drift.

    At each step, kappa decays toward 0 based on the current distance:
        kappa_{j+1} = kappa_j * (1 - b_kappa * rbf(d_j))

    This predicts that the human's perception will improve over the
    lookahead horizon, making the safety filter less conservative.

    Returns:
        states:   (N+1, 2) array of [distance, ego_speed] at each step.
        a_H_pred: (N,) array of predicted human accelerations under evolving kappa.
    """
    states = np.zeros((N + 1, 2))
    states[0] = s0
    a_H_pred = np.zeros(N)
    kappa_j = kappa_init

    for j in range(N):
        d_j, v_j = states[j]
        # Evolve kappa based on current distance
        f = rbf_feature(d_j, sigma_kappa)
        kappa_j = kappa_j * (1.0 - b_kappa * f)

        perceived = lead_speed * (1.0 + kappa_j)
        desired = max(0.0, perceived + K_DIST * (d_j - target_distance))
        a_H = float(np.clip(K_SPEED * (desired - v_j),
                            -MAX_ACCEL, MAX_ACCEL))
        a_H_pred[j] = a_H

        a = a_exec[j]
        v_next = max(0.0, v_j + a * dt)
        d_next = d_j + (lead_speed - v_j) * dt
        states[j + 1] = [d_next, v_next]

    return states, a_H_pred


def intervene_lookahead(human_accel, distance, ego_speed, lead_speed,
                        kappa_hat, d_safe, gamma, dt, target_distance,
                        prev_sol=None,
                        evolving=False, b_kappa=0.01, sigma_kappa=25.0):
    """Predictive CBF safety filter (MPC).

    Minimises deviation between the executed acceleration sequence and the
    **predicted** human action sequence (under kappa_hat), subject to CBF
    safety constraints over the horizon.

    objective:   min ||a_exec - a_H_pred(kappa_hat)||^2
    subject to:  h(x_{j+1}) >= (1-gamma) * h(x_j)   for all j
                 v(x_{j+1}) >= 0                      for all j

    Different kappa_hat values produce different a_H_pred trajectories,
    so the optimal safe action depends on which kappa estimate is used.

    Args:
        kappa_hat:   Estimated velocity error for the forward model.
        evolving:    If True, kappa drifts via RBF during the lookahead
                     (cbf_kalman). If False, kappa is held static
                     (cbf_wmean, cbf_contmean, cbf_mode).
        b_kappa:     RBF drift rate (only used if evolving=True).
        sigma_kappa: RBF kernel width (only used if evolving=True).
    """
    N = LOOKAHEAD_N
    s0 = np.array([distance, ego_speed])

    if evolving:
        def _sim(a_exec):
            return _sim_forward_evolving(
                a_exec, lead_speed, s0, kappa_hat, dt, N,
                target_distance, b_kappa, sigma_kappa)
    else:
        def _sim(a_exec):
            return _sim_forward_static(
                a_exec, lead_speed, s0, kappa_hat, dt, N,
                target_distance)

    def objective(a_exec):
        _, a_H_pred = _sim(a_exec)
        return float(np.sum((a_exec - a_H_pred) ** 2))

    def cbf_con(a_exec):
        st, _ = _sim(a_exec)
        h_j = st[:-1, 0] - d_safe
        h_jp1 = st[1:, 0] - d_safe
        return h_jp1 - (1.0 - gamma) * h_j

    def vel_con(a_exec):
        st, _ = _sim(a_exec)
        return st[1:, 1]

    constraints = [
        {"type": "ineq", "fun": cbf_con},
        {"type": "ineq", "fun": vel_con},
    ]

    if prev_sol is not None and len(prev_sol) == N:
        x0 = np.empty(N)
        x0[:-1] = prev_sol[1:]
        x0[-1] = human_accel
    else:
        x0 = np.full(N, human_accel)

    result = scipy_minimize(
        objective, x0, method="SLSQP",
        bounds=[(-MAX_ACCEL, MAX_ACCEL)] * N,
        constraints=constraints,
        options={"maxiter": 50, "ftol": 1e-6},
    )

    if result.success:
        new_sol = result.x.copy()
        safe_accel = float(np.clip(result.x[0], -MAX_ACCEL, MAX_ACCEL))
    else:
        new_sol = None
        a_max = cbf_max_accel(distance, ego_speed, lead_speed,
                              d_safe, gamma, dt)
        safe_accel = float(np.clip(min(human_accel, a_max),
                                   -MAX_ACCEL, MAX_ACCEL))

    intervened = abs(safe_accel - human_accel) > 1e-3
    return safe_accel, intervened, new_sol


# ---------------------------------------------------------------------------
#  Multi-vehicle lookahead
# ---------------------------------------------------------------------------

@dataclass
class VehicleInfo:
    """Info about one non-ego vehicle for the multi-vehicle lookahead."""
    vid: int
    distance: float       # longitudinal distance ahead of ego
    lateral: float        # lateral offset (+ = right of ego heading)
    speed: float          # current speed
    kappa_hat: float      # estimated velocity error
    evolving: bool        # whether kappa evolves via RBF in the rollout
    # Future trajectory: list of (x, y, heading, velocity) per step.
    # When available, the MPC uses actual positions instead of
    # constant-speed extrapolation.
    future_traj: Optional[List[tuple]] = None


def _predict_vehicle_distance(v_info: VehicleInfo,
                              ego_state: "VehicleState",
                              a_exec: np.ndarray, dt: float, N: int,
                              b_kappa: float = 0.01,
                              sigma_kappa: float = 25.0) -> np.ndarray:
    """Predict longitudinal distance between ego and one vehicle over N steps.

    If the vehicle has a known future trajectory, use actual positions.
    Otherwise fall back to constant-speed extrapolation.

    Returns:
        distances: (N+1,) array of ego-to-vehicle longitudinal distance.
    """
    # Ego forward projection: position along heading
    ego_fwd = np.array([np.cos(ego_state.heading),
                        np.sin(ego_state.heading)])
    ego_x, ego_y = ego_state.x, ego_state.y
    v_ego = ego_state.velocity

    distances = np.zeros(N + 1)
    distances[0] = v_info.distance

    has_future = (v_info.future_traj is not None
                  and len(v_info.future_traj) >= N)

    for j in range(N):
        a = a_exec[j]
        v_ego = max(0.0, v_ego + a * dt)
        ego_x += v_ego * ego_fwd[0] * dt
        ego_y += v_ego * ego_fwd[1] * dt

        if has_future and j < len(v_info.future_traj):
            vx, vy = v_info.future_traj[j][0], v_info.future_traj[j][1]
            diff = np.array([vx - ego_x, vy - ego_y])
            distances[j + 1] = diff @ ego_fwd
        else:
            # Fallback: constant speed extrapolation
            distances[j + 1] = distances[j] + (v_info.speed - v_ego) * dt

    return distances


def intervene_lookahead_multi(human_accel, ego_state: "VehicleState",
                              vehicles: List[VehicleInfo],
                              lead_kappa_hat: float,
                              lead_speed: float, lead_distance: float,
                              d_safe, gamma, dt, target_distance,
                              prev_sol=None,
                              evolving=False, b_kappa=0.01, sigma_kappa=25.0):
    """Multi-vehicle predictive CBF safety filter.

    Same as intervene_lookahead but enforces CBF constraints against
    ALL vehicles ahead of the ego, not just the lead.  When vehicles
    have known future trajectories, the MPC uses actual positions
    rather than constant-speed extrapolation.

    The objective still minimises deviation from the predicted human
    actions (based on the lead vehicle's kappa), but the constraints
    ensure safety w.r.t. every nearby vehicle.
    """
    N = LOOKAHEAD_N
    ego_speed = ego_state.velocity
    s0_lead = np.array([lead_distance, ego_speed])

    # --- Objective: match predicted human actions (based on lead) ---
    if evolving:
        def _sim_lead(a_exec):
            return _sim_forward_evolving(
                a_exec, lead_speed, s0_lead, lead_kappa_hat, dt, N,
                target_distance, b_kappa, sigma_kappa)
    else:
        def _sim_lead(a_exec):
            return _sim_forward_static(
                a_exec, lead_speed, s0_lead, lead_kappa_hat, dt, N,
                target_distance)

    def objective(a_exec):
        _, a_H_pred = _sim_lead(a_exec)
        return float(np.sum((a_exec - a_H_pred) ** 2))

    # --- Constraints: CBF against every vehicle ---
    constraints = []

    for v_info in vehicles:
        def _make_cbf_con(vi):
            def con(a_exec):
                dists = _predict_vehicle_distance(
                    vi, ego_state, a_exec, dt, N, b_kappa, sigma_kappa)
                h_j = dists[:-1] - d_safe
                h_jp1 = dists[1:] - d_safe
                return h_jp1 - (1.0 - gamma) * h_j
            return con

        constraints.append({"type": "ineq", "fun": _make_cbf_con(v_info)})

    # Single velocity constraint (shared across all vehicles)
    def vel_con(a_exec):
        v = ego_speed
        vels = np.zeros(N)
        for j in range(N):
            v = max(0.0, v + a_exec[j] * dt)
            vels[j] = v
        return vels

    constraints.append({"type": "ineq", "fun": vel_con})

    # --- Solve ---
    if prev_sol is not None and len(prev_sol) == N:
        x0 = np.empty(N)
        x0[:-1] = prev_sol[1:]
        x0[-1] = human_accel
    else:
        x0 = np.full(N, human_accel)

    result = scipy_minimize(
        objective, x0, method="SLSQP",
        bounds=[(-MAX_ACCEL, MAX_ACCEL)] * N,
        constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-6},
    )

    if result.success:
        new_sol = result.x.copy()
        safe_accel = float(np.clip(result.x[0], -MAX_ACCEL, MAX_ACCEL))
    else:
        new_sol = None
        # Fallback: single-step CBF against closest vehicle
        min_a_max = MAX_ACCEL
        for vi in vehicles:
            a_max = cbf_max_accel(vi.distance, ego_speed, vi.speed,
                                  d_safe, gamma, dt)
            min_a_max = min(min_a_max, a_max)
        safe_accel = float(np.clip(min(human_accel, min_a_max),
                                   -MAX_ACCEL, MAX_ACCEL))

    intervened = abs(safe_accel - human_accel) > 1e-3
    return safe_accel, intervened, new_sol


# ---------------------------------------------------------------------------
#  CarFollowController  — the main callable
# ---------------------------------------------------------------------------

class CarFollowController:
    """Ego controller mirroring CarFollowAgent from igp2.

    Instantiate with config, then call as controller(obs) → Action.
    """

    def __init__(self, *,
                 velocity_errors: Dict[int, float] = None,
                 target_distance: float = 12.0,
                 d_safe: float = 8.0,
                 beta: float = 1.0,
                 gamma: float = 0.99,
                 inference: str = "none",
                 intervention: str = "none",
                 human: str = "static",
                 b_kappa: float = 0.01,
                 sigma_kappa: float = 25.0):
        self.target_distance = target_distance
        self.d_safe = d_safe
        self.beta = beta
        self.gamma = gamma
        self.inference_type = inference
        self.intervention_type = intervention
        self.human_type = human
        self.b_kappa = b_kappa
        self.sigma_kappa = sigma_kappa

        # True velocity errors per non-ego vehicle (from config)
        self._true_vel_errors: Dict[int, float] = dict(
            velocity_errors or {})

        # Assistive system's inferred belief (one per tracked vehicle)
        self._beliefs: Dict[int, VelocityErrorBelief] = {}
        for vid in self._true_vel_errors:
            self._beliefs[vid] = VelocityErrorBelief()

        # Kalman filter state per vehicle
        self._kf_kappa: Dict[int, float] = {v: 0.0 for v in self._true_vel_errors}
        self._kf_P: Dict[int, float] = {v: 0.5 for v in self._true_vel_errors}
        self._kf_Q = 0.001
        self._kf_R = 0.01    # observation noise (matches CarFollowAgent)
        self._kf_B = 0.01

        # Lookahead warm-start
        self._prev_sol: Optional[np.ndarray] = None

        # Per-step diagnostics (readable by the runner / plotter)
        self.last_step_info: Dict = {}

        # Step counter
        self._step = 0

    def _ensure_belief(self, vid: int):
        """Lazily create belief for a vehicle not in initial config."""
        if vid not in self._beliefs:
            self._beliefs[vid] = VelocityErrorBelief()
            self._true_vel_errors.setdefault(vid, 0.0)
            self._kf_kappa.setdefault(vid, 0.0)
            self._kf_P.setdefault(vid, 0.5)

    def _run_inference_for_vehicle(self, vid: int, ego_speed: float,
                                    vehicle_speed: float, distance: float,
                                    human_accel: float,
                                    is_lead: bool = True) -> dict:
        """Run belief inference for one vehicle.

        When *is_lead* is True, the human's observed action is informative
        about this vehicle's kappa, so we run the full observation model.

        When *is_lead* is False, the human's action was driven by a
        different vehicle — changing kappa for this vehicle doesn't change
        the predicted action, so all likelihoods are equal.  For Boltzmann
        inference we skip the update (posterior = prior).  For Kalman we
        run the prediction step only (kappa drifts, variance grows) but
        skip the measurement update.
        """
        belief = self._beliefs.get(vid)
        if belief is None:
            return {}

        # --- Non-lead: prediction only, no observation update ---
        if not is_lead:
            if self.inference_type == "boltzmann_kalman":
                # Kalman prediction step only: drift kappa, grow variance
                f_kappa = rbf_feature(distance, self.sigma_kappa)
                x_pred = self._kf_kappa[vid] * (1.0 - self._kf_B * f_kappa)
                P_pred = self._kf_P[vid] + self._kf_Q
                self._kf_kappa[vid] = x_pred
                self._kf_P[vid] = P_pred
                # Project onto discrete belief (wider distribution as P grows)
                log_probs = -(belief.candidates - x_pred) ** 2 / (2.0 * P_pred)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            # For all Boltzmann variants + oracle: posterior unchanged
            return {
                "inferred_mode": belief.mode,
                "inferred_mean": belief.mean,
                "inferred_std": belief.std,
                "inferred_dist": dict(zip(
                    np.round(belief.candidates, 2),
                    np.round(belief.probabilities, 4))),
                "kf_kappa": self._kf_kappa.get(vid),
                "kf_P": self._kf_P.get(vid),
            }

        # --- Lead vehicle: full observation model ---
        common = dict(ego_speed=ego_speed, lead_speed=vehicle_speed,
                      distance=distance, human_accel=human_accel,
                      target_distance=self.target_distance)

        if self.inference_type == "boltzmann_reactive":
            return infer_boltzmann_reactive(
                belief, beta=self.beta, **common)
        elif self.inference_type == "boltzmann_reactive_noprior":
            return infer_boltzmann_reactive(
                belief, beta=self.beta, use_prior=False, **common)
        elif self.inference_type == "boltzmann_reactive_floormix":
            return infer_boltzmann_reactive(
                belief, beta=self.beta, floor_mix=True, **common)
        elif self.inference_type == "boltzmann_kalman":
            diag = infer_boltzmann_kalman(
                self._kf_kappa[vid], self._kf_P[vid],
                belief=belief,
                sigma_kappa=self.sigma_kappa,
                kf_B=self._kf_B, kf_Q=self._kf_Q,
                kf_R=self._kf_R, **common)
            self._kf_kappa[vid] = diag["kf_kappa"]
            self._kf_P[vid] = diag["kf_P"]
            return diag
        elif self.inference_type == "oracle":
            return infer_oracle(
                belief, self._true_vel_errors.get(vid, 0.0))
        return {}

    def _get_kappa_hat(self, vid: int) -> float:
        """Get the kappa estimate for a vehicle based on intervention type."""
        belief = self._beliefs.get(vid)
        if self.intervention_type == "cbf_mode":
            return belief.mode if belief else 0.0
        elif self.intervention_type in ("cbf_wmean", "cbf_lookahead"):
            return belief.mean if belief else 0.0
        elif self.intervention_type in ("cbf_contmean", "cbf_kalman"):
            return self._kf_kappa.get(vid, 0.0)
        elif self.intervention_type == "cbf_chance":
            return belief.mode if belief else 0.0
        return 0.0

    def __call__(self, obs: Observation) -> Action:
        self._step += 1
        ego = obs.ego
        dt = obs.dt

        # --- Find lead vehicle (for human controller) ---
        lead_id, lead_state, lead_distance = get_lead_vehicle(ego, obs.others)

        # --- Find ALL vehicles ahead (for beliefs + intervention) ---
        vehicles_ahead = get_all_vehicles_ahead(ego, obs.others)
        for vid in vehicles_ahead:
            self._ensure_belief(vid)

        # --- Human belief evolution for ALL visible vehicles ---
        if self.human_type == "rbf":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                old_kappa = self._true_vel_errors.get(vid, 0.0)
                new_kappa = evolve_vel_err(
                    old_kappa, along, self.b_kappa, self.sigma_kappa)
                self._true_vel_errors[vid] = new_kappa

        # --- Human controller (reacts to lead vehicle only) ---
        if lead_state is None:
            human_accel = 0.0
            perceived_lead_speed = None
            desired_speed = ego.velocity
        else:
            vel_err = self._true_vel_errors.get(lead_id, 0.0)
            human_accel = compute_accel(
                ego.velocity, lead_state.velocity, lead_distance,
                vel_err, self.target_distance)
            perceived_lead_speed = lead_state.velocity * (1.0 + vel_err)
            desired_speed = max(
                0.0, perceived_lead_speed
                + K_DIST * (lead_distance - self.target_distance))

        self.last_step_info = {
            "step": self._step,
            "lead_id": lead_id,
            "distance": lead_distance if lead_distance < float("inf") else None,
            "ego_speed": ego.velocity,
            "lead_speed": lead_state.velocity if lead_state else None,
            "v_perceived": perceived_lead_speed,
            "v_desired": desired_speed,
            "accel": human_accel,
            "human_vel_err": (self._true_vel_errors.get(lead_id, 0.0)
                              if lead_id is not None else None),
        }

        # --- Inference for ALL vehicles ahead ---
        # Lead vehicle: full observation model (action is informative)
        # Non-lead: prediction only (action carries no info about this vehicle)
        per_vehicle_diag = {}
        if self.inference_type != "none":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                diag = self._run_inference_for_vehicle(
                    vid, ego.velocity, vs.velocity, along, human_accel,
                    is_lead=(vid == lead_id))
                diag["human_vel_err"] = self._true_vel_errors.get(vid, 0.0)
                diag["distance"] = along
                per_vehicle_diag[vid] = diag
                # Store lead vehicle diagnostics at top level for backward compat
                if vid == lead_id:
                    self.last_step_info.update(diag)
        self.last_step_info["per_vehicle"] = per_vehicle_diag

        # --- Intervention (CBF safety filter) ---
        executed_accel = human_accel
        intervened = False
        a_cbf_max = None

        if (lead_state is not None and lead_distance < float("inf")
                and self.intervention_type != "none"):
            a_cbf_max = cbf_max_accel(
                lead_distance, ego.velocity, lead_state.velocity,
                self.d_safe, self.gamma, dt)

            if self.intervention_type == "cbf_single":
                # Single-step CBF against closest vehicle ahead in-lane
                # Check all vehicles, take the most restrictive
                min_a_max = MAX_ACCEL
                for vid, (vs, along, lateral) in vehicles_ahead.items():
                    if abs(lateral) < LANE_WIDTH * 0.8:
                        a_m = cbf_max_accel(along, ego.velocity, vs.velocity,
                                            self.d_safe, self.gamma, dt)
                        min_a_max = min(min_a_max, a_m)
                a_cbf_max = min_a_max
                a_max_clipped = float(np.clip(min_a_max, -MAX_ACCEL, MAX_ACCEL))
                if human_accel > a_max_clipped:
                    executed_accel = a_max_clipped
                    intervened = True

            elif self.intervention_type == "always_policy":
                executed_accel = compute_accel(
                    ego.velocity, lead_state.velocity, lead_distance,
                    0.0, self.target_distance)
                intervened = True

            elif self.intervention_type in ("cbf_lookahead", "cbf_mode",
                                            "cbf_wmean", "cbf_contmean",
                                            "cbf_kalman", "cbf_chance"):
                use_evolving = (self.intervention_type == "cbf_kalman")

                # Build VehicleInfo for all vehicles ahead,
                # including their known future trajectories
                v_infos = []
                for vid, (vs, along, lateral) in vehicles_ahead.items():
                    kh = self._get_kappa_hat(vid)
                    ft = obs.future_trajectories.get(vid)
                    v_infos.append(VehicleInfo(
                        vid=vid, distance=along, lateral=lateral,
                        speed=vs.velocity, kappa_hat=kh,
                        evolving=use_evolving,
                        future_traj=ft))

                lead_kappa = self._get_kappa_hat(lead_id)

                executed_accel, intervened, self._prev_sol = \
                    intervene_lookahead_multi(
                        human_accel, ego, v_infos,
                        lead_kappa_hat=lead_kappa,
                        lead_speed=lead_state.velocity,
                        lead_distance=lead_distance,
                        d_safe=self.d_safe, gamma=self.gamma, dt=dt,
                        target_distance=self.target_distance,
                        prev_sol=self._prev_sol,
                        evolving=use_evolving,
                        b_kappa=self.b_kappa,
                        sigma_kappa=self.sigma_kappa)

        self.last_step_info["intervened"] = intervened
        self.last_step_info["a_max_safe"] = (
            float(a_cbf_max) if a_cbf_max is not None else None)
        self.last_step_info["executed_accel"] = executed_accel

        return Action(acceleration=executed_accel, steer_angle=0.0)

    # Convenience accessors for the plotter
    @property
    def beliefs(self) -> Dict[int, VelocityErrorBelief]:
        return self._beliefs

    @property
    def true_vel_errors(self) -> Dict[int, float]:
        return self._true_vel_errors
