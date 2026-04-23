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


def compute_accel_additive(ego_speed: float, lead_speed: float,
                           distance: float, vel_err_additive: float,
                           target_distance: float) -> float:
    """Proportional controller with additive velocity-error.

    v_desired = (v_lead + epsilon) + k_d * (d - d*)
    a = k_v * (v_desired - v_ego)
    """
    perceived = lead_speed + vel_err_additive
    dist_error = distance - target_distance
    desired = max(0.0, perceived + K_DIST * dist_error)
    accel = K_SPEED * (desired - ego_speed)
    return float(np.clip(accel, -MAX_ACCEL, MAX_ACCEL))


def _parse_numeric(s: str) -> float:
    """Parse a numeric string where leading zero means decimal.

    '1' → 1.0, '05' → 0.5, '15' → 1.5, '025' → 0.25, '2' → 2.0
    """
    if '.' in s:
        return float(s)
    if len(s) > 1 and s[0] == '0':
        return float(s[0] + '.' + s[1:])
    return float(s)


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
        "inferred_std_continuous": None,  # no Kalman state
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
        "inferred_std_continuous": np.sqrt(P_new),
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


def infer_revertkalman(kf_vhat, kf_P,
                       ego_speed, lead_speed, distance,
                       human_accel, target_distance,
                       belief: VelocityErrorBelief,
                       kf_Q=0.1, kf_R=1.0,
                       revert_alpha=0.9,
                       a_lead=0.0, dt=0.1) -> dict:
    """Kalman filter with mean-reverting process model in v_hat space.

    State x = v_hat (human's perceived lead velocity, m/s).

    Process:  v_hat_{k|k-1} = A_k * v_hat_{k-1|k-1} + B_k + w_k
              A_k = alpha
              B_k = (1 - alpha) * v_lead + a_lead * dt
              P_{k|k-1} = A^2 * P_{k-1|k-1} + Q

    Obs:      a = K_v * (max(0, v_hat + K_d*(d - d*)) - v_ego) + eta
              C = da/dv_hat = K_v = 1.0
    """
    # --- Predict ---
    A = revert_alpha
    B = (1.0 - revert_alpha) * lead_speed + a_lead * dt
    vhat_pred = A * kf_vhat + B
    P_pred = A ** 2 * kf_P + kf_Q

    # --- Observation model (v_hat as perceived velocity) ---
    C = K_SPEED  # = 1.0
    dist_error = distance - target_distance
    desired = max(0.0, vhat_pred + K_DIST * dist_error)
    a_pred = float(np.clip(C * (desired - ego_speed), -MAX_ACCEL, MAX_ACCEL))
    innovation = human_accel - a_pred

    # --- Update ---
    S = C ** 2 * P_pred + kf_R
    K_gain = P_pred * C / S
    norm_innovation = innovation / np.sqrt(S) if S > 0 else 0.0
    vhat_new = vhat_pred + K_gain * innovation
    P_new = max((1.0 - K_gain * C) * P_pred, 1e-8)

    # --- Convert to epsilon / kappa for the rest of the pipeline ---
    v = max(lead_speed, 0.1)
    eps = vhat_new - lead_speed
    kappa_hat = eps / v
    P_kappa = P_new / (v ** 2)

    log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    belief.probabilities = probs / probs.sum()

    return {
        "kf_vhat": vhat_new,
        "kf_epsilon": eps,
        "kf_P": P_new,
        "kf_kappa": kappa_hat,
        "norm_innovation": float(norm_innovation),
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_std": belief.std,
        "inferred_std_continuous": np.sqrt(P_kappa),
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


def infer_lingapkalman(kf_vhat, kf_P,
                       ego_speed, lead_speed, distance,
                       human_accel, target_distance,
                       belief: VelocityErrorBelief,
                       kf_Q=0.1, kf_R=1.0,
                       revert_alpha=0.9, lingap_beta=0.1,
                       a_lead=0.0, dt=0.1) -> dict:
    """Kalman filter with gap-dependent process model in v_hat space.

    State x = v_hat (human's perceived lead velocity, m/s).

    Process:  v_hat_{k|k-1} = A_k * v_hat_{k-1|k-1} + B_k + w_k
              A_k = alpha
              B_k = (1 - alpha) * v_lead + a_lead * dt + beta * gap
              P_{k|k-1} = A^2 * P_{k-1|k-1} + Q

    The beta * gap term models distance-dependent perception error:
    the further the lead, the larger the velocity estimation bias.
    """
    # --- Predict ---
    A = revert_alpha
    gap = max(distance, 0.0)
    B = (1.0 - revert_alpha) * lead_speed + a_lead * dt + lingap_beta * gap
    vhat_pred = A * kf_vhat + B
    P_pred = A ** 2 * kf_P + kf_Q

    # --- Observation model ---
    C = K_SPEED  # = 1.0
    dist_error = distance - target_distance
    desired = max(0.0, vhat_pred + K_DIST * dist_error)
    a_pred = float(np.clip(C * (desired - ego_speed), -MAX_ACCEL, MAX_ACCEL))
    innovation = human_accel - a_pred

    # --- Update ---
    S = C ** 2 * P_pred + kf_R
    K_gain = P_pred * C / S
    norm_innovation = innovation / np.sqrt(S) if S > 0 else 0.0
    vhat_new = vhat_pred + K_gain * innovation
    P_new = max((1.0 - K_gain * C) * P_pred, 1e-8)

    # --- Convert to epsilon / kappa ---
    v = max(lead_speed, 0.1)
    eps = vhat_new - lead_speed
    kappa_hat = eps / v
    P_kappa = P_new / (v ** 2)

    log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    belief.probabilities = probs / probs.sum()

    return {
        "kf_vhat": vhat_new,
        "kf_epsilon": eps,
        "kf_P": P_new,
        "kf_kappa": kappa_hat,
        "norm_innovation": float(norm_innovation),
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_std": belief.std,
        "inferred_std_continuous": np.sqrt(P_kappa),
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
        "inferred_std_continuous": None,  # no Kalman state
        "inferred_dist": dict(zip(
            np.round(belief.candidates, 2),
            np.round(belief.probabilities, 4))),
    }


# ---------------------------------------------------------------------------
#  CBF intervention
# ---------------------------------------------------------------------------
LOOKAHEAD_N = 50


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


def _sim_forward_multi(a_exec, ego_state, vehicles, dt, N,
                       target_distance, desired_speed=15.0,
                       evolving=False, b_kappa=0.01, sigma_kappa=25.0):
    """Forward-simulate N steps with dynamic lead determination.

    At each step, determines which vehicle (if any) is the lead based
    on future trajectories and lane position — exactly mirroring the
    real human controller logic:
      - Lead exists → compute_accel(v_ego, v_lead, d, kappa, d*)
      - No lead    → K_SPEED * (desired_speed - v_ego)

    Args:
        a_exec:    (N,) optimizer acceleration sequence.
        ego_state: Current ego VehicleState.
        vehicles:  List of VehicleInfo with future trajectories.
        evolving:  If True, kappa drifts via RBF at each step.

    Returns:
        ego_positions: (N+1, 2) array of [ego_x, ego_speed] at each step.
        a_H_pred:      (N,) array of predicted human accelerations.
    """
    ego_fwd = np.array([np.cos(ego_state.heading),
                        np.sin(ego_state.heading)])
    ego_right = np.array([-ego_fwd[1], ego_fwd[0]])

    ego_x = ego_state.x
    ego_y = ego_state.y
    v_ego = ego_state.velocity

    ego_positions = np.zeros((N + 1, 2))
    ego_positions[0] = [ego_x, v_ego]
    a_H_pred = np.zeros(N)

    # Per-vehicle kappa state (for evolving)
    kappas = {vi.vid: vi.kappa_hat for vi in vehicles}

    for j in range(N):
        # Advance ego position
        a = a_exec[j]
        v_ego = max(0.0, v_ego + a * dt)
        ego_x += v_ego * ego_fwd[0] * dt
        ego_y += v_ego * ego_fwd[1] * dt

        # Determine lead vehicle at this horizon step
        best_dist = float("inf")
        best_vid = None
        best_speed = None

        for vi in vehicles:
            if vi.future_traj is not None and j < len(vi.future_traj):
                vx, vy = vi.future_traj[j][0], vi.future_traj[j][1]
                v_speed = vi.future_traj[j][3]
            else:
                # Constant speed extrapolation
                vx = vi.distance * ego_fwd[0] + ego_state.x + vi.speed * ego_fwd[0] * (j + 1) * dt
                vy = vi.lateral * ego_right[1] + ego_state.y + vi.speed * ego_fwd[1] * (j + 1) * dt
                v_speed = vi.speed

            diff = np.array([vx - ego_x, vy - ego_y])
            along = diff @ ego_fwd
            lat = abs(diff @ ego_right)

            if along > 0 and _in_lane(lat) and along < best_dist:
                best_dist = along
                best_vid = vi.vid
                best_speed = v_speed

        # Compute predicted human action
        if best_vid is not None:
            kappa = kappas.get(best_vid, 0.0)
            if evolving:
                f = rbf_feature(best_dist, sigma_kappa)
                kappa = kappa * (1.0 - b_kappa * f)
                kappas[best_vid] = kappa

            perceived = best_speed * (1.0 + kappa)
            desired = max(0.0, perceived + K_DIST * (best_dist - target_distance))
            a_H = float(np.clip(K_SPEED * (desired - v_ego),
                                -MAX_ACCEL, MAX_ACCEL))
        else:
            # No lead → cruise at desired speed
            a_H = float(np.clip(K_SPEED * (desired_speed - v_ego),
                                -MAX_ACCEL, MAX_ACCEL))

        a_H_pred[j] = a_H
        ego_positions[j + 1] = [ego_x, v_ego]

    return ego_positions, a_H_pred


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


def _predict_vehicle_distance_and_in_lane(
        v_info: VehicleInfo, ego_state: "VehicleState",
        a_exec: np.ndarray, dt: float, N: int,
        b_kappa: float = 0.01, sigma_kappa: float = 25.0,
) -> tuple:
    """Predict longitudinal distance AND per-step lane membership between
    ego and one vehicle over N steps.

    Uses the vehicle's known future trajectory to determine at each
    timestep whether it is in the ego's lane.

    Returns:
        distances: (N+1,) array of ego-to-vehicle longitudinal distance.
        in_lane:   (N+1,) boolean array — True when vehicle is in ego's lane.
    """
    ego_fwd = np.array([np.cos(ego_state.heading),
                        np.sin(ego_state.heading)])
    ego_right = np.array([-ego_fwd[1], ego_fwd[0]])
    ego_x, ego_y = ego_state.x, ego_state.y
    v_ego = ego_state.velocity

    distances = np.zeros(N + 1)
    in_lane = np.zeros(N + 1, dtype=bool)
    distances[0] = v_info.distance
    in_lane[0] = _in_lane(v_info.lateral)

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
            lat = abs(diff @ ego_right)
            in_lane[j + 1] = _in_lane(lat)
        else:
            distances[j + 1] = distances[j] + (v_info.speed - v_ego) * dt
            in_lane[j + 1] = in_lane[j]  # hold last known state

    return distances, in_lane


def _in_lane(lateral_offset: float,
             lane_width: float = LANE_WIDTH) -> bool:
    """Binary lane check: is the vehicle in the ego's lane?"""
    return abs(lateral_offset) < lane_width * 0.8


def intervene_lookahead_multi(human_accel, ego_state: "VehicleState",
                              vehicles: List[VehicleInfo],
                              d_safe, gamma, dt, target_distance,
                              desired_speed=15.0,
                              prev_sol=None,
                              evolving=False, b_kappa=0.01, sigma_kappa=25.0,
                              ):
    """Multi-vehicle predictive CBF safety filter.

    Enforces CBF constraints against ALL vehicles ahead of the ego.
    The objective minimises deviation from the predicted human actions,
    where the forward model dynamically determines the lead vehicle
    at each horizon step (mirroring the real human controller).
    """
    N = LOOKAHEAD_N
    ego_speed = ego_state.velocity

    # --- Objective: match predicted human actions ---
    # Uses dynamic lead determination at each horizon step, mirroring
    # the actual human controller (cruise when no lead, follow when lead).
    def objective(a_exec):
        _, a_H_pred = _sim_forward_multi(
            a_exec, ego_state, vehicles, dt, N,
            target_distance, desired_speed=desired_speed,
            evolving=evolving, b_kappa=b_kappa, sigma_kappa=sigma_kappa)
        return float(np.sum((a_exec - a_H_pred) ** 2))

    # --- Constraints: CBF against every vehicle ---
    constraints = []

    for v_info in vehicles:
        def _make_cbf_con(vi):
            def con(a_exec):
                dists, in_lane_flags = _predict_vehicle_distance_and_in_lane(
                    vi, ego_state, a_exec, dt, N, b_kappa, sigma_kappa)
                # Only enforce d_safe at timesteps where the vehicle
                # is in the ego's lane.  When out-of-lane, d_safe=0
                # so the constraint is trivially satisfied.
                d_safe_t = np.where(in_lane_flags, d_safe, 0.0)
                h_j = dists[:-1] - d_safe_t[:-1]
                h_jp1 = dists[1:] - d_safe_t[1:]
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
                 desired_speed: float = 15.0,
                 beta: float = 1.0,
                 gamma: float = 0.99,
                 inference: str = "none",
                 intervention: str = "none",
                 human: str = "static",
                 b_kappa: float = 0.01,
                 sigma_kappa: float = 25.0,
                 kf_Q: float = 0.001,
                 kf_R: float = 0.01,
                 kf_alpha: float = 0.0,
                 kf_epsilon_init: float = 0.0,
                 kf_revert_alpha: float = 0.9,
                 kf_lingap_beta: float = 0.1,
                 # Human model parameters (override name-encoded values)
                 human_sigma: float = None,
                 human_mu: float = None,
                 human_walk_Q: float = None,
                 human_revert_alpha: float = 0.9,
                 human_lingap_beta: float = 0.1,
                 action_noise_std: float = 0.0):
        self.target_distance = target_distance
        self.d_safe = d_safe
        self.desired_speed = desired_speed
        self.beta = beta
        self.gamma = gamma
        self.inference_type = inference
        self.intervention_type = intervention
        self.b_kappa = b_kappa
        self.sigma_kappa = sigma_kappa
        self.action_noise_std = action_noise_std

        # Resolve human type and parameters.
        # New-style: human="walk" + human_walk_Q=0.1
        # Old-style: human="walk_01" (still supported, config params override)
        self.human_type = human
        self._human_sigma = human_sigma
        self._human_mu = human_mu
        self._human_walk_Q = human_walk_Q
        self._human_revert_alpha = human_revert_alpha
        self._human_lingap_beta = human_lingap_beta
        self._kf_revert_alpha = kf_revert_alpha
        self._kf_lingap_beta = kf_lingap_beta
        self._resolve_human_params()

        # True velocity errors per non-ego vehicle (from config)
        self._true_vel_errors: Dict[int, float] = dict(
            velocity_errors or {})

        # Assistive system's inferred belief (one per tracked vehicle)
        self._beliefs: Dict[int, VelocityErrorBelief] = {}
        for vid in self._true_vel_errors:
            self._beliefs[vid] = VelocityErrorBelief()

        # Kalman filter state per vehicle
        self._kf_kappa: Dict[int, float] = {v: 0.0 for v in self._true_vel_errors}
        self._kf_P: Dict[int, float] = {v: 10.0 for v in self._true_vel_errors}
        self._kf_epsilon: Dict[int, float] = {v: kf_epsilon_init for v in self._true_vel_errors}
        # v_hat state for revertkalman (initialized lazily on first observation)
        self._kf_vhat: Dict[int, Optional[float]] = {v: None for v in self._true_vel_errors}
        self._kf_epsilon_init = kf_epsilon_init
        self._kf_Q = kf_Q
        self._kf_R = kf_R
        self._kf_B = 0.01
        self._kf_alpha = kf_alpha

        # Random walk state for walk_* human (per vehicle, in m/s)
        self._walk_epsilon: Dict[int, float] = {v: 0.0 for v in self._true_vel_errors}

        # Lead acceleration tracking (per vehicle)
        self._prev_lead_speed: Dict[int, float] = {}

        # Lookahead warm-start
        self._prev_sol: Optional[np.ndarray] = None

        # Per-step diagnostics (readable by the runner / plotter)
        self.last_step_info: Dict = {}

        # Step counter
        self._step = 0

    def _resolve_human_params(self):
        """Parse human type name into canonical type + params.

        Supports both:
          New-style: human="walk", human_walk_Q=0.1
          Old-style: human="walk_01"  (params encoded in name)
        Config params (human_sigma, etc.) override name-encoded values.
        """
        h = self.human_type

        if h.startswith("gaussian_meanstd_"):
            parts = h.split("gaussian_meanstd_")[1].split("_")
            if self._human_mu is None:
                self._human_mu = _parse_numeric(parts[0])
            if self._human_sigma is None:
                self._human_sigma = (
                    _parse_numeric(parts[1]) if len(parts) > 1 else 1.0)
            self.human_type = "gaussian_mean"
        elif h.startswith("gaussian_std_"):
            if self._human_sigma is None:
                self._human_sigma = _parse_numeric(
                    h.split("gaussian_std_")[1])
            self.human_type = "gaussian"
        elif h.startswith("walk_"):
            if self._human_walk_Q is None:
                self._human_walk_Q = _parse_numeric(
                    h.split("walk_")[1])
            self.human_type = "walk"
        elif h == "gaussian":
            pass  # params from config
        elif h == "gaussian_mean":
            pass
        elif h == "walk":
            pass
        elif h == "revert":
            pass
        elif h == "lingap":
            pass

        # Defaults for params not set by name or config
        if self._human_sigma is None:
            self._human_sigma = 1.0
        if self._human_mu is None:
            self._human_mu = 0.0
        if self._human_walk_Q is None:
            self._human_walk_Q = 0.1

    def _ensure_belief(self, vid: int):
        """Lazily create belief for a vehicle not in initial config."""
        if vid not in self._beliefs:
            self._beliefs[vid] = VelocityErrorBelief()
            self._true_vel_errors.setdefault(vid, 0.0)
            self._kf_kappa.setdefault(vid, 0.0)
            self._kf_P.setdefault(vid, 10.0)
            self._kf_epsilon.setdefault(vid, 0.0)
            self._kf_vhat.setdefault(vid, None)

    def _run_inference_for_vehicle(self, vid: int, ego_speed: float,
                                    vehicle_speed: float, distance: float,
                                    human_accel: float,
                                    is_lead: bool = True,
                                    obs_dt: float = 0.1) -> dict:
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
                log_probs = -(belief.candidates - x_pred) ** 2 / (2.0 * P_pred)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            elif self.inference_type == "walkkalman":
                # A=1 prediction: v_hat_pred = v_hat + a_lead * dt
                if self._kf_vhat.get(vid) is None:
                    self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
                prev_v = self._prev_lead_speed.get(vid)
                a_lead = (vehicle_speed - prev_v) / obs_dt if prev_v is not None else 0.0
                self._prev_lead_speed[vid] = vehicle_speed
                vhat_pred = self._kf_vhat[vid] + a_lead * obs_dt
                P_pred = self._kf_P[vid] + self._kf_Q
                self._kf_vhat[vid] = vhat_pred
                self._kf_P[vid] = P_pred
                eps_pred = vhat_pred - vehicle_speed
                self._kf_epsilon[vid] = eps_pred
                v = max(vehicle_speed, 0.1)
                kappa_hat = eps_pred / v
                P_kappa = P_pred / (v ** 2)
                self._kf_kappa[vid] = kappa_hat
                log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            elif self.inference_type == "iidkalman":
                # A=0 prediction: v_hat_pred = v_lead + a_lead*dt (forgets prev)
                if self._kf_vhat.get(vid) is None:
                    self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
                prev_v = self._prev_lead_speed.get(vid)
                a_lead = (vehicle_speed - prev_v) / obs_dt if prev_v is not None else 0.0
                self._prev_lead_speed[vid] = vehicle_speed
                # A=0: vhat_pred = 0*vhat + 1*v_lead + a_lead*dt
                vhat_pred = vehicle_speed + a_lead * obs_dt
                P_pred = self._kf_Q  # A²=0, so just Q
                self._kf_vhat[vid] = vhat_pred
                self._kf_P[vid] = P_pred
                eps_pred = vhat_pred - vehicle_speed
                self._kf_epsilon[vid] = eps_pred
                v = max(vehicle_speed, 0.1)
                kappa_hat = eps_pred / v
                P_kappa = P_pred / (v ** 2)
                self._kf_kappa[vid] = kappa_hat
                log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            elif self.inference_type == "revertkalman":
                # v_hat prediction: v_hat_pred = alpha*v_hat + (1-alpha)*v_lead + a_lead*dt
                A = self._kf_revert_alpha
                if self._kf_vhat.get(vid) is None:
                    self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
                prev_v = self._prev_lead_speed.get(vid)
                a_lead = (vehicle_speed - prev_v) / obs_dt if prev_v is not None else 0.0
                self._prev_lead_speed[vid] = vehicle_speed
                B = (1.0 - A) * vehicle_speed + a_lead * obs_dt
                vhat_pred = A * self._kf_vhat[vid] + B
                P_pred = A ** 2 * self._kf_P[vid] + self._kf_Q
                self._kf_vhat[vid] = vhat_pred
                self._kf_P[vid] = P_pred
                eps_pred = vhat_pred - vehicle_speed
                self._kf_epsilon[vid] = eps_pred
                v = max(vehicle_speed, 0.1)
                kappa_hat = eps_pred / v
                P_kappa = P_pred / (v ** 2)
                self._kf_kappa[vid] = kappa_hat
                log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            elif self.inference_type == "lingapkalman":
                # v_hat prediction with gap-dependent term
                A = self._kf_revert_alpha
                if self._kf_vhat.get(vid) is None:
                    self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
                prev_v = self._prev_lead_speed.get(vid)
                a_lead = (vehicle_speed - prev_v) / obs_dt if prev_v is not None else 0.0
                self._prev_lead_speed[vid] = vehicle_speed
                gap = max(distance, 0.0)
                B = (1.0 - A) * vehicle_speed + a_lead * obs_dt + self._kf_lingap_beta * gap
                vhat_pred = A * self._kf_vhat[vid] + B
                P_pred = A ** 2 * self._kf_P[vid] + self._kf_Q
                self._kf_vhat[vid] = vhat_pred
                self._kf_P[vid] = P_pred
                eps_pred = vhat_pred - vehicle_speed
                self._kf_epsilon[vid] = eps_pred
                v = max(vehicle_speed, 0.1)
                kappa_hat = eps_pred / v
                P_kappa = P_pred / (v ** 2)
                self._kf_kappa[vid] = kappa_hat
                log_probs = -(belief.candidates - kappa_hat) ** 2 / (2.0 * P_kappa)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                belief.probabilities = probs / probs.sum()
            # For all Boltzmann variants + oracle: posterior unchanged
            kf_k = self._kf_kappa.get(vid)
            kf_p = self._kf_P.get(vid)
            return {
                "inferred_mode": belief.mode,
                "inferred_mean": belief.mean,
                "inferred_std": belief.std,
                "inferred_std_continuous": (
                    np.sqrt(kf_p) if kf_p is not None else None),
                "inferred_dist": dict(zip(
                    np.round(belief.candidates, 2),
                    np.round(belief.probabilities, 4))),
                "kf_kappa": kf_k,
                "kf_P": kf_p,
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
        elif self.inference_type == "walkkalman":
            # A=1: v_hat_pred = v_hat + a_lead*dt
            if self._kf_vhat.get(vid) is None:
                self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
            prev_v = self._prev_lead_speed.get(vid)
            a_lead = (vehicle_speed - prev_v) / (obs_dt if obs_dt > 0 else 0.1) if prev_v is not None else 0.0
            self._prev_lead_speed[vid] = vehicle_speed

            diag = infer_revertkalman(
                self._kf_vhat[vid], self._kf_P[vid],
                belief=belief,
                kf_Q=self._kf_Q, kf_R=self._kf_R,
                revert_alpha=1.0,
                a_lead=a_lead,
                dt=obs_dt if obs_dt > 0 else 0.1,
                **common)
            self._kf_vhat[vid] = diag["kf_vhat"]
            self._kf_epsilon[vid] = diag["kf_epsilon"]
            self._kf_kappa[vid] = diag["kf_kappa"]
            self._kf_P[vid] = diag["kf_P"]
            return diag
        elif self.inference_type == "iidkalman":
            # A=0: v_hat_pred = v_lead + a_lead*dt (forgets previous estimate)
            if self._kf_vhat.get(vid) is None:
                self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
            prev_v = self._prev_lead_speed.get(vid)
            a_lead = (vehicle_speed - prev_v) / (obs_dt if obs_dt > 0 else 0.1) if prev_v is not None else 0.0
            self._prev_lead_speed[vid] = vehicle_speed

            diag = infer_revertkalman(
                self._kf_vhat[vid], self._kf_P[vid],
                belief=belief,
                kf_Q=self._kf_Q, kf_R=self._kf_R,
                revert_alpha=0.0,
                a_lead=a_lead,
                dt=obs_dt if obs_dt > 0 else 0.1,
                **common)
            self._kf_vhat[vid] = diag["kf_vhat"]
            self._kf_epsilon[vid] = diag["kf_epsilon"]
            self._kf_kappa[vid] = diag["kf_kappa"]
            self._kf_P[vid] = diag["kf_P"]
            return diag
        elif self.inference_type == "revertkalman":
            # Lazy init vhat on first observation
            if self._kf_vhat.get(vid) is None:
                self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
            # Estimate lead acceleration
            prev_v = self._prev_lead_speed.get(vid)
            a_lead = (vehicle_speed - prev_v) / (obs_dt if obs_dt > 0 else 0.1) if prev_v is not None else 0.0
            self._prev_lead_speed[vid] = vehicle_speed

            diag = infer_revertkalman(
                self._kf_vhat[vid], self._kf_P[vid],
                belief=belief,
                kf_Q=self._kf_Q, kf_R=self._kf_R,
                revert_alpha=self._kf_revert_alpha,
                a_lead=a_lead, dt=obs_dt if obs_dt > 0 else 0.1,
                **common)
            self._kf_vhat[vid] = diag["kf_vhat"]
            self._kf_epsilon[vid] = diag["kf_epsilon"]
            self._kf_kappa[vid] = diag["kf_kappa"]
            self._kf_P[vid] = diag["kf_P"]
            return diag
        elif self.inference_type == "lingapkalman":
            if self._kf_vhat.get(vid) is None:
                self._kf_vhat[vid] = vehicle_speed + self._kf_epsilon_init
            prev_v = self._prev_lead_speed.get(vid)
            a_lead = (vehicle_speed - prev_v) / (obs_dt if obs_dt > 0 else 0.1) if prev_v is not None else 0.0
            self._prev_lead_speed[vid] = vehicle_speed

            diag = infer_lingapkalman(
                self._kf_vhat[vid], self._kf_P[vid],
                belief=belief,
                kf_Q=self._kf_Q, kf_R=self._kf_R,
                revert_alpha=self._kf_revert_alpha,
                lingap_beta=self._kf_lingap_beta,
                a_lead=a_lead, dt=obs_dt if obs_dt > 0 else 0.1,
                **common)
            self._kf_vhat[vid] = diag["kf_vhat"]
            self._kf_epsilon[vid] = diag["kf_epsilon"]
            self._kf_kappa[vid] = diag["kf_kappa"]
            self._kf_P[vid] = diag["kf_P"]
            return diag
        elif self.inference_type == "oracle":
            true_ve = self._true_vel_errors.get(vid, 0.0)
            diag = infer_oracle(belief, true_ve)
            # Set Kalman state so cbf_contmean can use it
            self._kf_kappa[vid] = true_ve
            self._kf_P[vid] = 1e-8
            diag["kf_kappa"] = true_ve
            diag["kf_P"] = 1e-8
            return diag
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

        # --- Human belief evolution (same-lane vehicles only) ---
        if self.human_type == "rbf":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if abs(lateral) < LANE_WIDTH * 0.8:
                    old_kappa = self._true_vel_errors.get(vid, 0.0)
                    new_kappa = evolve_vel_err(
                        old_kappa, along, self.b_kappa, self.sigma_kappa)
                    self._true_vel_errors[vid] = new_kappa
        elif self.human_type == "gaussian_mean":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if abs(lateral) < LANE_WIDTH * 0.8 and vs.velocity > 0.1:
                    noise = np.random.normal(
                        self._human_mu, self._human_sigma)
                    self._true_vel_errors[vid] = noise / vs.velocity
                else:
                    self._true_vel_errors[vid] = 0.0
        elif self.human_type == "gaussian":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if abs(lateral) < LANE_WIDTH * 0.8 and vs.velocity > 0.1:
                    noise = np.random.normal(0.0, self._human_sigma)
                    self._true_vel_errors[vid] = noise / vs.velocity
                else:
                    self._true_vel_errors[vid] = 0.0
        elif self.human_type == "walk":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if vid not in self._walk_epsilon:
                    self._walk_epsilon[vid] = 0.0
                if abs(lateral) < LANE_WIDTH * 0.8 and vs.velocity > 0.1:
                    self._walk_epsilon[vid] += np.random.normal(
                        0.0, np.sqrt(self._human_walk_Q))
                    self._true_vel_errors[vid] = (
                        self._walk_epsilon[vid] / vs.velocity)
                else:
                    self._true_vel_errors[vid] = 0.0
        elif self.human_type == "revert":
            # AR(1): eps_k = alpha * eps_{k-1} + N(0, sqrt(Q))
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if vid not in self._walk_epsilon:
                    self._walk_epsilon[vid] = 0.0
                if abs(lateral) < LANE_WIDTH * 0.8 and vs.velocity > 0.1:
                    self._walk_epsilon[vid] = (
                        self._human_revert_alpha * self._walk_epsilon[vid]
                        + np.random.normal(0.0, np.sqrt(self._human_walk_Q)))
                    self._true_vel_errors[vid] = (
                        self._walk_epsilon[vid] / vs.velocity)
                else:
                    self._true_vel_errors[vid] = 0.0
        elif self.human_type == "lingap":
            # AR(1) + gap-dependent drift:
            # eps_k = alpha * eps_{k-1} + beta * gap + N(0, sqrt(Q))
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                if vid not in self._walk_epsilon:
                    self._walk_epsilon[vid] = 0.0
                if abs(lateral) < LANE_WIDTH * 0.8 and vs.velocity > 0.1:
                    gap = max(along, 0.0)
                    self._walk_epsilon[vid] = (
                        self._human_revert_alpha * self._walk_epsilon[vid]
                        + self._human_lingap_beta * gap
                        + np.random.normal(0.0, np.sqrt(self._human_walk_Q)))
                    self._true_vel_errors[vid] = (
                        self._walk_epsilon[vid] / vs.velocity)
                else:
                    self._true_vel_errors[vid] = 0.0

        # --- Human controller (reacts to lead vehicle only) ---
        if lead_state is None:
            desired_speed = self.desired_speed
            human_accel = float(np.clip(
                K_SPEED * (desired_speed - ego.velocity),
                -MAX_ACCEL, MAX_ACCEL))
            perceived_lead_speed = None
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

        # --- Observed action (human action + observation noise) ---
        if self.action_noise_std > 0:
            observed_accel = human_accel + np.random.normal(
                0.0, self.action_noise_std)
        else:
            observed_accel = human_accel
        self.last_step_info["observed_accel"] = observed_accel

        # --- Inference for ALL vehicles ahead ---
        # Lead vehicle: full observation model (action is informative)
        # Non-lead: prediction only (action carries no info about this vehicle)
        per_vehicle_diag = {}
        if self.inference_type != "none":
            for vid, (vs, along, lateral) in vehicles_ahead.items():
                diag = self._run_inference_for_vehicle(
                    vid, ego.velocity, vs.velocity, along, observed_accel,
                    is_lead=(vid == lead_id), obs_dt=dt)
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

        has_vehicles = len(vehicles_ahead) > 0
        if has_vehicles and self.intervention_type != "none":
            if lead_state is not None and lead_distance < float("inf"):
                a_cbf_max = cbf_max_accel(
                    lead_distance, ego.velocity, lead_state.velocity,
                    self.d_safe, self.gamma, dt)

            if self.intervention_type == "cbf_single":
                # Single-step CBF against in-lane vehicles only
                min_a_max = MAX_ACCEL
                for vid, (vs, along, lateral) in vehicles_ahead.items():
                    if _in_lane(lateral):
                        a_m = cbf_max_accel(along, ego.velocity, vs.velocity,
                                            self.d_safe, self.gamma, dt)
                        min_a_max = min(min_a_max, a_m)
                a_cbf_max = min_a_max
                a_max_clipped = float(np.clip(min_a_max, -MAX_ACCEL, MAX_ACCEL))
                if human_accel > a_max_clipped:
                    executed_accel = a_max_clipped
                    intervened = True

            elif self.intervention_type == "always_policy":
                if lead_state is not None:
                    executed_accel = compute_accel(
                        ego.velocity, lead_state.velocity, lead_distance,
                        0.0, self.target_distance)
                else:
                    executed_accel = float(np.clip(
                        K_SPEED * (self.desired_speed - ego.velocity),
                        -MAX_ACCEL, MAX_ACCEL))
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

                executed_accel, intervened, self._prev_sol = \
                    intervene_lookahead_multi(
                        human_accel, ego, v_infos,
                        d_safe=self.d_safe, gamma=self.gamma, dt=dt,
                        target_distance=self.target_distance,
                        desired_speed=self.desired_speed,
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
