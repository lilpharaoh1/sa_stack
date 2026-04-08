"""CBF safety filters for the car-following experiment.

Provides single-step and predictive (MPC) safety filters based on
discrete-time Control Barrier Functions.

Safe set:   C = { x : h(x) >= 0 }
Barrier:    h(x) = d - d_safe

Discrete-time CBF condition (per step):
    h(x_{t+1}) >= (1 - gamma) * h(x_t)
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from igp2.carfollow.human_model import (
    compute_accel, K_DIST, K_SPEED, MAX_ACCEL, rbf_feature,
)

logger = logging.getLogger(__name__)

LOOKAHEAD_N = 25


# ---------------------------------------------------------------------------
# Barrier function
# ---------------------------------------------------------------------------

def barrier(distance: float, d_safe: float) -> float:
    """h(x) = d - d_safe."""
    return distance - d_safe


def barrier_vec(distances: np.ndarray, d_safe: float) -> np.ndarray:
    return distances - d_safe


def cbf_max_accel(distance: float, ego_speed: float,
                  lead_speed: float, d_safe: float, gamma: float,
                  fps: int) -> float:
    """Two-step CBF maximum acceleration (forward Euler dynamics)."""
    dt = 1.0 / fps
    h = barrier(distance, d_safe)
    g = gamma
    return (2.0 * (lead_speed - ego_speed) * dt
            + g * (2.0 - g) * h) / (dt ** 2)


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate_forward(a_exec, lead_speed, s0, kappa, dt, N,
                     target_distance, ceiling=False):
    """Forward-simulate N steps with static kappa."""
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

        a = min(a_H, a_exec[j]) if ceiling else a_exec[j]

        v_next = max(0.0, v_j + a * dt)
        d_next = d_j + (lead_speed - v_j) * dt
        states[j + 1] = [d_next, v_next]

    return states, a_H_pred


def simulate_forward_evolving(a_exec, lead_speed, s0, kappa_init, dt, N,
                              target_distance, b_kappa=0.01,
                              sigma_kappa=25.0, ceiling=False):
    """Forward-simulate N steps with kappa evolving via RBF drift."""
    states = np.zeros((N + 1, 2))
    states[0] = s0
    a_H_pred = np.zeros(N)
    kappa_j = kappa_init

    for j in range(N):
        d_j, v_j = states[j]

        f = rbf_feature(d_j, sigma_kappa)
        kappa_j = kappa_j * (1.0 - b_kappa * f)

        perceived = lead_speed * (1.0 + kappa_j)
        desired = max(0.0, perceived + K_DIST * (d_j - target_distance))
        a_H = float(np.clip(K_SPEED * (desired - v_j),
                            -MAX_ACCEL, MAX_ACCEL))
        a_H_pred[j] = a_H

        a = min(a_H, a_exec[j]) if ceiling else a_exec[j]

        v_next = max(0.0, v_j + a * dt)
        d_next = d_j + (lead_speed - v_j) * dt
        states[j + 1] = [d_next, v_next]

    return states, a_H_pred


# ---------------------------------------------------------------------------
# Single-step CBF filter
# ---------------------------------------------------------------------------

def intervene_cbf_single(human_accel, distance, ego_speed, lead_speed,
                         d_safe, gamma, fps):
    """One-step CBF safety filter: clip to a_cbf_max."""
    a_max = cbf_max_accel(distance, ego_speed, lead_speed,
                          d_safe, gamma, fps)
    if human_accel > a_max:
        return float(np.clip(a_max, -MAX_ACCEL, MAX_ACCEL)), True, a_max
    return human_accel, False, a_max


# ---------------------------------------------------------------------------
# Predictive CBF safety filter (MPC)
# ---------------------------------------------------------------------------

def intervene_lookahead(human_accel, distance, ego_speed, lead_speed,
                        kappa_hat, d_safe, gamma, fps, target_distance,
                        prev_sol=None, robust_kappas=None,
                        evolving=False, b_kappa=0.01, sigma_kappa=25.0):
    """Predictive safety filter (CBF-MPC).

    Args:
        human_accel:    Human's intended acceleration.
        distance:       Current following distance.
        ego_speed:      Current ego speed.
        lead_speed:     Current lead speed.
        kappa_hat:      Estimated velocity error for objective.
        d_safe:         Minimum safe distance.
        gamma:          CBF decay rate.
        fps:            Frame rate.
        target_distance: Desired following distance.
        prev_sol:       Previous solution for warm-start (or None).
        robust_kappas:  If not None, list of kappas for robust constraints.
        evolving:       If True, use evolving kappa simulator.
        b_kappa:        RBF drift rate (for evolving).
        sigma_kappa:    RBF sigma (for evolving).

    Returns:
        (safe_accel, intervened, new_prev_sol)
    """
    dt = 1.0 / fps
    N = LOOKAHEAD_N
    s0 = np.array([distance, ego_speed])

    _sim_args = dict(lead_speed=lead_speed, dt=dt, N=N,
                     target_distance=target_distance)

    if evolving:
        def _sim(a_exec, kappa, ceiling=False):
            return simulate_forward_evolving(
                a_exec, ceiling=ceiling, s0=s0, kappa_init=kappa,
                b_kappa=b_kappa, sigma_kappa=sigma_kappa, **_sim_args)
    else:
        def _sim(a_exec, kappa, ceiling=False):
            return simulate_forward(
                a_exec, ceiling=ceiling, s0=s0, kappa=kappa, **_sim_args)

    # Objective: minimise deviation from human
    def objective(a_exec):
        _, a_H = _sim(a_exec, kappa_hat)
        return float(np.sum((a_exec - a_H) ** 2))

    # CBF + speed constraints
    constraints = []
    kappas_check = robust_kappas if robust_kappas is not None else [kappa_hat]
    use_ceiling = robust_kappas is not None

    for kappa_k in kappas_check:
        def _make_cbf_con(k, ceil, g=gamma):
            def con(a_exec):
                st, _ = _sim(a_exec, k, ceiling=ceil)
                h_j = barrier_vec(st[:-1, 0], d_safe)
                h_jp1 = barrier_vec(st[1:, 0], d_safe)
                return h_jp1 - (1.0 - g) * h_j
            return con

        def _make_v_con(k, ceil):
            def con(a_exec):
                st, _ = _sim(a_exec, k, ceiling=ceil)
                return st[1:, 1]
            return con

        constraints.append({"type": "ineq", "fun": _make_cbf_con(kappa_k, use_ceiling)})
        constraints.append({"type": "ineq", "fun": _make_v_con(kappa_k, use_ceiling)})

    # Warm-start
    if prev_sol is not None and len(prev_sol) == N:
        x0 = np.empty(N)
        x0[:-1] = prev_sol[1:]
        x0[-1] = human_accel
    else:
        x0 = np.full(N, human_accel)

    result = scipy_minimize(
        objective, x0,
        method="SLSQP",
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
                              d_safe, gamma, fps)
        safe_accel = float(np.clip(min(human_accel, a_max),
                                   -MAX_ACCEL, MAX_ACCEL))

    intervened = abs(safe_accel - human_accel) > 1e-3
    return safe_accel, intervened, new_sol
