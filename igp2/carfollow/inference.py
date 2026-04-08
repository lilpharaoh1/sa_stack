"""Inference methods for estimating the human's velocity-error belief."""

import logging
from typing import Dict, Optional

import numpy as np

from igp2.carfollow.beliefs import VelocityErrorBelief, CarFollowBeliefState
from igp2.carfollow.human_model import compute_accel, rbf_feature

logger = logging.getLogger(__name__)

FLOOR_MIX_EPSILON = 0.05


def infer_boltzmann_reactive(belief: VelocityErrorBelief,
                             ego_speed: float, lead_speed: float,
                             distance: float, human_accel: float,
                             target_distance: float, beta: float,
                             use_prior: bool = True,
                             floor_mix: bool = False) -> dict:
    """Boltzmann rationality update over discrete velocity-error candidates.

    Returns dict of diagnostics (inferred_mode, inferred_mean, inferred_dist).
    """
    likelihoods = np.empty(len(belief.candidates))
    for i, kappa in enumerate(belief.candidates):
        candidate_accel = compute_accel(
            ego_speed, lead_speed, distance, kappa, target_distance)
        likelihoods[i] = np.exp(-beta * (human_accel - candidate_accel) ** 2)

    if use_prior or floor_mix:
        prior = belief.probabilities
        if floor_mix:
            uniform = np.ones_like(prior) / len(prior)
            prior = (1.0 - FLOOR_MIX_EPSILON) * prior + FLOOR_MIX_EPSILON * uniform
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
        "inferred_dist": dict(
            zip(np.round(belief.candidates, 2),
                np.round(belief.probabilities, 4))),
    }


def infer_boltzmann_kalman(kf_kappa: float, kf_P: float,
                           ego_speed: float, lead_speed: float,
                           distance: float, human_accel: float,
                           target_distance: float,
                           belief: VelocityErrorBelief,
                           sigma_kappa: float = 25.0,
                           kf_B: float = 0.01,
                           kf_Q: float = 0.001,
                           kf_R: float = 0.0001) -> dict:
    """Kalman filter inference over velocity-error.

    Returns dict with kf_kappa, kf_P, and discrete belief diagnostics.
    """
    from igp2.carfollow.human_model import K_SPEED

    # Prediction
    f_kappa = rbf_feature(distance, sigma_kappa)
    x_pred = kf_kappa * (1.0 - kf_B * f_kappa)
    P_pred = kf_P + kf_Q

    # Observation (linearised: C = da_H/dkappa = K_SPEED * v_lead)
    C = K_SPEED * lead_speed
    a_pred = compute_accel(ego_speed, lead_speed, distance,
                           x_pred, target_distance)
    innovation = human_accel - a_pred

    S = C ** 2 * P_pred + kf_R
    K_gain = P_pred * C / S
    x_hat_new = x_pred + K_gain * innovation
    P_new = (1.0 - K_gain * C) * P_pred
    P_new = max(P_new, 1e-8)

    # Project Gaussian onto discrete candidates (log-sum-exp for stability)
    log_probs = -(belief.candidates - x_hat_new) ** 2 / (2.0 * P_new)
    log_probs -= log_probs.max()  # shift so max is 0 -> exp(0) = 1
    probs = np.exp(log_probs)
    belief.probabilities = probs / probs.sum()

    return {
        "kf_kappa": x_hat_new,
        "kf_P": P_new,
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_dist": dict(
            zip(np.round(belief.candidates, 2),
                np.round(belief.probabilities, 4))),
    }


def infer_oracle(belief: VelocityErrorBelief, true_vel_err: float) -> dict:
    """Set belief to delta at the nearest candidate to the true value."""
    dists = np.abs(belief.candidates - true_vel_err)
    belief.probabilities = np.zeros_like(belief.probabilities)
    belief.probabilities[np.argmin(dists)] = 1.0

    return {
        "inferred_mode": belief.mode,
        "inferred_mean": belief.mean,
        "inferred_dist": dict(
            zip(np.round(belief.candidates, 2),
                np.round(belief.probabilities, 4))),
    }
