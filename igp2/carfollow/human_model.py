"""Human driver model: proportional controller + RBF belief evolution."""

import logging
from typing import Dict, Optional

import numpy as np

from igp2.core.agentstate import AgentState

logger = logging.getLogger(__name__)

# Controller gains
K_DIST = 0.3    # distance-error gain
K_SPEED = 1.0   # speed-error gain
MAX_ACCEL = 5.0  # acceleration clamp (m/s^2)


def compute_accel(ego_speed: float, lead_speed: float,
                  distance: float, vel_err: float,
                  target_distance: float) -> float:
    """Compute the acceleration the human controller would output.

    v_desired = v_lead * (1 + kappa) + k_d * (d - d*)
    a = k_v * (v_desired - v_ego)
    """
    perceived_lead_speed = lead_speed * (1.0 + vel_err)
    dist_error = distance - target_distance
    desired_speed = perceived_lead_speed + K_DIST * dist_error
    desired_speed = max(0.0, desired_speed)
    accel = K_SPEED * (desired_speed - ego_speed)
    return float(np.clip(accel, -MAX_ACCEL, MAX_ACCEL))


def rbf_feature(distance: float, sigma: float = 25.0) -> float:
    """RBF kernel feature on following distance."""
    return float(np.exp(-distance ** 2 / (2.0 * sigma ** 2)))


def evolve_vel_err(vel_err: float, distance: float,
                   b_kappa: float = 0.01, sigma: float = 25.0) -> float:
    """One step of deterministic RBF drift toward perfect perception.

    vel_err_new = vel_err * (1 - b_kappa * f_kappa)
    """
    f = rbf_feature(distance, sigma)
    return vel_err * (1.0 - b_kappa * f)


def get_lead_vehicle(ego_id: int, frame: Dict[int, AgentState],
                     other_agents: Dict[int, object]):
    """Find the closest agent ahead of ego along the heading direction.

    Returns:
        (agent_id, AgentState, along_distance) or (None, None, None).
    """
    my_state = frame.get(ego_id)
    if my_state is None:
        return None, None, None

    heading = my_state.heading
    fwd = np.array([np.cos(heading), np.sin(heading)])

    best_aid = None
    best_state = None
    best_dist = float("inf")

    for aid in other_agents:
        if aid not in frame:
            continue
        diff = frame[aid].position - my_state.position
        along = diff @ fwd
        if 0 < along < best_dist:
            best_aid = aid
            best_state = frame[aid]
            best_dist = along

    return best_aid, best_state, best_dist
