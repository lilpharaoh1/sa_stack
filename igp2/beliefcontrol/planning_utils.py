"""Shared planning utilities used by TwoStagePolicy and BeliefInference.

Provides:
- milp_to_nlp_warmstart: Convert MILP states to NLP warm-start
- sample_road_boundaries: Query road boundaries at given s positions
- predict_obstacles_cv: Constant-velocity obstacle prediction
"""

import logging
from typing import Dict, Optional

import numpy as np

from igp2.core.agentstate import AgentState
from igp2.beliefcontrol.frenet import FrenetFrame

logger = logging.getLogger(__name__)


def milp_to_nlp_warmstart(milp_states: np.ndarray,
                          frenet_state: np.ndarray,
                          horizon: int,
                          dt: float,
                          wheelbase: float,
                          nlp_params: dict):
    """Convert MILP [s, d, vs, vd] to NLP [s, d, phi, v] + controls.

    Args:
        milp_states: (H+1, 4) MILP states [s, d, vs, vd].
        frenet_state: [s, d, phi, v] current Frenet state.
        horizon: Planning horizon.
        dt: Planning timestep.
        wheelbase: Vehicle wheelbase (m).
        nlp_params: Dict of NLP parameters (needs a_min, a_max, delta_max).

    Returns:
        (nlp_states, nlp_controls) tuple.
    """
    H = horizon
    L = wheelbase

    s_arr = milp_states[:, 0]
    d_arr = milp_states[:, 1]
    vs_arr = milp_states[:, 2]
    vd_arr = milp_states[:, 3]

    speeds = np.sqrt(vs_arr**2 + vd_arr**2)
    phis = np.arctan2(vd_arr, vs_arr)
    phis[0] = frenet_state[2]

    nlp_states = np.column_stack([s_arr, d_arr, phis, speeds])
    nlp_controls = np.zeros((H, 2))

    for k in range(H):
        a_k = np.clip((speeds[k + 1] - speeds[k]) / dt,
                      nlp_params['a_min'], nlp_params['a_max'])

        d_phi = (phis[k + 1] - phis[k] + np.pi) % (2 * np.pi) - np.pi
        spd = max(speeds[k], 0.5)
        sin_arg = np.clip(d_phi * L / (2.0 * spd * dt), -1.0, 1.0)
        delta_k = np.clip(np.arcsin(sin_arg),
                          -nlp_params['delta_max'], nlp_params['delta_max'])

        nlp_controls[k] = [a_k, delta_k]

    return nlp_states, nlp_controls


def sample_road_boundaries(s_values: np.ndarray,
                           scenario_map,
                           frenet: Optional[FrenetFrame]):
    """Sample road boundaries at given s positions.

    Args:
        s_values: (N,) array of arc-length positions.
        scenario_map: Road layout for boundary queries. May be None.
        frenet: FrenetFrame for the reference path. May be None.

    Returns:
        (road_left, road_right) arrays of lateral offsets.
    """
    if scenario_map is not None and frenet is not None:
        return frenet.road_boundaries(s_values, scenario_map,
                                      search_radius=15.0)
    return (np.full(len(s_values), 3.5),
            np.full(len(s_values), -3.5))


def predict_obstacles_cv(agent_states: Dict[int, AgentState],
                         visible_aids: set,
                         frenet: FrenetFrame,
                         horizon: int,
                         dt: float) -> list:
    """Constant-velocity obstacle prediction for visible agents only.

    Produces the same output format as TwoStagePolicy._predict_obstacles.

    Args:
        agent_states: Snapshot of agent states.
        visible_aids: Set of agent IDs to include (dynamic agents not in
            this set are excluded; static agents with negative IDs are
            always included).
        frenet: FrenetFrame for coordinate conversion.
        horizon: Planning horizon.
        dt: Planning timestep.

    Returns:
        List of obstacle dicts.
    """
    if frenet is None:
        return []

    H = horizon
    obstacles = []

    for aid, state in agent_states.items():
        # Static objects (negative IDs) are always included;
        # dynamic agents are included only if visible in this config
        if aid >= 0 and aid not in visible_aids:
            continue

        pos = np.array(state.position, dtype=float)
        vel = np.array(state.velocity, dtype=float)

        # Constant-velocity world positions
        world_positions = np.empty((H + 1, 2))
        for k in range(H + 1):
            world_positions[k] = pos + vel * k * dt

        # Headings from displacement
        headings = np.empty(H + 1)
        headings[0] = float(state.heading)
        for k in range(1, H + 1):
            dx = world_positions[k, 0] - world_positions[k - 1, 0]
            dy = world_positions[k, 1] - world_positions[k - 1, 1]
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                headings[k] = np.arctan2(dy, dx)
            else:
                headings[k] = headings[k - 1]

        # Convert to Frenet
        s_arr, d_arr = frenet.world_to_frenet_batch(world_positions)

        obs_meta = getattr(state, 'metadata', None)
        obs_length = obs_meta.length if obs_meta is not None else 4.5
        obs_width = obs_meta.width if obs_meta is not None else 1.8

        obstacles.append({
            's': s_arr,
            'd': d_arr,
            'world_positions': world_positions,
            'headings': headings,
            'length': obs_length,
            'width': obs_width,
            'heading': float(state.heading),
            'agent_id': aid,
            'uses_planned_trajectory': False,
        })

    return obstacles
