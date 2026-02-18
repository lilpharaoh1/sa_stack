"""
BeliefAgent experiment runner.

Loads a scenario config, runs the BeliefAgent in CARLA, collects per-step
diagnostics, and saves the results to a pickle file.

Usage:
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic -o my_run --seed 42
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic --steps 300 --no-plot
"""

import sys
import os
import logging
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import dill
import carla
import numpy as np
from shapely.geometry import Polygon

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Diagnostics captured at a single simulation step."""
    step: int
    wall_time: float  # seconds since experiment start

    # Ego state
    ego_position: Optional[np.ndarray]
    ego_speed: Optional[float]
    ego_heading: Optional[float]

    # Human (belief) policy outputs
    human_rollout: Optional[np.ndarray]        # (H+1, 4) [x, y, heading, speed]
    human_milp_rollout: Optional[np.ndarray]   # (H+1, 2) [x, y]
    human_nlp_converged: Optional[bool]
    human_obstacles: Optional[List]
    human_other_agents: Optional[Dict]
    human_trajectories: Dict[int, np.ndarray]

    # True (ground-truth) policy outputs
    true_rollout: Optional[np.ndarray]
    true_milp_rollout: Optional[np.ndarray]
    true_nlp_converged: Optional[bool]
    true_obstacles: Optional[List]
    true_other_agents: Optional[Dict]
    true_trajectories: Dict[int, np.ndarray]

    # Scene snapshot
    dynamic_agents: Dict[int, Any]   # non-ego, ID >= 0
    static_obstacles: Dict[int, Any] # ID < 0

    # Completion (based on true policy)
    goal_reached: bool

    # True policy constraint diagnostics (from TwoStageOPT._analyse_constraints)
    #   - nlp_ok: whether the NLP solver converged
    #   - velocity_violated: speed outside [v_min, v_max]
    #   - acceleration_violated: acceleration outside [a_min, a_max]
    #   - steering_violated: steering angle exceeds delta_max
    #   - jerk_violated: jerk exceeds jerk_max
    #   - steer_rate_violated: steering rate exceeds delta_rate_max
    #   - road_boundary_violations: number of corner-timestep road violations
    #   - collision_violations: number of corner-timestep collision violations
    true_diag_nlp_ok: Optional[bool] = None
    true_diag_velocity_violated: Optional[bool] = None
    true_diag_acceleration_violated: Optional[bool] = None
    true_diag_steering_violated: Optional[bool] = None
    true_diag_jerk_violated: Optional[bool] = None
    true_diag_steer_rate_violated: Optional[bool] = None
    true_diag_road_violations: int = 0
    true_diag_collision_violations: int = 0

    # Per-step ego timing breakdown (seconds).
    # Keys match BeliefAgent.last_step_timing, e.g.:
    #   human_predict, true_predict, human_policy, true_policy, plotting
    ego_timing: Optional[Dict[str, float]] = None

    # Prediction error: mean L2 distance (metres) between the 1-step-ahead
    # predicted positions from the *previous* step and the actual positions
    # observed at *this* step, averaged over all predicted agents.
    # NOTE: Currently uses the TRUE (ground-truth) predicted trajectories.
    # When belief inference is implemented, a separate belief_prediction_error
    # field should be added for the human-policy predictions.
    prediction_error: Optional[float] = None


@dataclass
class ExperimentResult:
    """Full result of a single experiment run."""
    # Metadata
    scenario_name: str
    config: Dict[str, Any]
    seed: int
    fps: int
    max_steps: int
    start_time: str       # ISO timestamp

    # Outcome
    solved: bool = False
    solved_step: Optional[int] = None
    total_steps: int = 0
    wall_time_seconds: float = 0.0

    # Failure outcome (human policy optimisation failure)
    failed: bool = False
    failure_step: Optional[int] = None
    failure_reason: Optional[str] = None

    # Per-step data
    steps: List[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BeliefAgent experiment runner")
    parser.add_argument("--map", "-m", type=str, required=True,
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (without extension). "
                             "Defaults to '{map}_{seed}'.")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum number of simulation steps")
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator",
                        help="Path to CARLA installation")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the BeliefAgent plotter")
    return parser.parse_args()


def generate_random_frame(ego: int,
                          layout: ip.Map,
                          spawn_vel_ranges: List[Tuple[ip.Box, Tuple[float, float]]],
                          rng: np.random.RandomState = None,
                          ) -> Dict[int, ip.AgentState]:
    """Generate a new frame with randomised spawns and velocities.

    Args:
        ego: Starting agent ID (usually 0).
        layout: Parsed road map.
        spawn_vel_ranges: List of (spawn_box, (vel_min, vel_max)) per agent.
        rng: Random state for reproducibility.  Falls back to the global
             ``np.random`` if *None*.
    """
    if rng is None:
        rng = np.random.RandomState()

    ret = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego):
        poly = Polygon(spawn.boundary)
        best_lane = layout.best_lane_at(spawn.center, max_distance=500.0)

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * rng.random() + start_d
        spawn_position = np.array(best_lane.point_at(position_d))

        speed = (vel[1] - vel[0]) * rng.random() + vel[0]
        heading = best_lane.get_heading_at(position_d)
        ret[i] = ip.AgentState(time=0,
                               position=spawn_position,
                               velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
                               acceleration=np.array([0.0, 0.0]),
                               heading=heading)
    return ret


def create_agent(agent_config, frame, fps, scenario_map, plot_interval=True):
    """Create an agent from its config dict."""
    base = {
        "agent_id": agent_config["id"],
        "initial_state": frame[agent_config["id"]],
        "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])),
        "fps": fps,
    }

    agent_type = agent_config["type"]

    if agent_type == "BeliefAgent":
        agent_beliefs = agent_config.get("beliefs", None)
        human = agent_config.get("human", True)
        return ip.BeliefAgent(**base, scenario_map=scenario_map,
                              plot_interval=plot_interval,
                              agent_beliefs=agent_beliefs,
                              human=human)
    elif agent_type == "TrafficAgent":
        open_loop = agent_config.get("open_loop", False)
        return ip.TrafficAgent(**base, open_loop=open_loop)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def collect_step(step: int, t0: float, ego_agent, ego_goal, frame,
                 prev_true_trajectories: Optional[Dict[int, np.ndarray]] = None,
                 ) -> StepRecord:
    """Collect diagnostics for one simulation step.

    Args:
        step: Current step index.
        t0: Wall-clock start time of the experiment.
        ego_agent: The ego BeliefAgent instance.
        ego_goal: Ego goal (used for goal-reached check).
        frame: Current observation frame.
        prev_true_trajectories: True-policy predicted trajectories from the
            *previous* step.  Used to compute 1-step-ahead prediction error.
    """
    ego_id = ego_agent.agent_id
    ego_state = frame.get(ego_id) if frame else None

    human_policy = getattr(ego_agent, '_human_policy', None)
    true_policy = getattr(ego_agent, '_true_policy', None)

    # Human (belief) policy
    human_rollout = getattr(human_policy, 'last_rollout', None) if human_policy else None
    human_milp = getattr(human_policy, 'last_milp_rollout', None) if human_policy else None
    human_nlp_converged = None
    if human_policy is not None:
        human_nlp_converged = getattr(human_policy, '_prev_nlp_states', None) is not None
    human_obstacles = getattr(human_policy, 'last_obstacles', None) if human_policy else None
    human_other_agents = getattr(human_policy, 'last_other_agents', None) if human_policy else None
    human_trajectories = dict(ego_agent._human_agent_trajectories)

    # True (ground-truth) policy
    true_rollout = getattr(true_policy, 'last_rollout', None) if true_policy else None
    true_milp = getattr(true_policy, 'last_milp_rollout', None) if true_policy else None
    true_nlp_converged = None
    if true_policy is not None:
        true_nlp_converged = getattr(true_policy, '_prev_nlp_states', None) is not None
    true_obstacles = getattr(true_policy, 'last_obstacles', None) if true_policy else None
    true_other_agents = getattr(true_policy, 'last_other_agents', None) if true_policy else None
    true_trajectories = dict(ego_agent._true_agent_trajectories)

    dynamic_agents = {aid: s for aid, s in frame.items()
                      if aid != ego_id and aid >= 0} if frame else {}
    static_obstacles = {aid: s for aid, s in frame.items()
                        if aid < 0} if frame else {}

    # Goal reached: based on TRUE policy rollout
    goal_reached = False
    if ego_goal is not None and true_rollout is not None:
        for pt in true_rollout[:, :2]:
            if ego_goal.reached(pt):
                goal_reached = True
                break

    # True policy constraint diagnostics
    true_diag = getattr(true_policy, 'last_diagnostics', None) if true_policy else None

    # Ego timing breakdown
    ego_timing = dict(ego_agent.last_step_timing) if ego_agent.last_step_timing else None

    # 1-step-ahead prediction error: compare previous step's predicted
    # positions (index 1 of each trajectory) with actual positions now.
    # NOTE: uses TRUE trajectories. When belief inference is added, compute
    # a separate error from human (belief) trajectories.
    prediction_error = None
    if prev_true_trajectories and frame:
        errors = []
        for aid, pred_traj in prev_true_trajectories.items():
            actual_state = frame.get(aid)
            if actual_state is None or len(pred_traj) < 2:
                continue
            predicted_pos = pred_traj[1]  # 1-step-ahead prediction
            actual_pos = actual_state.position
            errors.append(float(np.linalg.norm(predicted_pos - actual_pos)))
        if errors:
            prediction_error = float(np.mean(errors))

    return StepRecord(
        step=step,
        wall_time=time.time() - t0,
        ego_position=np.array(ego_state.position) if ego_state else None,
        ego_speed=float(ego_state.speed) if ego_state else None,
        ego_heading=float(ego_state.heading) if ego_state else None,
        human_rollout=human_rollout,
        human_milp_rollout=human_milp,
        human_nlp_converged=human_nlp_converged,
        human_obstacles=human_obstacles,
        human_other_agents=human_other_agents,
        human_trajectories=human_trajectories,
        true_rollout=true_rollout,
        true_milp_rollout=true_milp,
        true_nlp_converged=true_nlp_converged,
        true_obstacles=true_obstacles,
        true_other_agents=true_other_agents,
        true_trajectories=true_trajectories,
        dynamic_agents=dynamic_agents,
        static_obstacles=static_obstacles,
        goal_reached=goal_reached,
        # Constraint diagnostics from true policy
        true_diag_nlp_ok=true_diag.get('nlp_ok') if true_diag else None,
        true_diag_velocity_violated=true_diag.get('velocity_violated') if true_diag else None,
        true_diag_acceleration_violated=true_diag.get('acceleration_violated') if true_diag else None,
        true_diag_steering_violated=true_diag.get('steering_violated') if true_diag else None,
        true_diag_jerk_violated=true_diag.get('jerk_violated') if true_diag else None,
        true_diag_steer_rate_violated=true_diag.get('steer_rate_violated') if true_diag else None,
        true_diag_road_violations=len(true_diag.get('road_boundary_violations', [])) if true_diag else 0,
        true_diag_collision_violations=len(true_diag.get('collision_violations', [])) if true_diag else 0,
        # Timing and prediction
        ego_timing=ego_timing,
        prediction_error=prediction_error,
    )


def dump_results(result: ExperimentResult, name: str):
    """Save experiment results to pickle."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, name + ".pkl")
    with open(filepath, 'wb') as f:
        dill.dump(result, f)
    logger.info("Results saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Spawn preview plot
# ---------------------------------------------------------------------------

def plot_spawn_preview(scenario_map: 'ip.Map',
                       config: dict,
                       frame: Dict[int, 'ip.AgentState'],
                       title: str = "Spawn Preview",
                       raw_config: dict = None):
    """Show the road layout, spawn group boxes, goal boxes, and sampled positions.

    Displays a blocking matplotlib figure.  Close the window to continue.

    Args:
        scenario_map: Parsed road map.
        config: The *expanded* (old-format) config with ``"agents"`` list
            and concrete ``"static_objects"`` list.
        frame: Sampled initial states keyed by agent ID.
        title: Figure window title.
        raw_config: Optional original config with ``dynamic_groups`` and
            ``static_groups``.  When provided the group spawn boxes are drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle as MplRect, FancyBboxPatch
    from matplotlib.lines import Line2D
    from igp2.opendrive.plot_map import plot_map
    from igp2.core.util import calculate_multiple_bboxes

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plot_map(scenario_map, ax=ax, markings=True)

    # Collect all positions to auto-fit the view
    all_x, all_y = [], []

    # Colours
    COLOUR_EGO = (0.2, 0.4, 0.9)         # blue
    COLOUR_DYNAMIC = (0.85, 0.75, 0.3)   # desaturated yellow
    COLOUR_STATIC = (0.9, 0.4, 0.1)      # orange

    # --- Draw dynamic group spawn boxes (from raw config) ---
    if raw_config is not None:
        for group in raw_config.get("dynamic_groups", []):
            sb = group["spawn"]["box"]
            box = ip.Box(np.array(sb["center"]), sb["length"], sb["width"],
                         sb.get("heading", 0.0))
            corners = np.array(box.boundary)
            poly = plt.Polygon(corners, closed=True,
                                facecolor=(*COLOUR_DYNAMIC, 0.15),
                                edgecolor=(*COLOUR_DYNAMIC, 0.6),
                                linewidth=1.5, linestyle='--', zorder=2)
            ax.add_patch(poly)
            all_x.extend(corners[:, 0])
            all_y.extend(corners[:, 1])

        # Draw static group spawn boxes
        for group in raw_config.get("static_groups", []):
            sb = group["spawn"]["box"]
            box = ip.Box(np.array(sb["center"]), sb["length"], sb["width"],
                         sb.get("heading", 0.0))
            corners = np.array(box.boundary)
            poly = plt.Polygon(corners, closed=True,
                                facecolor=(*COLOUR_STATIC, 0.15),
                                edgecolor=(*COLOUR_STATIC, 0.6),
                                linewidth=1.5, linestyle='--', zorder=2)
            ax.add_patch(poly)
            all_x.extend(corners[:, 0])
            all_y.extend(corners[:, 1])

    # --- Draw ego spawn box and goal box ---
    ego_id = config["agents"][0]["id"]
    ego_cfg = config["agents"][0]
    sb = ego_cfg["spawn"]["box"]
    box = ip.Box(np.array(sb["center"]), sb["length"], sb["width"],
                 sb.get("heading", 0.0))
    corners = np.array(box.boundary)
    poly = plt.Polygon(corners, closed=True,
                        facecolor=(*COLOUR_EGO, 0.15),
                        edgecolor=(*COLOUR_EGO, 0.6),
                        linewidth=1.5, linestyle='--', zorder=2)
    ax.add_patch(poly)
    all_x.extend(corners[:, 0])
    all_y.extend(corners[:, 1])

    gb = ego_cfg["goal"]["box"]
    gbox = ip.Box(np.array(gb["center"]), gb["length"], gb["width"],
                  gb.get("heading", 0.0))
    gcorners = np.array(gbox.boundary)
    goal_poly = plt.Polygon(gcorners, closed=True,
                            facecolor=(*COLOUR_EGO, 0.10),
                            edgecolor=(*COLOUR_EGO, 0.5),
                            linewidth=1.5, linestyle=':', zorder=2)
    ax.add_patch(goal_poly)
    ax.annotate("Goal ego", xy=gb["center"], fontsize=7,
                ha='center', va='center', color=COLOUR_EGO, alpha=0.7, zorder=3)
    all_x.extend(gcorners[:, 0])
    all_y.extend(gcorners[:, 1])

    # --- Draw goal boxes for traffic agents ---
    for agent_cfg in config["agents"][1:]:
        gb = agent_cfg["goal"]["box"]
        gbox = ip.Box(np.array(gb["center"]), gb["length"], gb["width"],
                      gb.get("heading", 0.0))
        gcorners = np.array(gbox.boundary)
        goal_poly = plt.Polygon(gcorners, closed=True,
                                facecolor=(*COLOUR_DYNAMIC, 0.10),
                                edgecolor=(*COLOUR_DYNAMIC, 0.5),
                                linewidth=1.5, linestyle=':', zorder=2)
        ax.add_patch(goal_poly)
        aid = agent_cfg["id"]
        ax.annotate(f"Goal {aid}", xy=gb["center"], fontsize=7,
                    ha='center', va='center', color=COLOUR_DYNAMIC, alpha=0.7, zorder=3)
        all_x.extend(gcorners[:, 0])
        all_y.extend(gcorners[:, 1])

    # --- Draw sampled vehicle positions ---
    for aid, state in frame.items():
        is_ego = (aid == ego_id)
        colour = COLOUR_EGO if is_ego else COLOUR_DYNAMIC
        pos = state.position
        heading = state.heading

        vl, vw = 4.5, 1.8
        corners = calculate_multiple_bboxes(
            [pos[0]], [pos[1]], vl, vw, heading)[0]
        veh_poly = plt.Polygon(corners, closed=True,
                               facecolor=(*colour, 0.6),
                               edgecolor=(*colour, 1.0),
                               linewidth=2.0, zorder=5)
        ax.add_patch(veh_poly)

        arrow_len = 3.0
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=2), zorder=6)

        label = f"Ego (id={aid})" if is_ego else f"id={aid}"
        ax.annotate(label, xy=(pos[0], pos[1] + 2.5), fontsize=8, fontweight='bold',
                    ha='center', color=colour, zorder=7)

        ax.annotate(f"{state.speed:.1f} m/s", xy=(pos[0], pos[1] - 2.5), fontsize=7,
                    ha='center', color=colour, alpha=0.8, zorder=7)

        all_x.append(pos[0])
        all_y.append(pos[1])

    # --- Draw sampled static objects ---
    static_objs = config.get("static_objects", [])
    for obj in static_objs:
        pos = obj["position"][:2]
        heading = obj.get("heading", 0.0)
        ol = obj.get("length", 4.5)
        ow = obj.get("width", 1.8)

        corners = calculate_multiple_bboxes(
            [pos[0]], [pos[1]], ol, ow, heading)[0]
        obj_poly = plt.Polygon(corners, closed=True,
                               facecolor=(*COLOUR_STATIC, 0.7),
                               edgecolor=(*COLOUR_STATIC, 1.0),
                               linewidth=2.0, zorder=5)
        ax.add_patch(obj_poly)

        bp = obj.get("blueprint", "static")
        short_bp = bp.split(".")[-1] if "." in bp else bp
        ax.annotate(short_bp, xy=(pos[0], pos[1] + 2.5), fontsize=7,
                    ha='center', color=COLOUR_STATIC, zorder=7)

        all_x.append(pos[0])
        all_y.append(pos[1])

    # --- Auto-fit with margin ---
    if all_x and all_y:
        pad = 20.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # --- Legend ---
    legend_handles = [
        Line2D([0], [0], color=COLOUR_EGO, linewidth=6, alpha=0.6, label='Ego vehicle'),
        Line2D([0], [0], color=COLOUR_DYNAMIC, linewidth=6, alpha=0.6, label='Traffic vehicle'),
        Line2D([0], [0], color=COLOUR_STATIC, linewidth=6, alpha=0.7, label='Static object'),
        plt.Polygon([[0, 0]], closed=True,
                     facecolor=(0.5, 0.5, 0.5, 0.15), edgecolor=(0.5, 0.5, 0.5, 0.5),
                     linestyle='--', label='Spawn region'),
        plt.Polygon([[0, 0]], closed=True,
                     facecolor=(0.5, 0.5, 0.5, 0.08), edgecolor=(0.5, 0.5, 0.5, 0.4),
                     linestyle=':', label='Goal region'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    n_agents = len(config["agents"])
    n_static = len(static_objs)
    ax.set_title(f"{title}  ({n_agents} agents, {n_static} static)", fontsize=12)
    ax.set_aspect('equal')
    fig.tight_layout()

    print("Showing spawn preview. Close the plot window to continue...")
    plt.show(block=True)


# ---------------------------------------------------------------------------
# New-format config helpers
# ---------------------------------------------------------------------------

def is_new_format(config: dict) -> bool:
    """Check whether a config uses the new dynamic_groups format."""
    return "ego" in config and "dynamic_groups" in config


def expand_static_groups(config: dict,
                         rng: np.random.RandomState = None,
                         ) -> list:
    """Sample concrete static objects from ``static_groups``.

    Each group specifies a spawn box, a count range ``[min, max]``, and shared
    properties (blueprint, z_offset, length, width, heading).  For each sampled
    object the position is chosen uniformly within the spawn box.

    Any literal entries in ``"static_objects"`` are included unchanged.

    Returns:
        List of concrete static-object dicts ready for
        ``CarlaSim.spawn_static_objects_from_config``.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Start with any explicitly listed static objects
    result = list(config.get("static_objects", []))

    for group in config.get("static_groups", []):
        lo, hi = group["count"]
        count = int(rng.randint(lo, hi + 1))

        spawn_cfg = group["spawn"]["box"]
        center = np.array(spawn_cfg["center"])
        half_len = spawn_cfg["length"] / 2.0
        half_wid = spawn_cfg["width"] / 2.0
        box_heading = spawn_cfg.get("heading", 0.0)
        cos_h = np.cos(box_heading)
        sin_h = np.sin(box_heading)

        heading = group.get("heading", box_heading)

        for _ in range(count):
            # Sample uniformly in the box's local frame then rotate
            local_x = (rng.random() * 2 - 1) * half_len
            local_y = (rng.random() * 2 - 1) * half_wid
            x = center[0] + local_x * cos_h - local_y * sin_h
            y = center[1] + local_x * sin_h + local_y * cos_h

            obj = {
                "position": [float(x), float(y)],
                "heading": float(heading),
            }
            for key in ("blueprint", "z_offset", "length", "width"):
                if key in group:
                    obj[key] = group[key]
            result.append(obj)

    return result


def expand_new_config(config: dict,
                      layout: 'ip.Map',
                      rng: np.random.RandomState = None) -> dict:
    """Convert a new-format config (ego + dynamic_groups) to old-format (agents list).

    Samples vehicle counts from each group's [min, max] range, assigns sequential
    IDs starting from 1, and builds the ego's ``beliefs`` dict from per-group
    belief entries.  Also expands ``static_groups`` into concrete
    ``static_objects``.

    Returns:
        Old-format config dict with an ``"agents"`` list.
    """
    if rng is None:
        rng = np.random.RandomState()

    ego_cfg = config["ego"]
    groups = config["dynamic_groups"]

    # Ego is always agent 0
    ego_agent = {
        "id": 0,
        "type": ego_cfg.get("type", "BeliefAgent"),
        "spawn": ego_cfg["spawn"],
        "goal": ego_cfg["goal"],
    }
    if "human" in ego_cfg:
        ego_agent["human"] = ego_cfg["human"]

    agents = [ego_agent]
    beliefs = {}
    next_id = 1

    for group in groups:
        lo, hi = group["count"]
        count = int(rng.randint(lo, hi + 1))
        for _ in range(count):
            agent = {
                "id": next_id,
                "type": group.get("agent_type", "TrafficAgent"),
                "open_loop": group.get("open_loop", False),
                "spawn": group["spawn"],
                "goal": group["goal"],
            }
            agents.append(agent)
            if "belief" in group:
                beliefs[str(next_id)] = group["belief"]
            next_id += 1

    ego_agent["beliefs"] = beliefs

    static_objects = expand_static_groups(config, rng=rng)

    expanded = {
        "scenario": config["scenario"],
        "agents": agents,
        "static_objects": static_objects,
    }
    return expanded


def check_viability(frame: Dict[int, 'ip.AgentState'],
                    static_objects: list = None,
                    min_dist: float = 10.0) -> bool:
    """Return True if no objects are too close to each other.

    Checks all pairwise distances between dynamic agents, between dynamic
    agents and static objects, and between static objects themselves.
    """
    # Collect all positions
    positions = [state.position for state in frame.values()]
    if static_objects:
        for obj in static_objects:
            positions.append(np.array(obj["position"][:2]))

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                return False
    return True


def sample_viable_config(config: dict,
                         layout: 'ip.Map',
                         min_dist: float = 10.0,
                         max_attempts: int = 50,
                         seed: int = None,
                         ) -> Tuple[dict, Dict[int, 'ip.AgentState']]:
    """Repeatedly expand a new-format config and sample positions until viable.

    Returns:
        (expanded_config, frame) where frame passes the viability check.

    Raises:
        RuntimeError: after *max_attempts* without a viable sample.
    """
    rng = np.random.RandomState(seed)

    for attempt in range(max_attempts):
        expanded = expand_new_config(config, layout, rng=rng)

        ego_id = expanded["agents"][0]["id"]
        agent_spawns = []
        for ac in expanded["agents"]:
            spawn_box = ip.Box(
                np.array(ac["spawn"]["box"]["center"]),
                ac["spawn"]["box"]["length"],
                ac["spawn"]["box"]["width"],
                ac["spawn"]["box"]["heading"],
            )
            vel_range = ac["spawn"]["velocity"]
            agent_spawns.append((spawn_box, vel_range))

        frame = generate_random_frame(ego_id, layout, agent_spawns, rng=rng)

        if check_viability(frame, expanded.get("static_objects", []), min_dist):
            return expanded, frame

    raise RuntimeError(
        f"Could not find viable spawn configuration after {max_attempts} attempts"
    )


# ---------------------------------------------------------------------------
# Scene summary
# ---------------------------------------------------------------------------

def print_scene_summary(config: dict, frame: Dict[int, 'ip.AgentState']):
    """Print a compact summary of the scene configuration."""
    agents_cfg = config["agents"]
    ego_id = agents_cfg[0]["id"]
    ego_beliefs = agents_cfg[0].get("beliefs", {})

    print("  Agents")
    print("  " + "-" * 56)
    for ac in agents_cfg:
        aid = ac["id"]
        atype = ac["type"]
        is_ego = (aid == ego_id)
        state = frame.get(aid)

        # Position and speed from sampled frame
        if state is not None:
            pos = state.position
            spd = state.speed
            pos_str = f"({pos[0]:7.1f}, {pos[1]:7.1f})"
            spd_str = f"{spd:.1f} m/s"
        else:
            pos_str = "(?)"
            spd_str = "?"

        # Goal centre
        gc = ac["goal"]["box"]["center"]
        goal_str = f"({gc[0]:7.1f}, {gc[1]:7.1f})"

        # Belief about this agent (from ego's perspective)
        belief_str = ""
        if not is_ego:
            b = ego_beliefs.get(str(aid))
            if b is not None:
                vis = "visible" if b.get("visible", True) else "hidden"
                verr = b.get("velocity_error", 0.0)
                belief_str = f"  [{vis}"
                if verr != 0.0:
                    belief_str += f", vel_err={verr:+.1f}"
                belief_str += "]"

        role = "ego" if is_ego else "   "
        label = f"  {role} Agent {aid}"
        print(f"{label:<16} {atype:<14} pos={pos_str}  "
              f"v={spd_str:<8}  goal={goal_str}{belief_str}")

    # Static objects
    static_objs = config.get("static_objects", [])
    if static_objs:
        print(f"  Static objects: {len(static_objs)}")
        for obj in static_objs:
            pos = obj["position"]
            bp = obj.get("blueprint", "prop")
            # Shorten blueprint name
            short_bp = bp.split(".")[-1] if "." in bp else bp
            print(f"    {short_bp:<20} pos=({pos[0]:7.1f}, {pos[1]:7.1f})")
    print()


# ---------------------------------------------------------------------------
# Single experiment runner (extracted from main)
# ---------------------------------------------------------------------------

def run_single_experiment(config: dict,
                          frame: Dict[int, 'ip.AgentState'],
                          scenario_map: 'ip.Map',
                          carla_sim: 'ip.carlasim.CarlaSim',
                          max_steps: int,
                          fps: int,
                          plot_interval: bool = True,
                          seed: int = 21,
                          scenario_name: str = "experiment",
                          ) -> ExperimentResult:
    """Run a single experiment episode.

    Creates agents, steps the simulation, collects diagnostics, and returns
    an :class:`ExperimentResult`.  Does **not** clean up CARLA — the caller
    is responsible for removing agents and static objects afterwards.
    """
    ego_id = config["agents"][0]["id"]

    agents = {}
    for agent_config in config["agents"]:
        aid = agent_config["id"]
        agents[aid] = create_agent(agent_config, frame, fps, scenario_map,
                                   plot_interval=plot_interval)
        carla_sim.add_agent(agents[aid], "ego" if aid == ego_id else None)

    # Add static objects from config
    static_objs = config.get("static_objects", [])
    if static_objs:
        carla_sim.spawn_static_objects_from_config(static_objs)

    # Set up camera to follow the ego vehicle
    ego_wrapper = carla_sim.get_ego()
    if ego_wrapper is not None:
        camera_transform = carla.Transform(
            carla.Location(x=-10.0, z=6.0),
            carla.Rotation(pitch=-15.0),
        )
        carla_sim.attach_camera(ego_wrapper.actor, camera_transform)

    ego_agent = agents.get(ego_id)
    ego_goal = ego_agent.goal if ego_agent is not None else None

    # Tell the ego agent about the other agents
    if ego_agent is not None:
        ego_agent.set_agents(agents)

    # Prepare result object
    result = ExperimentResult(
        scenario_name=scenario_name,
        config=config,
        seed=seed,
        fps=fps,
        max_steps=max_steps,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print_scene_summary(config, frame)

    t0 = time.time()
    prev_true_trajectories = None

    for t in range(max_steps):
        t_step_start = time.perf_counter()
        obs, acts = carla_sim.step()
        carla_step_total = time.perf_counter() - t_step_start

        current_frame = obs.frame if obs is not None else None

        # Print per-step timing breakdown
        if ego_agent is not None and hasattr(ego_agent, 'last_step_timing'):
            st = ego_agent.last_step_timing
            if st:
                step_num = getattr(ego_agent, '_step_count', t)
                agent_total = sum(st.values())
                carla_overhead = carla_step_total - agent_total
                parts = "  ".join(f"{k}={v*1000:.1f}ms" for k, v in st.items())
                print(f"[Step {step_num:4d}] total={carla_step_total*1000:.0f}ms  "
                      f"carla_overhead={carla_overhead*1000:.0f}ms  {parts}")

        # Collect diagnostics
        if ego_agent is not None and current_frame is not None:
            record = collect_step(t, t0, ego_agent, ego_goal, current_frame,
                                  prev_true_trajectories=prev_true_trajectories)
            prev_true_trajectories = dict(ego_agent._true_agent_trajectories)
            result.steps.append(record)
            result.total_steps = t + 1

            if record.goal_reached:
                result.solved = True
                result.solved_step = t
                result.wall_time_seconds = time.time() - t0

                print(f"\n{'='*60}")
                print(f"  SCENARIO SOLVED at step {t}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Goal: {ego_goal}")
                print(f"  Wall time: {result.wall_time_seconds:.1f}s")
                print(f"{'='*60}\n")
                break

            # Stop if true policy optimisation failed
            if record.true_diag_nlp_ok is not None and not record.true_diag_nlp_ok:
                result.failed = True
                result.failure_step = t
                result.wall_time_seconds = time.time() - t0

                # Build failure reason — one entry per violated constraint type
                reasons = []
                if record.true_diag_collision_violations > 0:
                    reasons.append("collision avoidance infeasible")
                if record.true_diag_road_violations > 0:
                    reasons.append("road boundary infeasible")
                if record.true_diag_velocity_violated:
                    reasons.append("velocity bounds infeasible")
                if record.true_diag_acceleration_violated:
                    reasons.append("acceleration bounds infeasible")
                if record.true_diag_steering_violated:
                    reasons.append("steering bounds infeasible")
                if record.true_diag_jerk_violated:
                    reasons.append("jerk limits infeasible")
                if record.true_diag_steer_rate_violated:
                    reasons.append("steering rate infeasible")
                result.failure_reason = "; ".join(reasons) if reasons else "NLP infeasible (unknown cause)"

                print(f"\n{'='*60}")
                print(f"  TRUE POLICY FAILED at step {t}")
                print(f"  Reason: {result.failure_reason}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Wall time: {result.wall_time_seconds:.1f}s")
                print(f"{'='*60}\n")
                break
    else:
        result.wall_time_seconds = time.time() - t0
        print(f"\nScenario NOT solved within {max_steps} steps "
              f"({result.wall_time_seconds:.1f}s).")

    # Close any matplotlib figures opened by agent plotters
    import matplotlib.pyplot as plt
    plt.close('all')

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    ip.setup_logging(level=logging.INFO)
    np.random.seed(args.seed)
    np.seterr(divide="ignore")

    # Load scenario config
    config_path = os.path.join("scenarios", "configs", f"{args.map}.json")
    with open(config_path) as f:
        config = json.load(f)

    fps = config["scenario"].get("fps", 20)
    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10.0)

    scenario_xodr = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(scenario_xodr)
    map_name = config["scenario"].get("map_name", "Town01")

    rng = np.random.RandomState(args.seed)

    if is_new_format(config):
        # New format: expand dynamic groups and sample viable positions
        expanded, frame = sample_viable_config(
            config, scenario_map, seed=args.seed)
    else:
        # Old format: build spawn info and random initial frame
        expanded = config
        ego_id = config["agents"][0]["id"]
        agent_spawns = []
        for agent_config in config["agents"]:
            spawn_box = ip.Box(
                np.array(agent_config["spawn"]["box"]["center"]),
                agent_config["spawn"]["box"]["length"],
                agent_config["spawn"]["box"]["width"],
                agent_config["spawn"]["box"]["heading"],
            )
            vel_range = agent_config["spawn"]["velocity"]
            agent_spawns.append((spawn_box, vel_range))
        frame = generate_random_frame(ego_id, scenario_map, agent_spawns, rng=rng)

    plot_interval = False if args.no_plot else config["scenario"].get("plot_interval", True)

    # Show spawn preview before connecting to CARLA
    if plot_interval:
        plot_spawn_preview(scenario_map, expanded, frame,
                           title=f"Spawn Preview: {args.map}",
                           raw_config=config)

    # Create CARLA simulation
    carla_sim = ip.carlasim.CarlaSim(
        map_name=map_name,
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    result = run_single_experiment(
        config=expanded,
        frame=frame,
        scenario_map=scenario_map,
        carla_sim=carla_sim,
        max_steps=args.steps,
        fps=fps,
        plot_interval=plot_interval,
        seed=args.seed,
        scenario_name=args.map,
    )

    output_name = args.output if args.output else f"{args.map}_{args.seed}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.map}")
    print(f"  Seed: {args.seed}  |  FPS: {fps}  |  Max steps: {args.steps}")
    print(f"  Output: {output_name}.pkl")
    print(f"{'='*60}\n")

    # Save results
    filepath = dump_results(result, output_name)
    print(f"\nResults saved: {filepath}")
    print(f"  solved={result.solved}  failed={result.failed}  "
          f"steps={result.total_steps}  time={result.wall_time_seconds:.1f}s")
    if result.failure_reason:
        print(f"  failure_reason: {result.failure_reason}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
