"""Scenario configuration, agent creation, and spawn utilities."""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from shapely.geometry import Polygon

import igp2 as ip

logger = logging.getLogger(__name__)


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
        intervention_type = agent_config.get("intervention_type", "none")
        inference_type = agent_config.get("inference_type", "naive")
        relevance_method = agent_config.get("relevance_method", "dual")
        planning_mode = agent_config.get("planning_mode", "2d")
        ref_controls = agent_config.get("ref_controls", "opt")
        human_type = agent_config.get("human_type", "static")
        n_kappa_particles = agent_config.get("n_kappa_particles", 0)
        kappa_min = agent_config.get("kappa_min", 0.3)
        b_kappa = agent_config.get("b_kappa", 0.1)
        q_kappa = agent_config.get("q_kappa", 0.0)
        plot_mcts_tree = agent_config.get("plot_mcts_tree", False)
        return ip.BeliefAgent(**base, scenario_map=scenario_map,
                              plot_interval=plot_interval,
                              agent_beliefs=agent_beliefs,
                              human=human,
                              intervention_type=intervention_type,
                              inference_type=inference_type,
                              relevance_method=relevance_method,
                              planning_mode=planning_mode,
                              ref_controls=ref_controls,
                              human_type=human_type,
                              n_kappa_particles=n_kappa_particles,
                              kappa_min=kappa_min,
                              b_kappa=b_kappa,
                              q_kappa=q_kappa,
                              plot_mcts_tree=plot_mcts_tree)
    elif agent_type == "TrafficAgent":
        open_loop = agent_config.get("open_loop", False)
        return ip.TrafficAgent(**base, open_loop=open_loop)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


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
    # Pass through all ego-level keys that create_agent reads
    for key in ("human", "intervention_type", "inference_type",
                "relevance_method", "planning_mode", "ref_controls",
                "human_type", "n_kappa_particles", "kappa_min",
                "b_kappa", "q_kappa", "plot_mcts_tree"):
        if key in ego_cfg:
            ego_agent[key] = ego_cfg[key]

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

    logger.info("Showing spawn preview. Close the plot window to continue...")
    plt.show(block=True)
