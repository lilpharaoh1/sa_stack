"""
Car-following example: ego (CarFollowAgent) follows a lead (TrafficAgent).

The ego uses a basic proportional controller to maintain a target following
distance.  The ego has a velocity-error belief about the lead vehicle,
meaning it perceives the lead's speed as biased.  The assistive system
infers the velocity-error via a Boltzmann rationality model.

Run from the repo root:
    python scripts/debug/car_follow_example.py
    python scripts/debug/car_follow_example.py -m carfollow
    python scripts/debug/car_follow_example.py --carla_path /opt/carla-simulator
"""

import sys
import os
import logging
import argparse
import json
import time as _time
from datetime import datetime

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.agents.car_follow_agent import CarFollowAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live probability plotter
# ---------------------------------------------------------------------------

class BeliefPlotter:
    """Live bar chart showing the velocity-error posterior each step."""

    def __init__(self, candidates: np.ndarray, true_vel_err: float = None):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 3))
        self._candidates = candidates
        self._true_vel_err = true_vel_err

        self._bars = self._ax.bar(candidates, np.ones(len(candidates)) / len(candidates),
                                  width=0.08, color="steelblue", edgecolor="white")
        if true_vel_err is not None:
            self._ax.axvline(true_vel_err, color="red", linestyle="--",
                             linewidth=1.5, label=f"true = {true_vel_err:.1f}")
            self._ax.legend(fontsize=9)

        self._ax.set_xlabel("velocity error")
        self._ax.set_ylabel("P(vel_err)")
        self._ax.set_title("Inferred velocity-error belief")
        self._ax.set_xlim(candidates[0] - 0.1, candidates[-1] + 0.1)
        self._ax.set_ylim(0, 1.0)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, probabilities: np.ndarray, step: int = None):
        for bar, p in zip(self._bars, probabilities):
            bar.set_height(p)
        self._ax.set_ylim(0, 1.05)
        if step is not None:
            self._ax.set_title(f"Inferred velocity-error belief  (t={step})")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


class HumanBeliefPlotter:
    """Live time-series of the human's velocity error and Kalman estimate."""

    def __init__(self, initial_vel_err: float, window: int = 500):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 3))
        self._window = window
        self._steps = []
        self._vel_errs = []
        self._kf_kappas = []
        self._kf_stds = []

        self._line_true, = self._ax.plot(
            [], [], color="steelblue", linewidth=1.5, label="human $\\kappa$")
        self._line_kf, = self._ax.plot(
            [], [], color="orange", linewidth=1.5, label="KF estimate")
        self._fill = None
        self._ax.axhline(0, color="grey", linewidth=0.5, linestyle="--",
                         label="perfect perception")
        self._ax.axhline(initial_vel_err, color="red", linewidth=1,
                         linestyle="--", alpha=0.5,
                         label=f"initial = {initial_vel_err:.2f}")
        self._ax.set_xlabel("step")
        self._ax.set_ylabel("velocity error ($\\kappa$)")
        self._ax.set_title("Human velocity-error belief")
        self._ax.legend(fontsize=8, loc="upper right")
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, step: int, vel_err: float,
               kf_kappa: float = None, kf_P: float = None):
        self._steps.append(step)
        self._vel_errs.append(vel_err)
        self._kf_kappas.append(kf_kappa)
        self._kf_stds.append(np.sqrt(kf_P) if kf_P is not None else None)

        lo = max(0, len(self._steps) - self._window)
        s = self._steps[lo:]
        v = self._vel_errs[lo:]
        self._line_true.set_data(s, v)

        # Kalman estimate + uncertainty band
        kf_k = self._kf_kappas[lo:]
        kf_s = self._kf_stds[lo:]
        if any(k is not None for k in kf_k):
            s_kf = [si for si, k in zip(s, kf_k) if k is not None]
            v_kf = [k for k in kf_k if k is not None]
            std_kf = [sd for sd in kf_s if sd is not None]
            self._line_kf.set_data(s_kf, v_kf)
            if self._fill is not None:
                self._fill.remove()
            upper = [k + 2 * sd for k, sd in zip(v_kf, std_kf)]
            lower = [k - 2 * sd for k, sd in zip(v_kf, std_kf)]
            self._fill = self._ax.fill_between(
                s_kf, lower, upper, color="orange", alpha=0.15)
        else:
            self._line_kf.set_data([], [])

        self._ax.set_xlim(s[0], s[-1] + 1)
        all_v = v + [k for k in kf_k if k is not None]
        if all_v:
            ymin = min(all_v) - 0.1
            ymax = max(all_v) + 0.1
            self._ax.set_ylim(ymin, ymax)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


class ActionPlotter:
    """Live time-series with two subplots:
      - Top:    human accel vs executed accel
      - Bottom: safety constraint (a_max_safe)
    """

    def __init__(self, window: int = 200):
        plt.ion()
        self._fig, (self._ax_act, self._ax_safe) = plt.subplots(
            2, 1, figsize=(10, 5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
        self._window = window

        self._steps = []
        self._human_accel = []
        self._executed_accel = []
        self._a_max_safe = []

        # Top: actions
        self._line_human, = self._ax_act.plot(
            [], [], color="steelblue", linewidth=1.5, label="human accel")
        self._line_exec, = self._ax_act.plot(
            [], [], color="orange", linewidth=1.5, label="executed accel")
        self._ax_act.axhline(0, color="grey", linewidth=0.5)
        self._ax_act.set_ylabel("acceleration (m/s²)")
        self._ax_act.set_title("Actions & safety constraint")
        self._ax_act.legend(loc="upper right", fontsize=9)

        # Bottom: safety constraint
        self._line_safe, = self._ax_safe.plot(
            [], [], color="red", linewidth=1.5,
            label="$a_{\\mathrm{max}}^{\\mathrm{safe}}$")
        self._ax_safe.axhline(0, color="grey", linewidth=0.5)
        self._ax_safe.set_xlabel("step")
        self._ax_safe.set_ylabel("$a_{\\mathrm{max}}^{\\mathrm{safe}}$")
        self._ax_safe.legend(loc="upper right", fontsize=9)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, step: int, human_accel: float,
               executed_accel: float, a_max_safe: float = None):
        self._steps.append(step)
        self._human_accel.append(human_accel)
        self._executed_accel.append(executed_accel)
        self._a_max_safe.append(a_max_safe)

        # Sliding window
        lo = max(0, len(self._steps) - self._window)
        s = self._steps[lo:]

        # Top: actions
        self._line_human.set_data(s, self._human_accel[lo:])
        self._line_exec.set_data(s, self._executed_accel[lo:])

        act_vals = self._human_accel[lo:] + self._executed_accel[lo:]
        if act_vals:
            self._ax_act.set_ylim(min(act_vals) - 1, max(act_vals) + 1)

        # Bottom: safety constraint
        safe_vals = self._a_max_safe[lo:]
        if any(v is not None for v in safe_vals):
            safe_s = [si for si, v in zip(s, safe_vals) if v is not None]
            safe_v = [v for v in safe_vals if v is not None]
            self._line_safe.set_data(safe_s, safe_v)
            self._ax_safe.set_ylim(min(safe_v) - 1, max(safe_v) + 1)
        else:
            self._line_safe.set_data([], [])

        self._ax_safe.set_xlim(s[0], s[-1] + 1)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Episode recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    """Collects per-step statistics for post-hoc analysis."""

    def __init__(self, true_vel_errors: Dict[int, float],
                 belief_candidates: np.ndarray):
        self._true_vel_errors = true_vel_errors
        self._candidates = belief_candidates

        # Build oracle distribution (delta at nearest candidate)
        self._oracle_dist = {}
        for aid, ve in true_vel_errors.items():
            oracle = np.zeros(len(belief_candidates))
            oracle[np.argmin(np.abs(belief_candidates - ve))] = 1.0
            self._oracle_dist[aid] = oracle

        # Per-step storage
        self.steps: list = []
        self.human_accel: list = []
        self.executed_accel: list = []
        self.jerk: list = []
        self.distance: list = []
        self.ego_speed: list = []
        self.lead_speed: list = []
        self.intervened: list = []
        self.kl_divergence: list = []
        self.belief_dists: list = []
        self.human_vel_err: list = []  # human's velocity error over time
        self.kf_kappa: list = []      # Kalman estimate over time
        self.kf_P: list = []          # Kalman variance over time
        self.frames: list = []  # full frame snapshots for replay

    def record(self, step: int, info: dict, frame: dict):
        self.steps.append(step)
        self.human_accel.append(info.get("accel", 0.0))
        exec_a = info.get("executed_accel", info.get("accel", 0.0))
        self.executed_accel.append(exec_a)
        self.distance.append(info.get("distance"))
        self.ego_speed.append(info.get("ego_speed", 0.0))
        self.lead_speed.append(info.get("lead_speed", 0.0))
        self.intervened.append(info.get("intervened", False))

        # Jerk: d(executed_accel)/dt
        if len(self.executed_accel) >= 2:
            self.jerk.append(self.executed_accel[-1] - self.executed_accel[-2])
        else:
            self.jerk.append(0.0)

        # KL divergence: KL(oracle || inferred)
        inferred_dist = info.get("inferred_dist")
        if inferred_dist is not None:
            probs = np.array(list(inferred_dist.values()))
            self.belief_dists.append(probs.tolist())
            # Use first agent's oracle (car-following has one lead)
            oracle = list(self._oracle_dist.values())[0]
            # KL(oracle || inferred) = sum_k oracle[k] * log(oracle[k]/p[k])
            # With delta oracle this is -log(p[k_true])
            eps = 1e-12
            kl = float(np.sum(oracle * np.log((oracle + eps) / (probs + eps))))
            self.kl_divergence.append(kl)
        else:
            self.belief_dists.append(None)
            self.kl_divergence.append(None)

        # Human's current velocity error + Kalman state
        self.human_vel_err.append(info.get("human_vel_err"))
        self.kf_kappa.append(info.get("kf_kappa"))
        self.kf_P.append(info.get("kf_P"))

        # Frame snapshot (serialise AgentState to dict)
        frame_snap = {}
        for aid, state in frame.items():
            frame_snap[int(aid)] = {
                "position": state.position.tolist(),
                "velocity": state.velocity.tolist() if hasattr(state.velocity, 'tolist') else float(state.velocity),
                "heading": float(state.heading),
                "speed": float(state.speed),
            }
        self.frames.append(frame_snap)

    def save(self, path: str):
        """Save all recorded data to a JSON file."""
        data = {
            "steps": self.steps,
            "human_accel": self.human_accel,
            "executed_accel": self.executed_accel,
            "jerk": self.jerk,
            "distance": self.distance,
            "ego_speed": self.ego_speed,
            "lead_speed": self.lead_speed,
            "intervened": self.intervened,
            "kl_divergence": self.kl_divergence,
            "belief_dists": self.belief_dists,
            "belief_candidates": self._candidates.tolist(),
            "true_vel_errors": {str(k): v for k, v in self._true_vel_errors.items()},
            "human_vel_err": self.human_vel_err,
            "kf_kappa": self.kf_kappa,
            "kf_P": self.kf_P,
            "frames": self.frames,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Episode data saved to %s", path)

    def print_summary(self):
        n = len(self.steps)
        if n == 0:
            return
        accel_dev = np.array(self.human_accel) - np.array(self.executed_accel)
        jerk_arr = np.array(self.jerk)
        kl_vals = [v for v in self.kl_divergence if v is not None]
        intv_count = sum(self.intervened)

        print(f"\n{'='*60}")
        print(f"  Episode Summary  ({n} steps)")
        print(f"{'='*60}")
        print(f"  Accel deviation  |  mean={np.mean(accel_dev):.4f}  "
              f"std={np.std(accel_dev):.4f}  "
              f"max={np.max(np.abs(accel_dev)):.4f}")
        print(f"  Jerk             |  mean={np.mean(jerk_arr):.4f}  "
              f"std={np.std(jerk_arr):.4f}  "
              f"max={np.max(np.abs(jerk_arr)):.4f}")
        if kl_vals:
            print(f"  KL(oracle||inf)  |  mean={np.mean(kl_vals):.4f}  "
                  f"final={kl_vals[-1]:.4f}")
        print(f"  Interventions    |  {intv_count}/{n} steps "
              f"({100*intv_count/n:.1f}%)")
        print(f"{'='*60}\n")




# ---------------------------------------------------------------------------
# Config parsing  (handles the ego + dynamic_groups format directly)
# ---------------------------------------------------------------------------

def parse_config(config: dict, scenario_map: ip.Map, fps: int,
                 rng=None, inference: str = "none",
                 intervention: str = "none", gamma: float = 0.99,
                 human: str = "static"):
    """Parse a car-follow config into agents and initial frame.

    Returns:
        agents: dict mapping agent_id -> Agent
        frame:  dict mapping agent_id -> AgentState
    """
    if rng is None:
        rng = np.random.RandomState()

    ego_cfg = config["ego"]
    groups = config.get("dynamic_groups", [])

    # --- Build spawn list and agent configs ---
    agent_configs = []  # list of (id, cfg_dict)

    # Ego is always id 0
    agent_configs.append((0, ego_cfg))

    # Each dynamic group entry is one agent (no count sampling for this
    # simple example)
    next_id = 1
    beliefs = {}
    for group in groups:
        agent_cfg = {
            "type": group.get("type", "TrafficAgent"),
            "spawn": group["spawn"],
            "goal": group["goal"],
            "open_loop": group.get("open_loop", False),
        }
        agent_configs.append((next_id, agent_cfg))
        if "belief" in group:
            beliefs[str(next_id)] = group["belief"]
        next_id += 1

    # --- Sample initial frame ---
    spawn_vel_ranges = []
    for aid, cfg in agent_configs:
        sb = cfg["spawn"]["box"]
        spawn_box = ip.Box(
            np.array(sb["center"]),
            sb["length"], sb["width"], sb.get("heading", 0.0),
        )
        vel_range = cfg["spawn"]["velocity"]
        spawn_vel_ranges.append((spawn_box, vel_range))

    frame = _generate_frame(0, scenario_map, spawn_vel_ranges, rng)

    # --- Create agents ---
    agents = {}
    for aid, cfg in agent_configs:
        initial_state = frame[aid]
        goal = ip.BoxGoal(ip.Box(**cfg["goal"]["box"]))
        agent_type = cfg.get("type", "TrafficAgent")

        if agent_type == "CarFollowAgent":
            agents[aid] = CarFollowAgent(
                agent_id=aid,
                initial_state=initial_state,
                goal=goal,
                fps=fps,
                scenario_map=scenario_map,
                agent_beliefs=beliefs,
                target_distance=cfg.get("target_distance", 20.0),
                inference=inference,
                intervention=intervention,
                gamma=gamma,
                human=human,
            )
        elif agent_type == "TrafficAgent":
            agents[aid] = ip.TrafficAgent(
                agent_id=aid,
                initial_state=initial_state,
                goal=goal,
                fps=fps,
                open_loop=cfg.get("open_loop", False),
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    return agents, frame


def _generate_frame(ego_id, layout, spawn_vel_ranges, rng):
    """Sample initial positions along lanes inside spawn boxes."""
    from shapely.geometry import Polygon

    frame = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego_id):
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
        frame[i] = ip.AgentState(
            time=0,
            position=spawn_position,
            velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
            acceleration=np.array([0.0, 0.0]),
            heading=heading,
        )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Car-following belief-agent example")
    parser.add_argument("--map", "-m", type=str, default="carfollow",
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=500,
                        help="Fixed episode length in steps")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Override output directory (default: auto-generated)")
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator",
                        help="Path to CARLA installation")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--inference", type=str, default="none",
                        choices=["none", "boltzmann_reactive",
                                 "boltzmann_reactive_noprior",
                                 "boltzmann_reactive_floormix",
                                 "boltzmann_kalman", "oracle"],
                        help="Inference mode")
    parser.add_argument("--intervention", type=str, default="none",
                        choices=["none", "cbf_single", "cbf_mode",
                                 "cbf_wmean", "cbf_contmean",
                                 "cbf_chance", "always_policy"],
                        help="Intervention mode")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="CBF decay rate (1.0 = reach boundary in one step, "
                             "lower = more conservative)")
    parser.add_argument("--human", type=str, default="static",
                        choices=["static", "rbf"],
                        help="Human belief model: 'static' (fixed vel-error) "
                             "or 'rbf' (RBF-based drift toward 0)")
    parser.add_argument("--real-time", action="store_true",
                        help="Run at real-time fps with live plots. "
                             "Without this flag, runs as fast as possible "
                             "with no plots.")
    return parser.parse_args()


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
    max_steps = args.steps
    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10.0)

    scenario_xodr = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(scenario_xodr)

    rng = np.random.RandomState(args.seed)
    agents, frame = parse_config(config, scenario_map, fps, rng,
                                  inference=args.inference,
                                  intervention=args.intervention,
                                  gamma=args.gamma,
                                  human=args.human)

    ego_id = 0
    ego_agent = agents[ego_id]

    # Print scene summary
    print(f"\n{'='*60}")
    print(f"  Car Follow Example  (fps={fps}, episode_steps={max_steps})")
    print(f"  inference={args.inference}  intervention={args.intervention}")
    print(f"{'='*60}")
    for aid, state in frame.items():
        agent = agents[aid]
        role = "EGO" if aid == ego_id else "   "
        print(f"  {role} Agent {aid} ({type(agent).__name__})  "
              f"pos=({state.position[0]:.1f}, {state.position[1]:.1f})  "
              f"v={state.speed:.1f} m/s")
    if isinstance(ego_agent, CarFollowAgent):
        print(f"  Target distance: {ego_agent.target_distance:.1f} m")
        print(f"  Velocity-error beliefs: "
              f"{ego_agent.agent_beliefs}")
    print(f"{'='*60}\n")

    # --- CARLA setup ---
    import carla

    map_name = config["scenario"].get("map_name", "Town01")
    carla_sim = ip.carlasim.CarlaSim(
        map_name=map_name,
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    for aid, agent in agents.items():
        carla_sim.add_agent(agent, "ego" if aid == ego_id else None)

    # Spawn static objects if any
    static_objs = config.get("static_objects", [])
    if static_objs:
        carla_sim.spawn_static_objects_from_config(static_objs)

    # Camera following ego
    ego_wrapper = carla_sim.get_ego()
    if ego_wrapper is not None:
        camera_transform = carla.Transform(
            carla.Location(x=-10.0, z=6.0),
            carla.Rotation(pitch=-15.0),
        )
        carla_sim.attach_camera(ego_wrapper.actor, camera_transform)

    # Tell ego about other agents
    if hasattr(ego_agent, "set_agents"):
        ego_agent.set_agents(agents)

    # Set up episode recorder
    recorder = None
    if isinstance(ego_agent, CarFollowAgent):
        for aid, belief in ego_agent.inferred_belief.velocity_errors.items():
            recorder = EpisodeRecorder(
                ego_agent._human_vel_errors, belief.candidates)
            break

    # Set up live plotters (only in real-time mode)
    belief_plotter = None
    action_plotter = None
    human_belief_plotter = None
    if args.real_time and isinstance(ego_agent, CarFollowAgent):
        for aid, belief in ego_agent.inferred_belief.velocity_errors.items():
            true_ve = ego_agent._human_vel_errors.get(aid)
            belief_plotter = BeliefPlotter(belief.candidates, true_vel_err=true_ve)
            break

        action_plotter = ActionPlotter()

        for aid, ve in ego_agent._human_vel_errors.items():
            human_belief_plotter = HumanBeliefPlotter(initial_vel_err=ve)
            break

    dt = 1.0 / fps  # real-time step duration

    logger.info("Starting simulation (%d steps, ego=%d)", max_steps, ego_id)
    t0 = _time.time()

    # --- Main loop ---
    for t in range(max_steps):
        t0_step = _time.time()
        obs, acts = carla_sim.step()

        if obs is None:
            continue

        current_frame = obs.frame
        ego_state = current_frame.get(ego_id)

        # Print step diagnostics
        if isinstance(ego_agent, CarFollowAgent) and ego_agent.last_step_info:
            info = ego_agent.last_step_info
            dist_str = (f"{info['distance']:.1f}" if info.get("distance")
                        else "N/A")
            inferred_mode = info.get('inferred_mode')
            inferred_mean = info.get('inferred_mean')
            inf_str = (f"mode={inferred_mode:.2f} mean={inferred_mean:.2f}"
                       if inferred_mode is not None else "N/A")
            intv = "INTERVENED" if info.get('intervened') else ""
            a_safe_str = (f"a_max_safe={info['a_max_safe']:.2f}"
                          if info.get('a_max_safe') is not None else "")
            print(f"[t={t:4d}]  dist={dist_str:>6s}m  "
                  f"v_perceived={info.get('v_perceived', 0) or 0:.2f}  "
                  f"v_desired={info.get('v_desired', 0) or 0:.2f}  "
                  f"accel={info.get('accel', 0):.2f}  "
                  f"inferred=[{inf_str}]  "
                  f"{a_safe_str}  {intv}")
            inferred_dist = info.get('inferred_dist')
            if inferred_dist is not None:
                probs_str = "  ".join(f"{k:+.1f}:{v:.4f}"
                                      for k, v in inferred_dist.items())
                print(f"         P(vel_err): {probs_str}")

            # Update live plots
            if belief_plotter is not None and inferred_dist is not None:
                probs = np.array(list(inferred_dist.values()))
                belief_plotter.update(probs, step=t)

            if action_plotter is not None:
                action_plotter.update(
                    step=t,
                    human_accel=info.get('accel', 0),
                    executed_accel=info.get('executed_accel', info.get('accel', 0)),
                    a_max_safe=info.get('a_max_safe'),
                )

            if human_belief_plotter is not None:
                hve = info.get("human_vel_err")
                if hve is not None:
                    human_belief_plotter.update(
                        t, hve,
                        kf_kappa=info.get("kf_kappa"),
                        kf_P=info.get("kf_P"))

            # Record step data
            if recorder is not None:
                recorder.record(t, info, current_frame)

        # Real-time pacing (only in real-time mode)
        if args.real_time:
            elapsed_step = _time.time() - t0_step
            sleep_time = dt - elapsed_step
            if sleep_time > 0:
                _time.sleep(sleep_time)

    # --- End of episode ---
    elapsed = _time.time() - t0
    print(f"\nEpisode finished: {max_steps} steps ({elapsed:.1f}s wall time).")

    if recorder is not None:
        recorder.print_summary()

        # Always save results
        results_root = os.path.join(
            os.path.dirname(__file__), "car_follow_results")
        if args.output_dir:
            run_dir = args.output_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(
                results_root,
                f"human_{args.human}_inf_{args.inference}"
                f"_int_{args.intervention}"
                f"_seed{args.seed}_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        # Save episode data
        recorder.save(os.path.join(run_dir, "episode.json"))

        # Save run metadata
        meta = {
            "inference": args.inference,
            "intervention": args.intervention,
            "gamma": args.gamma,
            "human": args.human,
            "seed": args.seed,
            "steps": max_steps,
            "fps": fps,
            "map": args.map,
            "target_distance": ego_agent.target_distance if isinstance(ego_agent, CarFollowAgent) else None,
            "d_safe": ego_agent._d_safe if isinstance(ego_agent, CarFollowAgent) else None,
            "wall_time_s": round(elapsed, 2),
        }
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Results saved to: {run_dir}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
