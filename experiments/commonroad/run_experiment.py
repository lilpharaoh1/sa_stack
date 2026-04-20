"""
Simulation runner for CommonRoad car-following / merging experiments.

All vehicles (ego and non-ego) are stepped procedurally each tick via
pluggable controllers and the bicycle kinematic model.  Non-ego vehicles
can optionally fall back to pre-baked XML trajectories.

Usage (from repo root, carla-igp2 env):

    # Live window — ego uses ACC, lead is constant-velocity controller
    python experiments/commonroad/run_experiment.py -s exp1_simple_acc

    # Save gif, headless
    python experiments/commonroad/run_experiment.py -s exp2_merge_in_front --save --headless

    # Constant-velocity ego (no ACC)
    python experiments/commonroad/run_experiment.py -s exp1_simple_acc -c constant_velocity
"""

import argparse
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

from commonroad.common.file_reader import CommonRoadFileReader

# ---------------------------------------------------------------------------
#  Vehicle parameters (matching igp2/core defaults)
# ---------------------------------------------------------------------------
WHEELBASE = 2.686
FRONT_OVERHANG = 0.91
REAR_OVERHANG = 1.094
MAX_ACCEL = 5.0       # m/s^2
MAX_ANGULAR_VEL = 2.0  # rad/s
VEH_LENGTH = 4.5
VEH_WIDTH = 1.8

CORRECTION = (REAR_OVERHANG - FRONT_OVERHANG) / 2.0
L_F = WHEELBASE / 2.0 + CORRECTION
L_R = WHEELBASE / 2.0 - CORRECTION

LANE_WIDTH = 3.5
N_LANES = 3

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OBSTACLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
EGO_COLOR = "#27ae60"


# ---------------------------------------------------------------------------
#  Shared types
# ---------------------------------------------------------------------------
@dataclass
class VehicleState:
    x: float
    y: float
    heading: float
    velocity: float
    acceleration: float = 0.0
    steering_angle: float = 0.0
    time_step: int = 0


@dataclass
class Action:
    acceleration: float = 0.0
    steer_angle: float = 0.0


@dataclass
class Observation:
    """What a controller sees each tick."""
    ego: VehicleState
    # vehicle_id → VehicleState  (all other vehicles visible to this one)
    others: Dict[int, VehicleState] = field(default_factory=dict)
    # vehicle_id → list of (x, y, heading, velocity) for future timesteps
    # Available when the simulation has access to pre-planned trajectories.
    future_trajectories: Dict[int, List[tuple]] = field(default_factory=dict)
    time_step: int = 0
    dt: float = 0.1


# ---------------------------------------------------------------------------
#  Bicycle kinematic model
# ---------------------------------------------------------------------------
def bicycle_step(state: VehicleState, action: Action, dt: float) -> VehicleState:
    """Advance one step (matches igp2/core/vehicle.py KinematicVehicle)."""
    accel = np.clip(action.acceleration, -MAX_ACCEL, MAX_ACCEL)
    velocity = max(0.0, state.velocity + accel * dt)

    beta = np.arctan(L_R * np.tan(action.steer_angle) / WHEELBASE)
    dx = velocity * np.cos(beta + state.heading) * dt
    dy = velocity * np.sin(beta + state.heading) * dt

    d_theta = velocity * np.tan(action.steer_angle) * np.cos(beta) / WHEELBASE
    d_theta = np.clip(d_theta, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
    heading = (state.heading + d_theta * dt + np.pi) % (2 * np.pi) - np.pi

    return VehicleState(
        x=state.x + dx, y=state.y + dy,
        heading=heading, velocity=velocity,
        acceleration=accel, steering_angle=action.steer_angle,
        time_step=state.time_step + 1,
    )


# ---------------------------------------------------------------------------
#  SimVehicle: wraps state + controller + history for any vehicle
# ---------------------------------------------------------------------------
class SimVehicle:
    """A vehicle in the simulation — ego or non-ego."""

    def __init__(self, vehicle_id: int, initial_state: VehicleState,
                 controller: Callable[["Observation"], Action],
                 is_ego: bool = False,
                 prebaked_traj: List[tuple] = None):
        """
        Args:
            vehicle_id:     Unique ID.
            initial_state:  Starting state.
            controller:     fn(Observation) → Action.  Called each tick.
            is_ego:         True for the ego vehicle.
            prebaked_traj:  Optional fallback trajectory from XML.
                            List of (x, y, heading, velocity) per timestep.
                            Only used if controller is None / "playback".
        """
        self.id = vehicle_id
        self.state = initial_state
        self.controller = controller
        self.is_ego = is_ego
        self.prebaked_traj = prebaked_traj
        self.history: List[VehicleState] = [initial_state]

    def step(self, obs: Observation, dt: float):
        """Advance this vehicle by one tick."""
        action = self.controller(obs)
        self.state = bicycle_step(self.state, action, dt)
        self.history.append(self.state)

    def as_tuple(self) -> tuple:
        """Current state as (x, y, heading, velocity)."""
        s = self.state
        return (s.x, s.y, s.heading, s.velocity)


# ---------------------------------------------------------------------------
#  Controllers
# ---------------------------------------------------------------------------

def constant_velocity_controller(obs: Observation) -> Action:
    """Maintain current velocity, no steering."""
    return Action(0.0, 0.0)


def simple_acc_controller(obs: Observation,
                          target_distance: float = 20.0,
                          desired_speed: float = 15.0,
                          k_dist: float = 0.3,
                          k_speed: float = 1.0) -> Action:
    """Proportional ACC (mirrors igp2/carfollow/human_model.py)."""
    ego = obs.ego
    fwd = np.array([np.cos(ego.heading), np.sin(ego.heading)])

    lead_dist = float("inf")
    lead_vel = None
    for vid, vs in obs.others.items():
        diff = np.array([vs.x - ego.x, vs.y - ego.y])
        along = diff @ fwd
        lateral = abs(diff[0] * (-fwd[1]) + diff[1] * fwd[0])
        if 0 < along < lead_dist and lateral < LANE_WIDTH * 0.8:
            lead_dist = along
            lead_vel = vs.velocity

    if lead_vel is not None:
        dist_error = lead_dist - target_distance
        v_desired = lead_vel + k_dist * dist_error
        v_desired = max(0.0, v_desired)
    else:
        v_desired = desired_speed

    accel = k_speed * (v_desired - ego.velocity)
    accel = np.clip(accel, -MAX_ACCEL, MAX_ACCEL)
    return Action(accel, 0.0)


def accelerating_controller_factory(accel: float = 1.0,
                                    v_max: float = 25.0):
    """Return a controller that applies constant acceleration up to v_max."""
    def controller(obs: Observation) -> Action:
        if obs.ego.velocity >= v_max:
            return Action(0.0, 0.0)
        return Action(accel, 0.0)
    return controller


def playback_controller_factory(prebaked_traj: List[tuple]):
    """Return a controller that replays a pre-baked trajectory.

    Falls back to constant-velocity when the trajectory runs out.
    """
    def controller(obs: Observation) -> Action:
        t = obs.time_step + 1  # next step index
        if t < len(prebaked_traj):
            tx, ty, th, tv = prebaked_traj[t]
            # Compute acceleration needed to reach the target velocity
            dv = tv - obs.ego.velocity
            accel = dv / obs.dt if obs.dt > 0 else 0.0
            # Compute steering to reach the target lateral position
            # Simple: use heading difference as proxy for steering
            dh = th - obs.ego.heading
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            steer = np.clip(dh * 2.0, -0.5, 0.5)
            return Action(accel, steer)
        return Action(0.0, 0.0)
    return controller


def lane_change_controller_factory(y_from: float, y_to: float,
                                   t_start: float, t_dur: float,
                                   vx_before: float,
                                   vx_after: float = None,
                                   t_decel_dur: float = 3.0):
    """Controller that drives straight, performs a smooth lane change,
    then optionally adjusts speed — all procedurally.

    Computes the desired heading from the analytic lateral velocity profile
    and uses a heading-tracking controller to steer, avoiding the
    position-error oscillation of a naive lateral P-controller.
    """
    if vx_after is None:
        vx_after = vx_before
    dy = y_to - y_from
    t_lc_end = t_start + t_dur
    t_speed_end = t_lc_end + t_decel_dur

    def controller(obs: Observation) -> Action:
        ego = obs.ego
        t = obs.time_step * obs.dt  # current sim time

        # --- desired longitudinal speed ---
        if t <= t_lc_end:
            v_target = vx_before
        elif t < t_speed_end:
            s = (t - t_lc_end) / t_decel_dur
            blend = s - np.sin(2 * np.pi * s) / (2 * np.pi)
            v_target = vx_before + (vx_after - vx_before) * blend
        else:
            v_target = vx_after

        accel = 2.0 * (v_target - ego.velocity)

        # --- desired heading from analytic lateral velocity ---
        if t < t_start:
            vy_desired = 0.0
            y_target = y_from
        elif t < t_lc_end:
            s = (t - t_start) / t_dur
            vy_desired = dy / t_dur * (1 - np.cos(2 * np.pi * s))
            y_target = y_from + dy * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
        else:
            vy_desired = 0.0
            y_target = y_to

        # Feedforward: desired heading from the velocity profile
        heading_ff = np.arctan2(vy_desired, v_target)

        # Feedback: small correction for accumulated lateral drift
        y_err = y_target - ego.y
        heading_fb = np.clip(y_err * 0.3, -0.05, 0.05)

        heading_desired = heading_ff + heading_fb

        # Steering tracks the desired heading
        heading_err = heading_desired - ego.heading
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        steer = np.clip(heading_err * 2.0, -0.3, 0.3)

        return Action(np.clip(accel, -MAX_ACCEL, MAX_ACCEL), steer)

    return controller


CONTROLLER_REGISTRY = {
    "constant_velocity": constant_velocity_controller,
    "simple_acc": simple_acc_controller,
}


# ---------------------------------------------------------------------------
#  Simulation engine
# ---------------------------------------------------------------------------
class Simulation:
    def __init__(self, scenario_name: str,
                 ego_controller: Callable[[Observation], Action] = None,
                 vehicle_overrides: Dict[int, Callable] = None):
        """
        Args:
            scenario_name:     Name of .xml file (without extension).
            ego_controller:    Controller for the ego vehicle.
            vehicle_overrides: Dict of obstacle_id → controller.
                               Overrides the default (playback from XML).
                               Use this to make non-ego vehicles procedural.
        """
        path = os.path.join(SCENARIO_DIR, f"{scenario_name}.xml")
        self.scenario, self.pps = CommonRoadFileReader(path).open()
        self.dt = self.scenario.dt
        self.vehicles: Dict[int, SimVehicle] = {}

        if vehicle_overrides is None:
            vehicle_overrides = {}

        # --- Create non-ego vehicles ---
        for obs in self.scenario.obstacles:
            oid = obs.obstacle_id
            s0 = obs.initial_state
            init = VehicleState(
                x=s0.position[0], y=s0.position[1],
                heading=s0.orientation, velocity=s0.velocity,
            )
            # Extract pre-baked trajectory
            prebaked = [(s0.position[0], s0.position[1],
                         s0.orientation, s0.velocity)]
            if obs.prediction is not None:
                for s in obs.prediction.trajectory.state_list:
                    prebaked.append((s.position[0], s.position[1],
                                    s.orientation, s.velocity))

            if oid in vehicle_overrides:
                ctrl = vehicle_overrides[oid]
            else:
                ctrl = playback_controller_factory(prebaked)

            self.vehicles[oid] = SimVehicle(
                vehicle_id=oid, initial_state=init,
                controller=ctrl, is_ego=False,
                prebaked_traj=prebaked,
            )

        # --- Create ego vehicle ---
        pp = list(self.pps.planning_problem_dict.values())[0]
        s0 = pp.initial_state
        ego_init = VehicleState(
            x=s0.position[0], y=s0.position[1],
            heading=s0.orientation, velocity=s0.velocity,
        )
        ego_ctrl = ego_controller or simple_acc_controller
        self.ego = SimVehicle(
            vehicle_id=0, initial_state=ego_init,
            controller=ego_ctrl, is_ego=True,
        )

        # Max steps from the longest pre-baked trajectory
        self.n_steps = max(
            (len(v.prebaked_traj) for v in self.vehicles.values()
             if v.prebaked_traj),
            default=200,
        )

    def _build_observation(self, for_vehicle: SimVehicle) -> Observation:
        """Build an Observation from the perspective of *for_vehicle*."""
        others = {}
        future_trajs = {}
        t = for_vehicle.state.time_step

        if for_vehicle.is_ego:
            for vid, veh in self.vehicles.items():
                others[vid] = veh.state
                # Provide future trajectory from pre-baked data if available
                if veh.prebaked_traj is not None:
                    future_trajs[vid] = veh.prebaked_traj[t + 1:]
        else:
            others[self.ego.id] = self.ego.state
            for vid, veh in self.vehicles.items():
                if vid != for_vehicle.id:
                    others[vid] = veh.state

        return Observation(
            ego=for_vehicle.state,
            others=others,
            future_trajectories=future_trajs,
            time_step=t,
            dt=self.dt,
        )

    def step(self) -> bool:
        """Advance all vehicles by one tick. Returns False when done."""
        t = self.ego.state.time_step
        if t >= self.n_steps - 1:
            return False

        # Step non-ego vehicles first (they act on current-frame info)
        for vid, veh in self.vehicles.items():
            obs = self._build_observation(veh)
            veh.step(obs, self.dt)

        # Step ego
        ego_obs = self._build_observation(self.ego)
        self.ego.step(ego_obs, self.dt)

        return True

    def run_all(self):
        while self.step():
            pass


# ---------------------------------------------------------------------------
#  Live visualisation
# ---------------------------------------------------------------------------
class SimulationRenderer:
    def __init__(self, sim: Simulation, title: str = "",
                 view_half_x: float = 60.0):
        self.sim = sim
        self.title = title
        self.view_half_x = view_half_x
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.y_lo = -2.0
        self.y_hi = N_LANES * LANE_WIDTH + 2.0

    def _draw_vehicle(self, ax, cx, cy, orient, color,
                      alpha=0.7, lw=1.5, label=None):
        rect = mpatches.FancyBboxPatch(
            (-VEH_LENGTH / 2, -VEH_WIDTH / 2), VEH_LENGTH, VEH_WIDTH,
            boxstyle="round,pad=0.15",
            facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw,
        )
        tr = Affine2D().rotate(orient).translate(cx, cy) + ax.transData
        rect.set_transform(tr)
        ax.add_patch(rect)
        if label:
            ax.annotate(label, xy=(cx, cy), fontsize=7, fontweight="bold",
                        ha="center", va="center", color="white", zorder=20)

    def draw_frame(self, t: int):
        ax = self.ax
        ax.clear()

        # Lane markings
        for i in range(N_LANES + 1):
            y = i * LANE_WIDTH
            is_edge = (i == 0 or i == N_LANES)
            ax.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                       linestyle="-" if is_edge else "--", zorder=1)
        for i in range(N_LANES):
            ax.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                       color="#f5f5f5", zorder=0)

        # Ego goal region
        for pp in self.sim.pps.planning_problem_dict.values():
            for gs in pp.goal.state_list:
                if hasattr(gs, "position") and gs.position is not None:
                    shape = gs.position
                    if hasattr(shape, "center") and hasattr(shape, "length"):
                        cx, cy = shape.center
                        rect = mpatches.Rectangle(
                            (cx - shape.length / 2, cy - shape.width / 2),
                            shape.length, shape.width,
                            linewidth=2, edgecolor=EGO_COLOR,
                            facecolor=EGO_COLOR, alpha=0.10,
                            linestyle="--", zorder=2,
                        )
                        ax.add_patch(rect)

        # Non-ego vehicles
        sorted_ids = sorted(self.sim.vehicles.keys())
        for idx, vid in enumerate(sorted_ids):
            veh = self.sim.vehicles[vid]
            s = veh.history[min(t, len(veh.history) - 1)]
            color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
            self._draw_vehicle(ax, s.x, s.y, s.heading, color, alpha=0.7)
            ax.annotate(f"{vid}", xy=(s.x, s.y), fontsize=7,
                        fontweight="bold", ha="center", va="center",
                        color="white", zorder=20)
            # Speed annotation
            ax.annotate(f"{s.velocity:.1f} m/s",
                        xy=(s.x, s.y + VEH_WIDTH),
                        fontsize=6, ha="center", color=color, zorder=15)
            # Trail
            trail_start = max(0, t - 30)
            trail = veh.history[trail_start:min(t + 1, len(veh.history))]
            if len(trail) > 1:
                ax.plot([h.x for h in trail], [h.y for h in trail],
                        "-", color=color, alpha=0.25, linewidth=2, zorder=1)

        # Ego vehicle
        ego_s = self.sim.ego.history[min(t, len(self.sim.ego.history) - 1)]
        self._draw_vehicle(ax, ego_s.x, ego_s.y, ego_s.heading,
                           EGO_COLOR, alpha=0.85, label="EGO")
        ax.annotate(f"{ego_s.velocity:.1f} m/s",
                    xy=(ego_s.x, ego_s.y + VEH_WIDTH),
                    fontsize=6, ha="center", color=EGO_COLOR, zorder=15)
        trail_start = max(0, t - 30)
        trail = self.sim.ego.history[trail_start:min(t + 1, len(self.sim.ego.history))]
        if len(trail) > 1:
            ax.plot([h.x for h in trail], [h.y for h in trail],
                    "-", color=EGO_COLOR, alpha=0.4, linewidth=2, zorder=1)

        # Camera
        ax.set_xlim(ego_s.x - self.view_half_x, ego_s.x + self.view_half_x)
        ax.set_ylim(self.y_lo, self.y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        time_s = t * self.sim.dt
        ax.set_title(
            f"{self.title}    t = {time_s:.1f} s   "
            f"v_ego = {ego_s.velocity:.1f} m/s   "
            f"a_ego = {ego_s.acceleration:.1f} m/s\u00b2",
            fontsize=11,
        )

    def animate(self, save_path: str = None, interval_ms: int = 50,
                headless: bool = False):
        """Run live animation. If save_path given, also save to file."""
        self.sim.run_all()
        n_frames = len(self.sim.ego.history)

        def update(frame):
            self.draw_frame(frame)

        anim = animation.FuncAnimation(
            self.fig, update, frames=n_frames,
            interval=interval_ms, repeat=False,
        )

        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".mp4":
                writer = animation.FFMpegWriter(fps=int(1000 / interval_ms))
            else:
                writer = animation.PillowWriter(fps=int(1000 / interval_ms))
            print(f"Saving animation to {save_path} ({n_frames} frames) ...")
            anim.save(save_path, writer=writer)
            print("  Done.")

        if not headless:
            plt.show()

        plt.close(self.fig)


# ---------------------------------------------------------------------------
#  Experiment presets
# ---------------------------------------------------------------------------
def build_exp1(ego_controller):
    """Exp 1: lead vehicle driven by constant-velocity controller."""
    return Simulation(
        "exp1_simple_acc",
        ego_controller=ego_controller,
        vehicle_overrides={
            100: constant_velocity_controller,  # lead: procedural constant-v
        },
    )


def build_exp2(ego_controller):
    """Exp 2: lead constant-v + merger with lane-change controller."""
    return Simulation(
        "exp2_merge_in_front",
        ego_controller=ego_controller,
        vehicle_overrides={
            100: constant_velocity_controller,
            101: lane_change_controller_factory(
                y_from=5.25, y_to=1.75,
                t_start=4.0, t_dur=3.0,
                vx_before=17.0, vx_after=15.0, t_decel_dur=3.0,
            ),
        },
    )


def build_exp3(ego_controller):
    """Exp 3: two vehicles in target lane, both constant-v."""
    return Simulation(
        "exp3_ego_merge",
        ego_controller=ego_controller,
        vehicle_overrides={
            100: constant_velocity_controller,  # front vehicle in lane 2
            101: constant_velocity_controller,  # rear vehicle in lane 2
        },
    )


EXPERIMENT_BUILDERS = {
    "exp1_simple_acc": build_exp1,
    "exp2_merge_in_front": build_exp2,
    "exp3_ego_merge": build_exp3,
}

SCENARIO_TITLES = {
    "exp1_simple_acc": "Exp 1: Simple ACC",
    "exp2_merge_in_front": "Exp 2: Merge in Front",
    "exp3_ego_merge": "Exp 3: Ego Merge",
}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a CommonRoad car-following experiment")
    parser.add_argument("--scenario", "-s", required=True,
                        choices=list(EXPERIMENT_BUILDERS.keys()),
                        help="Experiment to run")
    parser.add_argument("--controller", "-c", default="simple_acc",
                        choices=list(CONTROLLER_REGISTRY.keys()),
                        help="Ego controller (default: simple_acc)")
    parser.add_argument("--save", action="store_true",
                        help="Save animation to .gif")
    parser.add_argument("--headless", action="store_true",
                        help="No live window (use with --save)")
    parser.add_argument("--interval", type=int, default=50,
                        help="Animation interval in ms (default: 50)")
    parser.add_argument("--playback", action="store_true",
                        help="Use pre-baked XML trajectories for non-ego "
                             "vehicles instead of procedural controllers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.headless:
        matplotlib.use("Agg")

    ego_ctrl = CONTROLLER_REGISTRY[args.controller]
    title = SCENARIO_TITLES.get(args.scenario, args.scenario)

    if args.playback:
        # Fall back to XML trajectories for non-ego vehicles
        sim = Simulation(args.scenario, ego_controller=ego_ctrl)
    else:
        # Use procedural controllers for all vehicles
        builder = EXPERIMENT_BUILDERS[args.scenario]
        sim = builder(ego_ctrl)

    print(f"Running {args.scenario}  ego={args.controller}  "
          f"non-ego={'playback' if args.playback else 'procedural'}")

    renderer = SimulationRenderer(sim, title=title)

    save_path = None
    if args.save:
        save_path = os.path.join(
            FIG_DIR, f"{args.scenario}_{args.controller}.gif")

    renderer.animate(save_path=save_path, interval_ms=args.interval,
                     headless=args.headless)
