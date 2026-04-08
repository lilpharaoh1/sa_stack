"""
Create CommonRoad scenarios for three car-following / merging experiments.

Experiments:
  1. Simple ACC  – ego follows a lead vehicle at constant speed.
  2. Merge-in-front – same as (1) but a vehicle from the adjacent lane
     merges in front of the ego.
  3. Ego merge – ego must merge into an adjacent lane between two vehicles.

Road network: straight 3-lane highway, 500 m long, 3.5 m lane width.
All non-ego vehicles get pre-computed trajectories attached as
TrajectoryPrediction so they can be visualised directly.

Usage (from repo root, with carla-igp2 conda env):
    python experiments/commonroad/create_scenarios.py
"""

import os
import numpy as np

from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.lanelet import Lanelet, LaneletType, LineMarking
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, KSState
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import (
    PlanningProblem, PlanningProblemSet, GoalRegion,
)
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.util import Interval, AngleInterval

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
DT = 0.1            # time-step size [s]
N_STEPS = 200       # number of time steps  (20 s)
ROAD_LENGTH = 500.0  # [m]
LANE_WIDTH = 3.5     # [m]
N_LANES = 3
N_BOUNDARY_PTS = 50  # polyline resolution for each lanelet boundary
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 1.8

OUT_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def lane_center_y(lane_idx: int) -> float:
    """Return the y-coordinate of the lane centre (0-indexed from bottom)."""
    return (lane_idx + 0.5) * LANE_WIDTH


def make_boundary_pts(y: float) -> np.ndarray:
    """Straight horizontal polyline at height *y*, from x=0 to ROAD_LENGTH."""
    xs = np.linspace(0.0, ROAD_LENGTH, N_BOUNDARY_PTS)
    return np.column_stack([xs, np.full_like(xs, y)])


def build_lanelet_network(scenario: Scenario):
    """Add a 3-lane straight highway to *scenario*."""
    lanelet_ids = list(range(1, N_LANES + 1))  # 1, 2, 3  (bottom → top)

    for i, lid in enumerate(lanelet_ids):
        left_y = (i + 1) * LANE_WIDTH
        right_y = i * LANE_WIDTH
        center_y_val = (left_y + right_y) / 2.0

        left_v = make_boundary_pts(left_y)
        right_v = make_boundary_pts(right_y)
        center_v = make_boundary_pts(center_y_val)

        adj_left = lanelet_ids[i + 1] if i + 1 < N_LANES else None
        adj_right = lanelet_ids[i - 1] if i - 1 >= 0 else None

        lanelet = Lanelet(
            left_vertices=left_v,
            center_vertices=center_v,
            right_vertices=right_v,
            lanelet_id=lid,
            adjacent_left=adj_left,
            adjacent_left_same_direction=True if adj_left else None,
            adjacent_right=adj_right,
            adjacent_right_same_direction=True if adj_right else None,
            line_marking_left_vertices=LineMarking.DASHED if i + 1 < N_LANES else LineMarking.SOLID,
            line_marking_right_vertices=LineMarking.SOLID if i == 0 else LineMarking.DASHED,
            lanelet_type={LaneletType.HIGHWAY},
        )
        scenario.add_objects(lanelet)


def constant_velocity_trajectory(x0: float, y0: float, vx: float,
                                 heading: float = 0.0,
                                 n_steps: int = N_STEPS) -> list:
    """Straight-line constant-velocity state list (KSState, steps 1..n)."""
    states = []
    for k in range(1, n_steps + 1):
        t = k * DT
        states.append(KSState(
            time_step=k,
            position=np.array([x0 + vx * t, y0]),
            steering_angle=0.0,
            velocity=vx,
            orientation=heading,
        ))
    return states


def lane_change_trajectory(x0: float, y_from: float, y_to: float,
                           vx: float, t_start: float, t_dur: float,
                           vx_after: float = None,
                           t_decel_dur: float = 3.0,
                           heading: float = 0.0,
                           n_steps: int = N_STEPS) -> list:
    """Trajectory with a smooth (sinusoidal) lateral lane change.

    The vehicle drives straight at *vx* until *t_start*, then smoothly
    transitions from *y_from* to *y_to* over *t_dur* seconds.

    If *vx_after* is set (and differs from *vx*), the vehicle smoothly
    adjusts its longitudinal speed to *vx_after* over *t_decel_dur* seconds
    immediately after the lane change completes.
    """
    if vx_after is None:
        vx_after = vx

    t_lc_end = t_start + t_dur          # when lane change finishes
    t_speed_end = t_lc_end + t_decel_dur  # when speed transition finishes

    states = []
    dy = y_to - y_from
    # We integrate x position to handle varying speed correctly.
    x_prev = x0
    v_prev = vx

    for k in range(1, n_steps + 1):
        t = k * DT

        # --- longitudinal speed ---
        if t <= t_lc_end:
            v_cur = vx
        elif t < t_speed_end:
            # smooth sinusoidal speed blend from vx → vx_after
            s = (t - t_lc_end) / t_decel_dur
            blend = s - np.sin(2 * np.pi * s) / (2 * np.pi)
            v_cur = vx + (vx_after - vx) * blend
        else:
            v_cur = vx_after

        x = x_prev + v_cur * DT

        # --- lateral position ---
        if t < t_start:
            y = y_from
            vy = 0.0
        elif t < t_lc_end:
            s = (t - t_start) / t_dur  # normalised progress [0, 1]
            y = y_from + dy * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
            vy = dy / t_dur * (1 - np.cos(2 * np.pi * s))
        else:
            y = y_to
            vy = 0.0

        orient = np.arctan2(vy, v_cur)
        states.append(KSState(
            time_step=k,
            position=np.array([x, y]),
            steering_angle=0.0,
            velocity=np.hypot(v_cur, vy),
            orientation=orient,
        ))
        x_prev = x
        v_prev = v_cur

    return states


def make_obstacle(obs_id: int, x0: float, y0: float,
                  state_list: list, heading: float = 0.0,
                  velocity: float = 15.0) -> DynamicObstacle:
    """Wrap a state list into a DynamicObstacle with TrajectoryPrediction."""
    shape = Rectangle(length=VEHICLE_LENGTH, width=VEHICLE_WIDTH)
    initial = InitialState(
        time_step=0,
        position=np.array([x0, y0]),
        orientation=heading,
        velocity=velocity,
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )
    traj = Trajectory(initial_time_step=1, state_list=state_list)
    pred = TrajectoryPrediction(trajectory=traj, shape=shape)
    return DynamicObstacle(
        obstacle_id=obs_id,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=initial,
        prediction=pred,
    )


def make_planning_problem(pp_id: int, x0: float, y0: float,
                          v0: float, heading: float,
                          goal_x_min: float, goal_x_max: float,
                          goal_y_center: float,
                          goal_lane_id: int = None) -> PlanningProblem:
    """Create a planning problem (ego vehicle start + goal region)."""
    initial = InitialState(
        time_step=0,
        position=np.array([x0, y0]),
        orientation=heading,
        velocity=v0,
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )
    goal_state = KSState(
        time_step=Interval(0, N_STEPS),
        position=Rectangle(
            length=goal_x_max - goal_x_min,
            width=LANE_WIDTH,
            center=np.array([(goal_x_min + goal_x_max) / 2.0, goal_y_center]),
        ),
        velocity=Interval(0.0, 30.0),
        orientation=AngleInterval(-0.2, 0.2),
    )
    goal = GoalRegion(state_list=[goal_state])
    return PlanningProblem(
        planning_problem_id=pp_id,
        initial_state=initial,
        goal_region=goal,
    )


def save_scenario(scenario: Scenario, pps: PlanningProblemSet,
                  filename: str):
    path = os.path.join(OUT_DIR, filename)
    fw = CommonRoadFileWriter(
        scenario, pps,
        author="Emran",
        affiliation="",
        source="epistemic-planning experiments",
        tags={Tag.HIGHWAY},
    )
    fw.write_to_file(path, OverwriteExistingFile.ALWAYS)
    print(f"  Saved → {path}")


# ===================================================================
#  Experiment 1: Simple ACC (car following)
# ===================================================================
def create_exp1_simple_acc():
    print("Creating Experiment 1: Simple ACC ...")
    scenario = Scenario(dt=DT, scenario_id=ScenarioID(
        map_name="Highway", map_id=1, configuration_id=1,
    ))
    build_lanelet_network(scenario)

    # --- Non-ego: lead vehicle in lane 1 (bottom lane) ---
    lead_y = lane_center_y(0)      # ~1.75
    lead_x0 = 80.0
    lead_v = 15.0                  # 15 m/s ≈ 54 km/h
    lead_states = constant_velocity_trajectory(lead_x0, lead_y, lead_v)
    lead_obs = make_obstacle(100, lead_x0, lead_y, lead_states,
                             velocity=lead_v)
    scenario.add_objects(lead_obs)

    # --- Ego planning problem: behind lead in lane 1 ---
    ego_x0 = 40.0
    ego_v0 = 15.0
    pp = make_planning_problem(
        pp_id=1, x0=ego_x0, y0=lead_y, v0=ego_v0, heading=0.0,
        goal_x_min=350.0, goal_x_max=450.0, goal_y_center=lead_y,
    )
    pps = PlanningProblemSet(planning_problem_list=[pp])

    save_scenario(scenario, pps, "exp1_simple_acc.xml")
    return scenario, pps


# ===================================================================
#  Experiment 2: Merge in front of ego
# ===================================================================
def create_exp2_merge_in_front():
    print("Creating Experiment 2: Merge in front ...")
    scenario = Scenario(dt=DT, scenario_id=ScenarioID(
        map_name="Highway", map_id=1, configuration_id=2,
    ))
    build_lanelet_network(scenario)

    ego_lane_y = lane_center_y(0)       # lane 1 centre  ~1.75
    adj_lane_y = lane_center_y(1)       # lane 2 centre  ~5.25

    # --- Non-ego 1: lead vehicle in lane 1 (constant velocity) ---
    lead_x0 = 100.0
    lead_v = 15.0
    lead_states = constant_velocity_trajectory(lead_x0, ego_lane_y, lead_v)
    scenario.add_objects(
        make_obstacle(100, lead_x0, ego_lane_y, lead_states, velocity=lead_v))

    # --- Non-ego 2: merging vehicle starts in lane 2, merges into lane 1 ---
    #     Starts alongside / slightly ahead of ego, merges at t=4 s over 3 s.
    #     After completing the lane change, decelerates to match lead speed
    #     so it doesn't drive through the lead vehicle.
    merge_x0 = 70.0
    merge_v = 17.0   # slightly faster than ego during approach
    merge_states = lane_change_trajectory(
        x0=merge_x0, y_from=adj_lane_y, y_to=ego_lane_y,
        vx=merge_v, t_start=4.0, t_dur=3.0,
        vx_after=lead_v, t_decel_dur=3.0,
    )
    scenario.add_objects(
        make_obstacle(101, merge_x0, adj_lane_y, merge_states,
                      velocity=merge_v))

    # --- Ego planning problem ---
    ego_x0 = 50.0
    pp = make_planning_problem(
        pp_id=1, x0=ego_x0, y0=ego_lane_y, v0=15.0, heading=0.0,
        goal_x_min=350.0, goal_x_max=450.0, goal_y_center=ego_lane_y,
    )
    pps = PlanningProblemSet(planning_problem_list=[pp])

    save_scenario(scenario, pps, "exp2_merge_in_front.xml")
    return scenario, pps


# ===================================================================
#  Experiment 3: Ego merges into adjacent lane
# ===================================================================
def create_exp3_ego_merge():
    print("Creating Experiment 3: Ego merge ...")
    scenario = Scenario(dt=DT, scenario_id=ScenarioID(
        map_name="Highway", map_id=1, configuration_id=3,
    ))
    build_lanelet_network(scenario)

    ego_lane_y = lane_center_y(0)       # lane 1  ~1.75
    target_lane_y = lane_center_y(1)    # lane 2  ~5.25

    # --- Non-ego 1: front vehicle in target lane ---
    front_x0 = 100.0
    front_v = 14.0
    front_states = constant_velocity_trajectory(
        front_x0, target_lane_y, front_v)
    scenario.add_objects(
        make_obstacle(100, front_x0, target_lane_y, front_states,
                      velocity=front_v))

    # --- Non-ego 2: rear vehicle in target lane ---
    rear_x0 = 55.0
    rear_v = 14.0
    rear_states = constant_velocity_trajectory(
        rear_x0, target_lane_y, rear_v)
    scenario.add_objects(
        make_obstacle(101, rear_x0, target_lane_y, rear_states,
                      velocity=rear_v))

    # --- Ego planning problem: starts in lane 1, goal in lane 2 ---
    ego_x0 = 70.0
    ego_v0 = 15.0
    pp = make_planning_problem(
        pp_id=1, x0=ego_x0, y0=ego_lane_y, v0=ego_v0, heading=0.0,
        goal_x_min=300.0, goal_x_max=450.0, goal_y_center=target_lane_y,
    )
    pps = PlanningProblemSet(planning_problem_list=[pp])

    save_scenario(scenario, pps, "exp3_ego_merge.xml")
    return scenario, pps


# ===================================================================
#  Main
# ===================================================================
if __name__ == "__main__":
    s1, p1 = create_exp1_simple_acc()
    s2, p2 = create_exp2_merge_in_front()
    s3, p3 = create_exp3_ego_merge()
    print("\nAll scenarios created.")
