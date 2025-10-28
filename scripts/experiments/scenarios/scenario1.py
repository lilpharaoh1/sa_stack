import igp2 as ip
import random
import numpy as np
import matplotlib.pyplot as plt

# from scripts.experiments.scenarios.util import parse_args, generate_random_frame
from typing import List, Tuple, Dict

import igp2 as ip
import numpy as np
import argparse
from shapely.geometry import Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed of the simulation.")
    parser.add_argument("--max_speed", type=float, default=10.0, help="Maximum speed limit of the scenario.")
    parser.add_argument("--n_sim", type=int, default=10, help="Number of rollouts in MCTS.")
    parser.add_argument("--fps", type=int, default=20, help="Framerate of the simulation.")
    parser.add_argument("--period", type=float, default=2.0, help="Update frequency of MCTS in seconds.")
    parser.add_argument("--carla_path", type=str, default="C:\\Carla", help="Path to directory containing CARLA.")
    return parser.parse_args()


def generate_random_frame(ego: int,
                          layout: ip.Map,
                          spawn_vel_ranges: List[Tuple[ip.Box, Tuple[float, float]]]) -> Dict[int, ip.AgentState]:
    """ Generate a new frame with randomised spawns and velocities for each vehicle.

    Args:
        ego: The id of the ego
        layout: The road layout
        spawn_vel_ranges: A list of pairs of spawn ranges and velocity ranges.

    Returns:
        A new randomly generated frame
    """
    ret = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego):
        poly = Polygon(spawn.boundary)
        print(spawn)
        best_lane = layout.best_lane_at(spawn.center, max_distance=500.0)
        print(best_lane, spawn.center)

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d
        spawn_position = np.array(best_lane.point_at(position_d))

        speed = (vel[1] - vel[0]) * np.random.random() + vel[0]
        heading = best_lane.get_heading_at(position_d)
        ret[i] = ip.AgentState(time=0,
                               position=spawn_position,
                               velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
                               acceleration=np.array([0.0, 0.0]),
                               heading=heading)

    return ret


if __name__ == '__main__':
    ip.setup_logging()
    args = parse_args()

    # Set run parameters here
    seed = args.seed
    max_speed = args.max_speed
    ego_id = 0
    n_simulations = args.n_sim
    fps = args.fps  # Simulator frequency
    T = args.period  # MCTS update period

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")
    ip.Maneuver.MAX_SPEED = max_speed

    # Set randomised spawn parameters here
    ego_spawn_box = ip.Box(np.array([-80.0, -1.8]), 10, 3.5, 0.0)
    ego_vel_range = (5.0, max_speed)
    veh1_spawn_box = ip.Box(np.array([-70.0, 1.7]), 10, 3.5, 0.0)
    veh1_vel_range = (5.0, max_speed)
    veh2_spawn_box = ip.Box(np.array([-18.34, -25.5]), 3.5, 10, 0.0)
    veh2_vel_range = (5.0, max_speed)

    # Vehicle goals
    goals = {
        ego_id: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0)),
        ego_id + 1: ip.BoxGoal(ip.Box(np.array([-22, -25.5]), 3.5, 5, 0.0)),
        ego_id + 2: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0))
    }

    scenario_path = "scenarios/maps/scenario1.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_path)

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  [(ego_spawn_box, ego_vel_range),
                                   (veh1_spawn_box, veh1_vel_range),
                                   (veh2_spawn_box, veh2_vel_range)])

    ip.plot_map(scenario_map, markings=True, midline=True)
    plt.plot(*list(zip(*ego_spawn_box.boundary)))
    plt.plot(*list(zip(*veh1_spawn_box.boundary)))
    plt.plot(*list(zip(*veh2_spawn_box.boundary)))
    for aid, state in frame.items():
        plt.plot(*state.position, marker="x")
        plt.text(*state.position, aid)
    for goal in goals.values():
        plt.plot(*list(zip(*goal.box.boundary)), c="g")
    plt.gca().add_patch(plt.Circle(frame[0].position, 100, color='b', fill=False))
    plt.show()

    cost_factors = {"time": 0.1, "velocity": 0.0, "acceleration": 0.1, "jerk": 0., "heading": 0.0,
                    "angular_velocity": 0.1, "angular_acceleration": 0.1, "curvature": 0.0, "safety": 0.}
    reward_factors = {"time": 1.0, "jerk": -0.1, "angular_acceleration": -0.2, "curvature": -0.1}
    carla_sim = ip.carla.CarlaSim(xodr=scenario_path, carla_path=args.carla_path)

    agents = {}
    agents_meta = ip.AgentMetadata.default_meta_frame(frame)
    for aid in frame.keys():
        goal = goals[aid]

        if aid == ego_id:
            agents[aid] = ip.MCTSAgent(agent_id=aid,
                                       initial_state=frame[aid],
                                       t_update=T,
                                       scenario_map=scenario_map,
                                       goal=goal,
                                       cost_factors=cost_factors,
                                       reward_factors=reward_factors,
                                       fps=fps,
                                       n_simulations=n_simulations,
                                       view_radius=100,
                                       store_results="all")
            carla_sim.add_agent(agents[aid], "ego")
        else:
            agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps)
            carla_sim.add_agent(agents[aid], None)

    visualiser = ip.carla.Visualiser(carla_sim)
    visualiser.run()
