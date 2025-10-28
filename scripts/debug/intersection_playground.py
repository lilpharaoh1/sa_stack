from collections import defaultdict
import time
import igp2 as ip
import numpy as np
import random
import logging
import carla
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Any

import logging
import igp2 as ip
import numpy as np
import argparse
import json
from shapely.geometry import Polygon
from datetime import datetime
from typing import List, Tuple, Dict


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
    This script runs the full IGP2 system in a selected map or scenario using either the CARLA simulator or a
    simpler simulator for rapid testing, but less realistic simulations.
         """, formatter_class=argparse.RawTextHelpFormatter)

    # --------General arguments-------------#
    parser.add_argument('--save_log_path',
                        type=str,
                        default=None,
                        help='save log to the specified path')
    parser.add_argument('--seed',
                        default=None,
                        help="random seed to use",
                        type=int)
    parser.add_argument("--fps",
                        default=20,
                        help="framerate of the simulation",
                        type=int)
    parser.add_argument("--config_path",
                        type=str,
                        help="path to a configuration file.")
    parser.add_argument('--record', '-r',
                        help="whether to create an offline recording of the simulation",
                        action="store_true")  # TODO: Not implemented for simple simulator yet.
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="whether to display debugging plots and logging commands")
    parser.add_argument("--plot_map_only",
                        action="store_true",
                        default=False,
                        help="if true, only plot the scenario map and a random then exit the program")
    parser.add_argument("--plot",
                        type=int,
                        default=None,
                        help="display plots of the simulation with this period"
                             " when using the simple simulator.")
    parser.add_argument('--map', '-m',
                        default=None,
                        help="name of the map to use",
                        type=str)

    # -------Simulator specific config---------#
    parser.add_argument("--carla",
                        action="store_true",
                        default=False,
                        help="whether to use CARLA as the simulator instead of the simple simulator.")
    parser.add_argument('--server',
                        default="localhost",
                        help="server IP where CARLA is running",
                        type=str)
    parser.add_argument("--port",
                        default=2000,
                        help="port where CARLA is accessible on the server.",
                        type=int)
    parser.add_argument('--carla_path', '-p',
                        default="/opt/carla-simulator",
                        help="path to the directory where CARLA is installed. "
                             "Used to launch CARLA if not running.",
                        type=str)
    parser.add_argument('--launch_process',
                        default=False,
                        help="use this flag to launch a new process of CARLA instead of using a currently running one.",
                        action='store_true')
    parser.add_argument('--no_rendering',
                        help="whether to disable CARLA rendering",
                        action="store_true")
    parser.add_argument('--no_visualiser',
                        default=False,
                        help="whether to use detailed visualisation for the simulation",
                        action='store_true')
    parser.add_argument('--record_visualiser',
                        default=False,
                        help="whether to use store the PyGame surface during visualisation",
                        action='store_true')

    args = parser.parse_args()
    if args.plot is not None and args.carla:
        logger.debug("--plot is ignored when --carla is used.")
    if args.no_visualiser and args.record_visualiser:
        logger.debug("Using --no_visualiser with --record_visualiser. Latter option will be ignored.")
    return args


def setup_logging(main_logger: logging.Logger = None, debug: bool = False, log_path: str = None):
    # Add %(asctime)s  for time
    level = logging.DEBUG if debug else logging.INFO

    logging.getLogger("igp2.core.velocitysmoother").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger("")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if main_logger is not None:
        main_logger.setLevel(level)
        main_logger.addHandler(console_handler)

    if log_path:
        if not os.path.isdir(log_path):
            raise FileNotFoundError(f"Logging path {log_path} does not exist.")

        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"{log_path}/{date_time}.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


def load_config(args):
    if "map" in args and args.map is not None:
        path = os.path.join("scenarios", "configs", f"{args.map}.json")
    elif "config_path" in args and args.config_path is not None:
        path = args.config_path
    else:
        raise ValueError("No scenario was specified! Provide either --map or --config_path.")

    try:
        return json.load(open(path, "r"))
    except FileNotFoundError as e:
        logger.exception(msg="No configuration file was found for the given arguments", exc_info=e)
        raise e


def to_ma_list(ma_confs: List[Dict[str, Any]], agent_id: int,
               start_frame: Dict[int, ip.AgentState], scenario_map: ip.Map) \
        -> List[ip.MacroAction]:
    mas = []
    for config in ma_confs:
        config["open_loop"] = False
        frame = start_frame if not mas else mas[-1].final_frame
        if "target_sequence" in config:
            config["target_sequence"] = [scenario_map.get_lane(rid, lid) for rid, lid in config["target_sequence"]]
        ma = ip.MacroActionFactory.create(ip.MacroActionConfig(config), agent_id, frame, scenario_map)
        mas.append(ma)
    return mas


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
        best_lane = None
        max_overlap = 0.0
        for road in layout.roads.values():
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    overlap = lane.boundary.intersection(poly).area
                    if overlap > max_overlap:
                        best_lane = lane
                        max_overlap = overlap

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

# from scripts.experiments.scenarios.util import parse_args, generate_random_frame

ip.setup_logging()
logger = logging.getLogger(__name__)
args = parse_args()

# Set run parameters here
seed = 21 # args.seed
max_speed = 20.0 # args.max_speed
ego_id = 0
n_simulations = 2 # args.n_sim
fps = 20 # args.fps  # Simulator frequency
T = 2 # args.period  # MCTS update period

random.seed(seed)
np.random.seed(seed)
np.seterr(divide="ignore")
ip.Maneuver.MAX_SPEED = max_speed

scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/intersection.xodr")
veh1_spawn_box = ip.Box(np.array([-160.0, -102.5]), 10, 3.5, 0.0)
veh2_spawn_box = ip.Box(np.array([80, -97.5]), 10, 3.5, 0.0)
veh3_spawn_box = ip.Box(np.array([60, -204]), 3.5, 10, 0.25*np.pi)
frame = generate_random_frame(0, scenario_map,
                              [(veh1_spawn_box, (0.0, 0.0)),
                               (veh2_spawn_box, (0.0, 0.0)),
                               (veh3_spawn_box, (0.0, 0.0))])
goals = {
    0: ip.BoxGoal(ip.Box(np.array([-140.0, 0.0]), 5, 7, 0.25*np.pi)),
    1: ip.BoxGoal(ip.Box(np.array([30, -181.0]), 5, 7, 0.25*np.pi)),
    2: ip.BoxGoal(ip.Box(np.array([-140.0, 0.0]), 5, 7, 0.25*np.pi))
}
# ip.plot_map(scenario_map)
# plt.plot(*list(zip(*veh1_spawn_box.boundary)))
# plt.plot(*list(zip(*veh2_spawn_box.boundary)))
# plt.plot(*list(zip(*veh3_spawn_box.boundary)))
# for aid, state in frame.items():
#     plt.plot(*state.position, marker="x")
#     plt.text(*state.position, aid)
# for goal in goals.values():
#     plt.plot(*list(zip(*goal.box.boundary)), c="g")
# plt.show()

carla_sim = ip.carlasim.CarlaSim(xodr="scenarios/maps/intersection.xodr")
for actor in carla_sim.world.get_actors().filter("*vehicle*"):
    actor.destroy()
carla_sim.spectator.set_location(carla.Location(-50, 100, 25.0))
carla_sim.world.tick()

blueprint_library = carla_sim.world.get_blueprint_library()
blueprints = {
    0: blueprint_library.find('vehicle.audi.a2'),
    1: blueprint_library.find('vehicle.bmw.grandtourer'),
    2: blueprint_library.find('vehicle.ford.mustang')
}

agents = {}
agents_meta = ip.AgentMetadata.default_meta_frame(frame)
for aid in frame.keys():
    goal = goals[aid]
    agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps)
    carla_sim.add_agent(agents[aid], None, blueprint=blueprints[aid])

vels = defaultdict(list)
# target_speeds = {}
for t in range(40 * 20):
    obs, _ = carla_sim.step()
    frame = obs.frame
    for aid, state in frame.items():
        vels[aid].append(state.speed)
        # if carla_sim.agents[aid] is not None:
        #     target_speeds[aid] = carla_sim.agents[aid].target_speeds
    # time.sleep(1 / carla_sim.fps)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for i, (aid, agent_wrapper) in enumerate(carla_sim.agents.items()):
    avels = vels[aid]
    color = colors[i % len(colors)]
    plt.plot(range(len(avels)), avels, label=f"{aid} State Speed", c=color)
    # if len(avels) >= len(target_speeds[aid]):
    #     plt.plot(range(len(target_speeds[aid])), target_speeds[aid], "--", c=color, label=f"{aid} Target Speed")
    # else:
    #     plt.plot(range(len(avels)), target_speeds[aid][:len(avels)], "--", c=color, label=f"{aid} Target Speed")
    plt.legend()
    plt.show()