import carla

import igp2 as ip
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

import sys
import os
from typing import Dict, List, Any

import logging
import igp2 as ip
import numpy as np
import argparse
import json
import time
from shapely.geometry import Polygon
from datetime import datetime
from typing import List, Tuple, Dict
from igp2.pgp.plot_pgp_trajectories import plot_pgp_trajectories
from igp2.pgp.plot_graph_traversals import plot_graph_traversals
from igp2.pgp.plot_agent_histories import plot_agent_histories


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
        print(spawn)
        print(layout)
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

import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.ioff()

def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

class VideoWriter:
    def __init__(self, path, fps=10):
        self.path = path
        self.fps = fps
        self._writer = None
        self._size = None
        self._count = 0

    def write_rgb(self, rgb):
        if rgb is None:
            return

        h, w = rgb.shape[:2]
        if self._size is None:
            self._size = (w, h)

        # HARD CHECK: if this triggers, your corruption is from variable frame sizes
        if (w, h) != self._size:
            raise RuntimeError(f"Frame size changed: got {(w,h)} expected {self._size}")

        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open for path={self.path}")

        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._count += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        print(f"[VideoWriter] wrote {self._count} frames to {self.path}")


def capture_new_figs_rgb(plot_fn, *args, **kwargs):
    before = set(plt.get_fignums())
    plot_fn(*args, **kwargs)
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)

    rgbs = []
    for fnum in new_nums:
        fig = plt.figure(fnum)

        # FORCE CONSISTENT OUTPUT CANVAS SIZE
        fig.set_size_inches(12, 6, forward=True)
        fig.set_dpi(150)

        # Avoid layout that can resize canvas unpredictably
        # fig.tight_layout()  # <-- do NOT call here if it causes variability

        rgb = fig_to_rgb(fig)
        rgbs.append(rgb)
        plt.close(fig)

    return rgbs

if __name__ == '__main__':
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


    scenario = "scenario1"
    scenario_xodr = f"scenarios/maps/{scenario}.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_xodr)
    try:
        scenario_config = json.load(open(os.path.join("scenarios", "configs", f"{scenario}.json"), "r"))
    except FileNotFoundError as e:
        logger.exception(msg="No configuration file was found for the given arguments", exc_info=e)
        raise e

    print(scenario_config["agents"])
    agent_spawns = []
    goals = {}
    for agent_config in scenario_config["agents"]:
        spawn_box = ip.Box(np.array(agent_config['spawn']['box']['center']), agent_config['spawn']['box']['length'], \
                                    agent_config['spawn']['box']['width'], agent_config['spawn']['box']['heading'])
        vel_range = agent_config['spawn']['velocity']
        agent_spawns.append((spawn_box, vel_range))

        goal = ip.BoxGoal(ip.Box(np.array(agent_config['goal']['box']['center']), agent_config['goal']['box']['length'], \
                                          agent_config['goal']['box']['width'], agent_config['goal']['box']['heading']))

        goals[agent_config['id']] = goal

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  agent_spawns)

    # ip.plot_map(scenario_map, markings=True, midline=True)
    # for spawn in agent_spawns:
    #     plt.plot(*list(zip(*spawn[0].boundary)))
    # for aid, state in frame.items():
    #     plt.plot(*state.position, marker="x", color='k')
    #     plt.text(*state.position, aid)
    # for aid, goal in goals.items():
    #     plt.plot(*goal.box.center, marker="x", color='k')
    #     plt.text(*goal.box.center, aid)
    #     plt.plot(*list(zip(*goal.box.boundary)), c="g")
    # plt.gca().add_patch(plt.Circle(frame[0].position, 100, color='b', fill=False))
    # plt.show()

    cost_factors = {"time": 0.1, "velocity": 0.0, "acceleration": 0.1, "jerk": 0., "heading": 0.0,
                    "angular_velocity": 0.1, "angular_acceleration": 0.1, "curvature": 0.0, "safety": 0.}
    reward_factors = {"time": 1.0, "jerk": -0.1, "angular_acceleration": -0.2, "curvature": -0.1}
    carla_sim = ip.carlasim.CarlaSim(xodr=scenario_xodr, carla_path=args.carla_path)

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
                                       pgp_drive=True,
                                       pgp_control=True,
                                       store_results="all")
            carla_sim.add_agent(agents[aid], "ego")
            # carla_sim.spectator.set_location(
            #     # carla.Location(frame[aid].position[0], -frame[aid].position[1], 5.0)
            #     carla.Location(50.0, 0.0, 5.0)
            #     )
        else:
            agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps, pgp_drive=True, pgp_control=True)
            carla_sim.add_agent(agents[aid], None)

    observations = []
    actions = []
    colors = ['r', 'g', 'b', 'y', 'k']

    traj_video = VideoWriter("pgp_trajectories.mp4", fps=fps)
    trav_video = VideoWriter("pgp_traversals.mp4", fps=fps)  # optional

    record_trajectories = True
    record_traversals = False  
    for t in range(500):
        obs, acts = carla_sim.step()
        observations.append(obs)
        actions.append(acts)

        if carla_sim.timestep % carla_sim.pgp.interval != 0:
            continue

        agent_history = carla_sim.pgp.agent_history
        drive = carla_sim.pgp.drive
        drive_prob = carla_sim.pgp.drive_prob
        drive_traversal = carla_sim.pgp.drive_traversal

        # --- record only PGP trajectories ---
        if record_trajectories and drive is not None:
            rgbs = capture_new_figs_rgb(
                plot_pgp_trajectories,
                drive, drive_prob, agent_history, carla_sim.scenario_map,
                markings=True
            )
            for rgb in rgbs:
                # if multiple figs were created, write them all (or just rgbs[0])
                os.makedirs("debug_frames", exist_ok=True)

                # after you get rgb:
                cv2.imwrite(f"debug_frames/frame_{t:04d}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                traj_video.write_rgb(rgb)

        # --- optionally record traversals into separate video ---
        if record_traversals and drive_traversal is not None:
            rgbs = capture_new_figs_rgb(
                plot_graph_traversals,
                drive_traversal, agent_history, carla_sim.pgp.dataset, carla_sim.scenario_map,
                markings=True
            )
            for rgb in rgbs:
                trav_video.write_rgb(rgb)

    traj_video.close()
    trav_video.close()
