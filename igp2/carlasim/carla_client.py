import time
from typing import List, Dict, Optional, Union
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import carla
import subprocess
import psutil
import platform
import logging

from carla import Transform, Location, Rotation, Vector3D
from igp2.opendrive import Map
from igp2.vector_map import Dataset
from igp2.pgp.carlapgp import CarlaPGP
from igp2.agents.agent import Agent
from igp2.carlasim.traffic_manager import TrafficManager
from igp2.carlasim.carla_agent_wrapper import CarlaAgentWrapper
from igp2.core.vehicle import Observation, TrajectoryPrediction
from igp2.core.agentstate import AgentState


logger = logging.getLogger(__name__)


class CarlaSim:
    """ An interface to the CARLA simulator """
    TIMEOUT = 20.0

    def __init__(self,
                 fps: int = 20,
                 xodr: Union[str, Map] = None,
                 map_name: str = None,
                 server: str = "localhost",
                 port: int = 2000,
                 launch_process: bool = False,
                 carla_path: str = None,
                 record: bool = False,
                 rendering: bool = True):
        """ Launch the CARLA simulator and define a CarlaSim object, which keeps the connection to the CARLA
        server, manages agents and advances the environment.

        Note:
             The first agent of type MCTSAgent to be added to the simulation will be treated as the ego vehicle.

        Args:
            fps: number of frames simulated per second
            xodr: path to a .xodr OpenDrive file defining the road layout
            port: port to use for communication with the CARLA simulator
            launch_process: whether to launch a new CARLA instance
            carla_path: path to the root directory of the CARLA simulator
            rendering: controls whether graphics are rendered
        """
        logger.info(f"Launching CARLA simulation (server={server}, port={port}).")
        self.__launch_process = launch_process
        self.__carla_process = None
        if self.__launch_process:
            sys_name = platform.system()
            if sys_name == "Windows":
                if carla_path is None:
                    carla_path = r"C:\\Carla"
                if "CarlaUE4.exe" not in [p.name() for p in psutil.process_iter()]:
                    args = [os.path.join(carla_path, 'CarlaUE4.exe'), '-quality-level=Low',
                            f'-carla-rpc-port={port}', '-dx11']
                    self.__carla_process = subprocess.Popen(args)
            elif sys_name == "Linux":
                if carla_path is None:
                    carla_path = "/opt/carla-simulator"
                if "CarlaUE4.sh" not in [p.name() for p in psutil.process_iter()]:
                    args = [os.path.join(carla_path, 'CarlaUE4.sh'), '-quality-level=Low', f'-carla-rpc-port={port}']
                    self.__carla_process = subprocess.Popen(args)
            else:
                raise RuntimeError("Unsupported system!")

        self.__record = record
        self.__port = port
        self.__client = carla.Client(server, port)
        self.__client.set_timeout(self.TIMEOUT)  # seconds
        self.__wait_for_server()

        self.__agents = {}

        self.__world = self.__client.get_world()
        self.__map = self.__world.get_map()

        self.__scenario_map = None
        if isinstance(xodr, Map):
            self.__scenario_map = xodrf
        
        if not map_name is None:
            if self.__scenario_map is None:
                self.__scenario_map = Map.parse_from_opendrive(f"scenarios/maps/{map_name}.xodr")
                self.__pgp = CarlaPGP(f"scenarios/maps/{map_name}.xodr")
            # if map_name in self.__client.get_available_maps():
            # if not self.__client.get_world().get_map().name.endswith(map_name):
            try:
                self.__client.load_world(map_name)
            except RuntimeError as e:
                logger.debug(f"Failed to load world {map_name}: {e}")
                self.load_opendrive_world(self.__scenario_map.xodr_path)
            self.__pgp = CarlaPGP(self.__scenario_map.xodr_path)
        elif not xodr is None:
            if self.__scenario_map is None:
                self.__scenario_map = Map.parse_from_opendrive(xodr)
                self.__pgp = CarlaPGP(xodr)
            self.load_opendrive_world(self.__scenario_map.xodr_path)
        else:
            raise RuntimeError("Cannot load a map with the given parameters!")

        self.__original_settings = self.__world.get_settings()
        settings = self.__world.get_settings()
        settings.fixed_delta_seconds = 1 / fps
        settings.synchronous_mode = True
        settings.no_rendering_mode = not rendering
        self.__world.apply_settings(settings)

        self.__fps = fps
        self.__timestep = 0

        self.__record_path = None
        if self.__record:
            now = datetime.now()
            log_name = now.strftime("%d-%m-%Y_%H-%M-%S") + ".log"
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = Path(dir_path)
            repo_path = str(path.parent.parent)
            self.__record_path = os.path.join(repo_path, "scripts", "experiments", "data", "carla_recordings", log_name)
            logger.info(f"Recording simulation under path: {self.__client.start_recorder(self.__record_path, True)}")

        self.__spectator = self.__world.get_spectator()
        self.__spectator_parent = None
        self.__spectator_transform = None

        self.__traffic_manager = TrafficManager(self.__scenario_map)

        self.__world.tick()

    def __del__(self):
        self.__clear_agents()
        self.__world.apply_settings(self.__original_settings)
        if self.__record:
            self.__client.stop_recorder()

        self.__world.tick()

        if self.__launch_process:
            if self.__carla_process is not None:
                for child in psutil.Process(self.__carla_process.pid).children(recursive=True):
                    child.kill()

    def run(self, steps=400):
        """ Run the simulation for a number of time steps """
        for i in range(steps):
            self.step()
            time.sleep(1 / self.__fps)

    def step(self, tick: bool = True):
        """ Advance the simulation by one time step.

        Returns:
            Current observation of the environment before taking actions this step, and the actions that will be taken.
        """
        logger.debug(f"CARLA step {self.__timestep}.")

        if tick:
            self.__world.tick()

        self.__timestep += 1
        
        observation = self.__get_current_observation()
        prediction = self.__make_predictions(observation)
        actions = self.__take_actions(observation, prediction)
        self.__traffic_manager.update(self, observation)
        self.__update_spectator()

        return observation, actions

    def add_agent(self,
                  agent: Agent,
                  rolename: str = None,
                  blueprint: carla.ActorBlueprint = None):
        """ Add a vehicle to the simulation. Defaults to an Audi A2 for blueprints if not explicitly given.

        Args:
            agent: Agent to add.
            rolename: Unique name for the actor to spawn.
            blueprint: Optional blueprint defining the properties of the actor.

        Returns:
            The newly added actor.
        """
        if blueprint is None:
            blueprint_library = self.__world.get_blueprint_library()
            blueprint = blueprint_library.find('vehicle.audi.a2')

        if rolename is not None:
            blueprint.set_attribute('role_name', rolename)

        state = agent.state
        yaw = np.rad2deg(-state.heading)
        transform = Transform(Location(x=state.position[0], y=-state.position[1], z=0.1),
                              Rotation(yaw=yaw, roll=0.0, pitch=0.0))
        actor = self.__world.spawn_actor(blueprint, transform)
        actor.set_target_velocity(Vector3D(state.velocity[0], -state.velocity[1], 0.))

        carla_agent = CarlaAgentWrapper(agent, actor)
        self.agents[carla_agent.agent_id] = carla_agent
        self.__world.tick()
        logger.info(f"Added agent {carla_agent.agent_id} (actor {carla_agent.actor_id}).")

    def remove_agent(self, agent_id: int):
        """ Remove the given agent from the simulation.

        Args:
            agent_id: The ID of the agent to remove
        """
        logger.debug(f"Removing Agent {agent_id} with Actor {self.agents[agent_id].actor}")
        actor = self.agents[agent_id].actor
        actor.destroy()
        self.agents[agent_id].agent.alive = False
        self.agents[agent_id] = None
        self.__pgp.remove(agent_id)

    def __is_spawn_on_dead_end(self, position: np.ndarray, heading: float) -> bool:
        """Check if a spawn position is on a dead-end road (no successor lanes).

        A spawn point is considered problematic if the lane it's on has no
        successor lanes and is not near a junction, meaning the agent would
        have nowhere to go after reaching the end of the lane.

        Args:
            position: The (x, y) position in IGP2 coordinates (y-inverted from CARLA)
            heading: The heading angle in radians

        Returns:
            True if the spawn is on a dead-end road, False otherwise.
        """
        lane = self.scenario_map.best_lane_at(position, heading)
        if lane is None:
            return True  # No lane found, treat as problematic

        # Check if this lane has successors
        if lane.link is None or lane.link.successor is None or len(lane.link.successor) == 0:
            # No direct successors - check if it's a junction road (which is okay)
            if lane.parent_road.junction is not None:
                return False  # Junction roads can have no successors but connect elsewhere
            return True  # True dead-end

        return False

    def get_traffic_manager(self) -> "TrafficManager":
        """ Enables and returns the internal traffic manager of the simulation."""
        self.__traffic_manager.enabled = True

        spawn_points = []
        for p in self.__world.get_map().get_spawn_points():
            distances = [np.isclose((q.location - p.location).x, 0.0) and
                         np.isclose((q.location - p.location).y, 0.0) for q in spawn_points]
            if not any(distances) and len(self.scenario_map.roads_at((p.location.x, -p.location.y), drivable=True)) > 0:
                # Convert to IGP2 coordinates and get heading from spawn rotation
                igp2_position = np.array([p.location.x, -p.location.y])
                heading = np.deg2rad(-p.rotation.yaw)

                # Filter out spawn points on dead-end roads
                if self.__is_spawn_on_dead_end(igp2_position, heading):
                    logger.debug(f"Filtering out spawn point at {p.location} - dead-end road")
                    continue

                spawn_points.append(p)

        logger.info(f"Traffic manager initialized with {len(spawn_points)} valid spawn points")
        self.__traffic_manager.spawns = spawn_points
        return self.__traffic_manager

    def load_opendrive_world(self, xodr: str):
        with open(xodr, 'r') as f:
            opendrive = f.read()
        self.__client.set_timeout(60.0)
        self.__client.generate_opendrive_world(opendrive)
        self.__client.set_timeout(self.TIMEOUT)

    def attach_camera(self,
                      actor: carla.Actor,
                      transform: carla.Transform = carla.Transform(carla.Location(x=1.6, z=1.7))):
        """ Attach a camera to the back of the given actor in third-person view.

        Args:
            actor: The actor to follow
            transform: Optional transform to set the position of the camera
        """
        self.__spectator_parent = actor
        self.__spectator_transform = transform

    def get_ego(self, ego_name: str = "ego") -> Optional[CarlaAgentWrapper]:
        """ Returns the ego agent if it exists. """
        for agent in self.agents.values():
            if agent is not None and agent.name == ego_name:
                return agent
        return None

    def __update_spectator(self):
        if self.__spectator_parent is not None and self.__spectator_transform is not None:
            actor_transform = self.__spectator_parent.get_transform()
            actor_transform.location += self.__spectator_transform.location
            # actor_transform.rotation += self.__spectator_transform.rotation
            self.__spectator.set_transform(actor_transform)

    def __take_actions(self, observation: Observation, prediction: TrajectoryPrediction = None):
        commands = []
        controls = {}
        for agent_id, agent in self.agents.items():
            if agent is None:
                continue

            try:
                control = agent.next_control(observation, prediction)
            except RuntimeError as e:
                print(f"Couldn't generate control command for Agent {agent_id}: {e}")
                control = None

            if control is None:
                logger.debug("Removing agent because control is None")
                # self.remove_agent(agent.agent_id)
                self.__traffic_manager.remove_agent(agent, self)
                continue
            controls[agent_id] = control
            command = carla.command.ApplyVehicleControl(agent.actor, control)
            commands.append(command)

        self.__client.apply_batch_sync(commands)
        return controls

    def __make_predictions(self, observation: Observation) -> TrajectoryPrediction:
        self.__pgp.update(observation)
        if self.__timestep % self.__pgp.interval == 0:
            # EMRAN Clean up how we do this PLEASE!

            # for agent_id, agent in self.agents.items():
            #     print(f"Agent {agent_id}: {[(ma, type(ma)) for ma in agent.agent.macro_actions]}")
            #     try:
            #         print(f"Agent {agent_id}: {[ma.get_trajectory().path[:-1] for ma in agent.agent.macro_actions]}")
            #     except:
            #         print(f"Agent {agent_id}: Couldn't get trajectory")

            agent_waypoints = self.__get_agent_waypoints()
            self.__pgp.predict_trajectories(agent_waypoints=agent_waypoints)

        if self.__pgp.prediction is not None:
            return TrajectoryPrediction(
                # Pure predictions
                prediction=self.__pgp.prediction,
                prediction_prob=self.__pgp.prediction_prob,
                prediction_traversal=self.__pgp.prediction_traversal,
                # Driven trajectories
                drive=self.__pgp.drive,
                drive_prob=self.__pgp.drive_prob,
                drive_traversal=self.__pgp.drive_traversal,
                # Shared
                agent_history=self.__pgp.agent_history
            )
        else:
            return None

    def __get_current_observation(self) -> Observation:
        actor_list = self.__world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        agent_id_lookup = dict([(a.actor_id, a.agent_id) for a in self.agents.values() if a is not None])
        frame = {}
        for idx, vehicle in enumerate(vehicle_list):
            transform = vehicle.get_transform()
            heading = np.deg2rad(-transform.rotation.yaw)
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            state = AgentState(time=self.__timestep,
                                  position=np.array([transform.location.x, -transform.location.y]),
                                  velocity=np.array([velocity.x, -velocity.y]),
                                  acceleration=np.array([acceleration.x, -acceleration.y]),
                                  heading=heading)
            if vehicle.id in agent_id_lookup:
                agent_id = agent_id_lookup[vehicle.id]
            else:
                agent_id = vehicle.id
            frame[agent_id] = state
        return Observation(frame, self.scenario_map)

    def __wait_for_server(self):
        for i in range(10):
            try:
                self.__client = carla.Client('localhost', self.__port)
                self.__client.set_timeout(self.TIMEOUT)  # seconds
                self.__client.get_world()
                return
            except RuntimeError:
                pass
        self.__client.get_world()

    def __clear_agents(self):
        for agent_id, agent in self.agents.items():
            if agent is not None:
                self.remove_agent(agent_id)
    
    def __get_agent_waypoints(self):
        agent_waypoints = {}
        for agent_id, agent in self.agents.items():
            if agent is None:
                continue
            if agent.agent.current_macro is not None and agent.agent._pgp_drive:
                try:
                    out = np.concatenate([ma.get_trajectory().path[:-1] for ma in agent.agent.macro_actions])
                except:
                    out = agent.agent.current_macro.get_trajectory().path[:-1]
            else:
                out = []
            agent_waypoints[agent_id] = out

        # print("Got to the end!")
        # exit()
        return agent_waypoints

    @property
    def client(self) -> carla.Client:
        """ The CARLA client to the server. """
        return self.__client

    @property
    def world(self) -> carla.World:
        """The current CARLA world"""
        return self.__world

    @property
    def map(self) -> carla.Map:
        """ The CARLA map. """
        return self.__map

    @property
    def scenario_map(self) -> Map:
        """The current road layout. """
        return self.__scenario_map
    
    @property
    def dataset(self) -> Dataset:
        """The lane graph handler layout. """
        return self.__dataset

    @property
    def pgp(self):
        """The CarlaPGP module. """
        return self.__pgp

    @property
    def agents(self) -> Dict[int, CarlaAgentWrapper]:
        """ All IGP2 agents that are present or were present during the simulation."""
        return self.__agents

    @property
    def spectator(self) -> carla.Actor:
        """ The spectator camera of the world. """
        return self.__spectator

    @property
    def fps(self) -> int:
        """ Execution frequency of the simulation. """
        return self.__fps

    @property
    def timestep(self) -> int:
        """ Current timestep of the simulation"""
        return self.__timestep

    @property
    def dead_ids(self) -> List[int]:
        """ List of Agent IDs that have been used previously during the simulation"""
        return self.__dead_agent_ids

    @property
    def recording(self) -> bool:
        """ Whether we are recording the simulation. """
        return self.__record

    @property
    def recording_path(self) -> str:
        """ The save path of the recording. """
        return self.__record_path
