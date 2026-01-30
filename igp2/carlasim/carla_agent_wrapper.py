import numpy as np

import carla
from typing import Optional
import logging

from igp2.carlasim.local_planner import LocalPlanner, RoadOption
from igp2.core.vehicle import Observation
from igp2.core.agentstate import AgentState
from igp2.core.trajectory import VelocityTrajectory
from igp2.agents.agent import Agent
from igp2.agents.keyboard_agent import KeyboardAgent
from igp2.agents.shared_autonomy_agent import SharedAutonomyAgent
from igp2.agents.belief_agent import BeliefAgent

logger = logging.getLogger(__name__)

class CarlaAgentWrapper:
    """ Wrapper class that provides a simple way to retrieve control for the attached actor. """

    def __init__(self, agent: Agent, actor: carla.Actor):
        self.__agent = agent
        self.__actor = actor
        self.__name = self.__actor.attributes["role_name"]

        self.__world = self.__actor.get_world()
        self.__map = self.__world.get_map()

        self.__local_planner = LocalPlanner(self.__actor, self.__world, self.__map,
                                            dt=1.0 / self.__agent.fps)
        self.__waypoints = []  # List of CARLA waypoints to follow
        self.__current_ma = None
        self.__last_action = None  # Raw Action from most recent next_control() call

    def __repr__(self):
        return f"Actor {self.actor_id}; Agent {self.agent_id}"

    def next_control(self, observation: Observation, prediction=None) -> Optional[carla.VehicleControl]:
        limited_observation = self.__apply_view_radius(observation)
        action = self.__agent.next_action(limited_observation, prediction)
        self.__last_action = action
        self.agent.vehicle.execute_action(action, observation.frame[self.agent_id])
        if action is None or self.__agent.done(observation) or action.target_speed is None:
            logger.debug(f"observation.frame[self.agent.agent_id].position, self.agent.goal.center = {observation.frame[self.agent.agent_id].position, self.agent.goal.center}")
            logger.debug("Returning None for next_control.")
            return None

        # For KeyboardAgent, convert action directly to CARLA control (bypass LocalPlanner)
        if isinstance(self.__agent, KeyboardAgent) \
            or isinstance(self.__agent, SharedAutonomyAgent) \
            or isinstance(self.__agent, BeliefAgent):
            return self.__action_to_control(action)

        if hasattr(self.agent, "current_macro"):
            if self.__current_ma != self.agent.current_macro:
                self.__current_ma = self.agent.current_macro
                self.__trajectory_to_waypoints(self.__current_ma.get_trajectory())
                self.__local_planner.set_global_plan(
                    self.__waypoints, stop_waypoint_creation=True, clean_queue=True)

        target_speed = action.target_speed
        if target_speed is None:
            target_speed = 0.0
        self.__local_planner.set_speed(target_speed * 3.6)
        return self.__local_planner.run_step()

    def done(self, observation: Observation) -> bool:
        """ Returns whether the wrapped agent is done. """
        return self.__agent.done(observation)

    def reset_waypoints(self):
        self.__waypoints = []

    def __apply_view_radius(self, observation: Observation):
        if hasattr(self.agent, "view_radius"):
            pos = observation.frame[self.agent_id].position
            new_frame = {aid: state for aid, state in observation.frame.items()
                         if np.linalg.norm(pos - state.position) <= self.agent.view_radius}
            return Observation(new_frame, observation.scenario_map)
        return observation

    def __action_to_control(self, action) -> carla.VehicleControl:
        """Convert an Action directly to CARLA VehicleControl (for KeyboardAgent)."""
        control = carla.VehicleControl()

        # Convert acceleration to throttle/brake
        if action.acceleration >= 0:
            control.throttle = min(1.0, action.acceleration / 5.0)  # Normalize to [0, 1]
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, -action.acceleration / 8.0)  # Normalize to [0, 1]

        # Convert steering angle to CARLA steering [-1, 1]
        # action.steer_angle is in radians, CARLA expects [-1, 1]
        # Negate because IGP2 uses y-up coordinate system while CARLA uses y-down,
        # which mirrors the steering direction
        max_steer_angle = 0.7  # radians (about 40 degrees)
        control.steer = np.clip(-action.steer_angle / max_steer_angle, -1.0, 1.0)

        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def __trajectory_to_waypoints(self, trajectory: VelocityTrajectory):
        self.__waypoints = []
        for point in trajectory.path[:-1]:
            wp = self.__map.get_waypoint(carla.Location(point[0], -point[1]))
            wp = (wp, RoadOption.LANEFOLLOW)
            assert wp is not None, f"Invalid waypoint found at {point}."
            self.__waypoints.append(wp)

    @property
    def state(self) -> AgentState:
        return self.agent.state

    @property
    def agent_id(self) -> int:
        return self.__agent.agent_id

    @property
    def actor_id(self) -> int:
        return self.__actor.id

    @property
    def actor(self) -> carla.Actor:
        return self.__actor

    @property
    def agent(self) -> Agent:
        return self.__agent

    @property
    def name(self):
        """ The role name of the wrapped Actor. """
        return self.__name

    @property
    def last_action(self):
        """The raw Action from the most recent next_control() call."""
        return self.__last_action

    @property
    def waypoints(self):
        return self.__waypoints
