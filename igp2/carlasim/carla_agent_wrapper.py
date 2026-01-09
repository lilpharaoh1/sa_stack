import numpy as np

import carla
from typing import Optional
import logging

from igp2.carlasim.local_planner import LocalPlanner, RoadOption
from igp2.carlasim.util import world_to_ego_batch, ego_to_world_batch
from igp2.core.vehicle import Observation, TrajectoryPrediction
from igp2.core.agentstate import AgentState
from igp2.core.trajectory import VelocityTrajectory
from igp2.agents.agent import Agent

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

    def __repr__(self):
        return f"Actor {self.actor_id}; Agent {self.agent_id}"

    def next_control(self, observation: Observation, prediction: TrajectoryPrediction = None) -> Optional[carla.VehicleControl]:
        limited_observation = self.__apply_view_radius(observation)
        action = self.__agent.next_action(limited_observation, prediction)
        self.agent.vehicle.execute_action(action, observation.frame[self.agent_id])
        if action is None or self.__agent.done(observation) or action.target_speed is None:
            logger.debug(f"observation.frame[self.agent.agent_id].position, self.agent.goal.center = {observation.frame[self.agent.agent_id].position, self.agent.goal.center}")
            logger.debug("Returning None for next_control.")
            return None

        if hasattr(self.agent, "current_macro"):
            if self.__current_ma != self.agent.current_macro:
                self.__current_ma = self.agent.current_macro
                self.__trajectory_to_waypoints(self.__current_ma.get_trajectory())
                # print("Waypoints:", self.__waypoints)
                self.__local_planner.set_global_plan(
                    self.__waypoints, stop_waypoint_creation=True, clean_queue=True)

        if self.__agent._pgp_control and prediction is not None and self.agent_id in prediction.drive:
            trajs = prediction.drive[self.agent_id]
            probs = prediction.drive_prob[self.agent_id]
            history = prediction.agent_history[self.agent_id]
            self.__pgp_to_waypoints(trajs[np.argmax(probs)], history)
            self.__local_planner.set_global_plan(
                self.__waypoints, stop_waypoint_creation=True, clean_queue=True)

        target_speed = action.target_speed
        if target_speed is None:
            logger.debug("EMRAN) Setting target_speed to 0.0 km/hr")
            logger.debug(f"Macro actions for Agent {self.agent.agent_id}) self._macro_actions, self._current_macro: {self.agent._macro_actions, self.agent.current_macro, self.agent._current_macro_id}")
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

    def __pgp_to_waypoints(self, ego_trajectory, history):
        self.__waypoints = []
        x_ego, y_ego = ego_trajectory[:, 0], ego_trajectory[:, 1]
        x_world, y_world = ego_to_world_batch(x_ego, y_ego, [history[-1][0], history[-1][1], \
                            np.arctan2(history[-1][1] - history[-2][1], history[-1][0] - history[-2][0])])
        for x, y in zip(x_world, y_world):
            wp = self.__map.get_waypoint(carla.Location(np.float64(x), -np.float64(y)))
            wp = (wp, RoadOption.LANEFOLLOW)
            assert wp is not None, f"Invalid waypoint found at {point}."
            self.__waypoints.append(wp)

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
    def waypoints(self):
        return self.__waypoints
