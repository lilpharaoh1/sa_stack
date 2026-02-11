import abc
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from shapely.geometry import LineString, Point, Polygon

from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, FollowLane, Turn, \
    GiveWay, SwitchLaneLeft, SwitchLaneRight, Stop, TrajectoryManeuver
from igp2.planlibrary.controller import PIDController, AdaptiveCruiseControl
from igp2.core.agentstate import AgentState
from igp2.core.vehicle import Observation, Action
from igp2.core.util import Box
from igp2.opendrive import Map

logger = logging.getLogger(__name__)


class ClosedLoopManeuver(Maneuver, abc.ABC):
    """ Defines a maneuver in which sensor feedback is used """

    def next_action(self, observation: Observation) -> Action:
        """ Selects the next action for the vehicle to take

        Args:
            observation: current environment Observation

        Returns:
            Action that the vehicle should take
        """
        raise NotImplementedError

    def done(self, observation: Observation) -> bool:
        """ Checks if the maneuver is finished

        Args:
            observation: current environment Observation


        Returns:
            Bool indicating whether the maneuver is completed
        """
        raise NotImplementedError

    def reset(self):
        """ Reset the internal state of the macro action (if any). """
        raise NotImplementedError


class WaypointManeuver(ClosedLoopManeuver, abc.ABC):
    WAYPOINT_MARGIN = 1
    COMPLETION_MARGIN = 0.5
    LATERAL_ARGS = {'K_P': 1.0, 'K_I': 0.2, 'K_D': 0.05}
    LONGITUDINAL_ARGS = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0}
    ACC_ARGS = {'a_a': 5, 'b_a': 5, 'delta': 4., 's_0': 2., 'T_a': 1.5}

    # Collision avoidance parameters
    COLLISION_CHECK_ENABLED = True  # Enable/disable cross-traffic collision checking
    COLLISION_HORIZON = 3.0  # seconds to look ahead for collisions
    COLLISION_SAFETY_MARGIN = 2.0  # meters of extra safety distance
    PROXIMITY_SLOWDOWN_DIST = 15.0  # meters; start slowing if vehicle within this distance
    PROXIMITY_STOP_DIST = 5.0  # meters; stop if vehicle within this distance on collision course

    def __init__(self,
                 config: ManeuverConfig,
                 agent_id: int,
                 frame: Dict[int, AgentState],
                 scenario_map: Map):
        super().__init__(config, agent_id, frame, scenario_map)
        self._controller = PIDController(1 / self.config.fps, self.LATERAL_ARGS, self.LONGITUDINAL_ARGS)
        self._acc = AdaptiveCruiseControl(1 / self.config.fps, **self.ACC_ARGS)
        self._open_loop = False  # When True, skip reactive behaviors (ACC, collision avoidance, yielding)
        # Scale waypoint margin so the temporal look-ahead stays consistent
        # across frame rates. At 20 fps with margin=1.0m the PID steers
        # toward a point ~4 frames ahead; keep that ratio at other rates.
        self._waypoint_margin = self.WAYPOINT_MARGIN * (20.0 / self.config.fps)

    def get_target_waypoint(self, state: AgentState):
        """ Get the index of the target waypoint in the reference trajectory"""
        dist = np.linalg.norm(self.trajectory.path - state.position, axis=1)
        closest_idx = np.argmin(dist)
        margin = self._waypoint_margin
        if dist[-1] < margin:
            target_wp_idx = len(self.trajectory.path) - 1
        else:
            far_waypoints_dist = dist[closest_idx:]
            target_wp_idx = closest_idx + np.argmax(far_waypoints_dist >= margin)
        return target_wp_idx, closest_idx

    def next_action(self, observation: Observation) -> Action:
        target_wp_idx, closest_idx = self.get_target_waypoint(observation.frame[self.agent_id])
        target_waypoint = self.trajectory.path[target_wp_idx]
        target_velocity = max(Maneuver.MIN_SPEED, self.trajectory.velocity[closest_idx])
        return self._get_action(target_waypoint, target_velocity, observation)

    def _get_action(self, target_waypoint: np.ndarray, target_velocity: float, observation: Observation):
        velocity_error = self._get_acceleration(target_velocity, observation.frame)
        heading_error = self._get_steering(target_waypoint, observation.frame)

        acceleration, steering = self._controller.next_action(velocity_error, heading_error)

        action = Action(acceleration, steering, target_velocity)
        return action

    def _get_steering(self, target_waypoint: np.ndarray, frame: Dict[int, AgentState]) -> float:
        state = frame[self.agent_id]
        target_direction = target_waypoint - state.position
        waypoint_heading = np.arctan2(target_direction[1], target_direction[0])
        if np.all(target_waypoint == self.trajectory.path[-1]):
            waypoint_heading = self.trajectory.heading[-1]
        heading_error = np.diff(np.unwrap([state.heading, waypoint_heading]))[0]
        return heading_error

    def _get_acceleration(self, target_velocity: float, frame: Dict[int, AgentState]):
        state = frame[self.agent_id]
        acceleration = target_velocity - state.speed

        # In open-loop mode, skip all traffic-reactive behavior
        if self._open_loop:
            return acceleration

        # Original lane-based vehicle-in-front check
        vehicle_in_front, dist, _ = self.get_vehicle_in_front(self.agent_id, frame, self.lane_sequence)
        if vehicle_in_front is not None:
            in_front_speed = frame[vehicle_in_front].speed
            gap = dist - state.metadata.length
            acc_acceleration = self._acc.get_acceleration(self.MAX_SPEED, state.speed, in_front_speed, gap)
            acceleration = min(acceleration, acc_acceleration)

        # Additional cross-traffic and proximity collision checking
        if self.COLLISION_CHECK_ENABLED:
            collision_accel = self._get_collision_avoidance_acceleration(state, frame)
            acceleration = min(acceleration, collision_accel)

        return acceleration

    def _get_collision_avoidance_acceleration(self, state: AgentState, frame: Dict[int, AgentState]) -> float:
        """Check for potential collisions with all other vehicles and return appropriate acceleration.

        This method checks:
        1. Trajectory intersection - if another vehicle's projected path crosses ours
        2. Proximity - if another vehicle is dangerously close regardless of lane

        Returns:
            Float acceleration value (negative means braking)
        """
        acceleration = float('inf')  # Start with no constraint

        ego_pos = state.position
        ego_speed = state.speed

        # Use max_acceleration as max deceleration (typically same magnitude)
        max_decel = state.metadata.max_acceleration if state.metadata.max_acceleration else 5.0

        # Get ego's future trajectory points for collision checking
        ego_future_path = self._get_future_trajectory_segment(state)

        for other_id, other_state in frame.items():
            if other_id == self.agent_id:
                continue

            other_pos = other_state.position

            # Calculate relative position and distance
            rel_pos = other_pos - ego_pos
            distance = np.linalg.norm(rel_pos)

            # Skip if too far away to matter
            if distance > self.PROXIMITY_SLOWDOWN_DIST * 2:
                continue

            # Check 1: Trajectory intersection
            collision_imminent, time_to_collision = self._check_trajectory_collision(
                state, other_state, ego_future_path)

            if collision_imminent:
                # Calculate required deceleration based on time to collision
                if time_to_collision < 0.5:
                    # Emergency stop
                    acceleration = min(acceleration, -max_decel)
                elif time_to_collision < self.COLLISION_HORIZON:
                    # Gradual slowdown - aim to stop before collision point
                    required_decel = -(ego_speed ** 2) / (2 * max(0.1, distance - self.COLLISION_SAFETY_MARGIN))
                    acceleration = min(acceleration, max(required_decel, -max_decel))

            # Check 2: Proximity-based check for very close vehicles
            if distance < self.PROXIMITY_STOP_DIST:
                # Check if we're on a collision course (heading toward each other)
                if self._on_collision_course(state, other_state):
                    # Apply emergency braking
                    acceleration = min(acceleration, -max_decel)
            elif distance < self.PROXIMITY_SLOWDOWN_DIST:
                # Check if approaching each other
                if self._on_collision_course(state, other_state):
                    # Apply proportional slowdown based on distance
                    slowdown_factor = 1.0 - (distance - self.PROXIMITY_STOP_DIST) / (
                            self.PROXIMITY_SLOWDOWN_DIST - self.PROXIMITY_STOP_DIST)
                    target_speed = ego_speed * (1.0 - slowdown_factor * 0.5)
                    required_accel = target_speed - ego_speed
                    acceleration = min(acceleration, required_accel)

        return acceleration  # Return inf when no constraint; min() in caller handles it correctly

    def _get_future_trajectory_segment(self, state: AgentState) -> np.ndarray:
        """Get the segment of trajectory ahead of current position for collision checking."""
        if self.trajectory is None or len(self.trajectory.path) == 0:
            # If no trajectory, project forward based on current heading
            future_dist = state.speed * self.COLLISION_HORIZON
            future_point = state.position + future_dist * np.array([np.cos(state.heading), np.sin(state.heading)])
            return np.array([state.position, future_point])

        # Find current position on trajectory and get points ahead
        dist_to_path = np.linalg.norm(self.trajectory.path - state.position, axis=1)
        current_idx = np.argmin(dist_to_path)

        # Get points for the next COLLISION_HORIZON seconds
        future_dist = state.speed * self.COLLISION_HORIZON
        cumulative_dist = 0
        end_idx = current_idx

        for i in range(current_idx, len(self.trajectory.path) - 1):
            segment_dist = np.linalg.norm(self.trajectory.path[i + 1] - self.trajectory.path[i])
            cumulative_dist += segment_dist
            end_idx = i + 1
            if cumulative_dist >= future_dist:
                break

        return self.trajectory.path[current_idx:end_idx + 1]

    def _check_trajectory_collision(self, ego_state: AgentState, other_state: AgentState,
                                     ego_future_path: np.ndarray) -> Tuple[bool, float]:
        """Check if another vehicle's trajectory will intersect with ego's trajectory.

        Returns:
            Tuple of (collision_imminent, time_to_collision)
        """
        if len(ego_future_path) < 2:
            return False, float('inf')

        # Project other vehicle's future position
        other_future_dist = other_state.speed * self.COLLISION_HORIZON
        other_direction = np.array([np.cos(other_state.heading), np.sin(other_state.heading)])
        other_future_pos = other_state.position + other_future_dist * other_direction

        # Create linestrings for both trajectories
        ego_line = LineString(ego_future_path)
        other_line = LineString([other_state.position, other_future_pos])

        # Check for intersection
        if ego_line.intersects(other_line):
            intersection = ego_line.intersection(other_line)
            if not intersection.is_empty:
                # Calculate time to intersection for both vehicles
                if hasattr(intersection, 'x'):
                    intersection_point = np.array([intersection.x, intersection.y])
                else:
                    # Multiple intersection points - take the first
                    intersection_point = np.array(intersection.geoms[0].coords[0])

                ego_dist_to_intersection = np.linalg.norm(intersection_point - ego_state.position)
                other_dist_to_intersection = np.linalg.norm(intersection_point - other_state.position)

                ego_time = ego_dist_to_intersection / max(0.1, ego_state.speed)
                other_time = other_dist_to_intersection / max(0.1, other_state.speed)

                # Check if both vehicles will reach intersection at similar times
                time_diff = abs(ego_time - other_time)
                if time_diff < 2.0:  # Within 2 seconds of each other
                    return True, min(ego_time, other_time)

        # Also check for close proximity at future positions
        ego_future_pos = ego_future_path[-1] if len(ego_future_path) > 0 else ego_state.position
        future_distance = np.linalg.norm(ego_future_pos - other_future_pos)
        vehicle_size = ego_state.metadata.length + other_state.metadata.length

        if future_distance < vehicle_size + self.COLLISION_SAFETY_MARGIN:
            time_to_close = np.linalg.norm(other_state.position - ego_state.position) / max(0.1, ego_state.speed)
            return True, time_to_close

        return False, float('inf')

    def _on_collision_course(self, ego_state: AgentState, other_state: AgentState) -> bool:
        """Check if two vehicles are on a collision course based on their velocities."""
        # Vector from ego to other
        rel_pos = other_state.position - ego_state.position
        distance = np.linalg.norm(rel_pos)

        if distance < 0.1:
            return True  # Already colliding

        # Relative velocity (other relative to ego)
        rel_vel = other_state.velocity - ego_state.velocity

        # Check if closing distance
        closing_speed = -np.dot(rel_pos, rel_vel) / distance

        if closing_speed <= 0:
            return False  # Moving apart

        # Time to closest approach
        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq < 0.01:
            return False  # Relative speed too low

        time_to_closest = -np.dot(rel_pos, rel_vel) / rel_speed_sq

        if time_to_closest < 0 or time_to_closest > self.COLLISION_HORIZON:
            return False  # Closest approach in past or too far in future

        # Position at closest approach
        ego_future = ego_state.position + ego_state.velocity * time_to_closest
        other_future = other_state.position + other_state.velocity * time_to_closest

        closest_distance = np.linalg.norm(other_future - ego_future)
        vehicle_size = ego_state.metadata.length + other_state.metadata.length

        return closest_distance < vehicle_size + self.COLLISION_SAFETY_MARGIN

    def done(self, observation: Observation) -> bool:
        state = observation.frame[self.agent_id]
        ls = LineString(self.trajectory.path)
        p = Point(state.position)
        dist_along = ls.project(p)
        dist_from_end = np.linalg.norm(state.position - self.trajectory.path[-1])
        # We want the vehicle to enter the next lane, so we are not done until we have not passed the midline
        ret = dist_along >= ls.length and dist_from_end > self.COMPLETION_MARGIN
        return ret

    def reset(self):
        return


class FollowLaneCL(FollowLane, WaypointManeuver):
    """ Closed loop follow lane maneuver """
    pass


class TurnCL(Turn, WaypointManeuver):
    """ Closed loop turn maneuver.

    Collision avoidance is disabled for turns because once committed to a turn
    in a junction, the vehicle should continue through rather than stopping.
    Yielding should happen in the GiveWay maneuver BEFORE entering the junction.
    """
    COLLISION_CHECK_ENABLED = False # EMRAN TODO: Problem really lies with _check_trajectory_collision predicting imminent collisions when they aren't so imminent (e.x. the ego vehicle can or has pass the oncoming traffic in time). Maybe predictions are made with the assumption that the other agents are closed loop?


class SwitchLaneLeftCL(SwitchLaneLeft, WaypointManeuver):
    """ Closed loop switch lane left maneuver """
    pass


class SwitchLaneRightCL(SwitchLaneRight, WaypointManeuver):
    """ Closed loop switch lane right maneuver """
    pass


class TrajectoryManeuverCL(TrajectoryManeuver, WaypointManeuver):
    """ Closed loop maneuver that follows a pre-defined trajectory """
    pass


class GiveWayCL(GiveWay, WaypointManeuver):
    """ Closed loop give way maneuver """

    def __stop_required(self, observation: Observation, target_wp_idx: int):
        ego_time_to_junction = self.trajectory.times[-1] - self.trajectory.times[target_wp_idx]
        times_to_junction = self._get_times_to_junction(
            observation.frame, observation.scenario_map, ego_time_to_junction)
        time_until_clear = self._get_time_until_clear(ego_time_to_junction, times_to_junction)
        blocked_time = self._get_blocking_vehicle(observation.frame, observation.scenario_map)
        return max(time_until_clear, blocked_time) > 0

    def next_action(self, observation: Observation) -> Action:
        # In open-loop mode, skip junction-stopping logic entirely
        if self._open_loop:
            return WaypointManeuver.next_action(self, observation)

        state = observation.frame[self.agent_id]
        target_wp_idx, closest_idx = self.get_target_waypoint(state)
        target_waypoint = self.trajectory.path[target_wp_idx]
        dist_to_junction = np.linalg.norm(self.trajectory.path[-1] - state.position)
        # Based on d = v^2 / (2 * mu * g), with mu=0.75 which corresponds to a slightly damp road friction coefficient
        stopping_distance = state.speed ** 2 / (2 * 0.75 * 9.8) + state.metadata.length / 2
        close_to_junction_entry = dist_to_junction < stopping_distance

        target_velocity = max(Maneuver.MIN_SPEED, self.trajectory.velocity[target_wp_idx])
        if close_to_junction_entry and \
                self.config.stop and \
                self.__stop_required(observation, target_wp_idx):
            target_velocity = 0
        return self._get_action(target_waypoint, target_velocity, observation)


class StopCL(Stop, WaypointManeuver):

    def __init__(self,
                 config: ManeuverConfig,
                 agent_id: int,
                 frame: Dict[int, AgentState],
                 scenario_map: Map):
        self.__stop_duration = 0
        super(StopCL, self).__init__(config, agent_id, frame, scenario_map)

    def next_action(self, observation: Observation) -> Action:
        # In open-loop mode, follow the trajectory velocity profile
        # (already decelerates) without forcing a stop hold
        if self._open_loop:
            return WaypointManeuver.next_action(self, observation)

        state = observation.frame[self.agent_id]
        target_wp_idx, closest_idx = self.get_target_waypoint(state)
        target_waypoint = self.trajectory.path[target_wp_idx]
        target_velocity = max(Maneuver.MIN_SPEED, self.trajectory.velocity[target_wp_idx])

        distance_to_stop = np.linalg.norm(self.trajectory.path[-1] - state.position)
        stopping_distance = state.speed ** 2 / (2 * 0.75 * 9.8) + state.metadata.length / 2
        if distance_to_stop < stopping_distance:
            self.__stop_duration += 1
            target_velocity = Stop.STOP_VELOCITY
        return self._get_action(target_waypoint, target_velocity, observation)

    def done(self, observation: Observation) -> bool:
        # In open-loop mode, complete when reaching end of trajectory
        # instead of waiting for stop_duration to elapse
        if self._open_loop:
            return WaypointManeuver.done(self, observation)
        return self.__stop_duration >= self.config.stop_duration * self.config.fps

    def reset(self):
        self.__stop_duration = 0


class CLManeuverFactory:
    """ Used to register and create closed-loop maneuvers. """

    maneuver_types = {"follow-lane": FollowLaneCL,
                      "switch-left": SwitchLaneLeftCL,
                      "switch-right": SwitchLaneRightCL,
                      "turn": TurnCL,
                      "give-way": GiveWayCL,
                      "stop": StopCL,
                      "trajectory": TrajectoryManeuverCL}

    @classmethod
    def create(cls, config: ManeuverConfig, agent_id: int, frame: Dict[int, AgentState], scenario_map: Map):
        """ Create a new closed-loop maneuver in the given state of the environment with the given configuration.

        Args:
            config: The maneuver configuration file.
            agent_id: The agent for whom the maneuver is created.
            frame: The state of all observable agents in the environment.
            scenario_map: The road layout.
        """
        assert config.type in cls.maneuver_types, f"Unregistered maneuver {config.type}. " \
                                                  f"Register with CLManeuverFactory.register_new_maneuver."
        config.config_dict["adjust_swerving"] = False
        return cls.maneuver_types[config.type](config, agent_id, frame, scenario_map)

    @classmethod
    def register_new_maneuver(cls, type_str: str, type_man: type(ClosedLoopManeuver)):
        """ Register a new closed-loop maneuver to the list of available maneuvers

        Args:
            type_str: The type name of the maneuver to register.
            type_man: The type of the maneuver to register.
        """
        assert isinstance(type_man, type(ClosedLoopManeuver)), f"Given type_man is not a MacroAction"
        assert type_str not in cls.maneuver_types, f"CLManeuver {type_str} already registered."

        cls.maneuver_types[type_str] = type_man
        logger.info(f"Register closed-loop maneuver {type_str} as {type_man}")
