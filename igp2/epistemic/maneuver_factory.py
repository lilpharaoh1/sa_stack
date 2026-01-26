"""
Factory for creating and configuring maneuvers for A* search.

This provides similar functionality to MacroActionFactory but operates
at the maneuver level, returning applicable maneuver types and their
possible argument variations.

Detection Flexibility:
- Turn: Detectable when approaching OR inside a junction
- GiveWay: Detectable when slowing down near a junction (velocity < threshold)
- Stop: Detectable when at low velocity (approaching stop or stopped)
"""

import logging
import numpy as np
from typing import Dict, List, Type, Optional

from shapely.geometry import Point

from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.opendrive.map import Map
from igp2.opendrive.elements.road_lanes import Lane
from igp2.planlibrary.maneuver import (
    Maneuver, ManeuverConfig,
    FollowLane, SwitchLaneLeft, SwitchLaneRight, Turn, GiveWay, Stop
)

logger = logging.getLogger(__name__)

# Thresholds for flexible maneuver detection
GIVE_WAY_APPROACH_DISTANCE = 30.0  # meters - distance to junction for GiveWay detection
GIVE_WAY_VELOCITY_THRESHOLD = 8.0  # m/s - if slower than this near junction, might be giving way
STOP_VELOCITY_THRESHOLD = 2.0  # m/s - if slower than this, might be stopping
JUNCTION_APPROACH_DISTANCE = 5.0  # meters - distance to junction for Turn detection


class ManeuverFactory:
    """Factory for getting applicable maneuvers and their parameter variations.

    This is analogous to MacroActionFactory but works at the maneuver level,
    enabling A* search directly over maneuvers instead of macro-actions.
    """

    maneuver_types = {
        "FollowLane": FollowLane,
        "SwitchLaneLeft": SwitchLaneLeft,
        "SwitchLaneRight": SwitchLaneRight,
        "Turn": Turn,
        "GiveWay": GiveWay,
        "Stop": Stop,
    }

    @classmethod
    def create(cls, config: ManeuverConfig, agent_id: int,
               frame: Dict[int, AgentState], scenario_map: Map) -> Maneuver:
        """Create a new maneuver with the given configuration.

        Args:
            config: The maneuver configuration
            agent_id: The agent for whom the maneuver is created
            frame: The state of all observable agents
            scenario_map: The road layout

        Returns:
            A new Maneuver instance
        """
        assert config.type in cls.maneuver_types, \
            f"Unregistered maneuver {config.type}. Register with ManeuverFactory.register_maneuver."
        return cls.maneuver_types[config.type](config, agent_id, frame, scenario_map)

    @classmethod
    def register_maneuver(cls, type_str: str, type_maneuver: type):
        """Register a new maneuver type.

        Args:
            type_str: The type name of the maneuver
            type_maneuver: The maneuver class
        """
        assert type_str not in cls.maneuver_types, f"Maneuver {type_str} already registered."
        cls.maneuver_types[type_str] = type_maneuver
        logger.info(f"Registered maneuver {type_str} as {type_maneuver}")

    @classmethod
    def get_applicable_maneuvers(cls, agent_state: AgentState, scenario_map: Map,
                                  goal: Goal = None) -> List[Type[Maneuver]]:
        """Return all applicable maneuver types for the given state.

        This method uses flexible detection thresholds to better capture
        maneuvers like GiveWay and Stop based on observed behavior.

        Args:
            agent_state: Current state of the agent
            scenario_map: The road layout
            goal: Optional goal to consider

        Returns:
            List of applicable maneuver types
        """
        maneuvers = []

        # FollowLane is usually applicable when on a drivable lane
        if FollowLane.applicable(agent_state, scenario_map):
            maneuvers.append(FollowLane)

        # Lane changes
        if SwitchLaneLeft.applicable(agent_state, scenario_map):
            maneuvers.append(SwitchLaneLeft)

        if SwitchLaneRight.applicable(agent_state, scenario_map):
            maneuvers.append(SwitchLaneRight)

        # Turn is only applicable when inside a junction
        if Turn.applicable(agent_state, scenario_map):
            maneuvers.append(Turn)

        # GiveWay is applicable when approaching a junction
        # Use flexible detection: also consider velocity-based detection
        # print(f"\nGiveWay Applicability: {GiveWay.applicable(agent_state, scenario_map)} {cls._is_giving_way(agent_state, scenario_map)}")
        if GiveWay.applicable(agent_state, scenario_map):
            maneuvers.append(GiveWay)
        elif cls._is_giving_way(agent_state, scenario_map):
            # Flexible detection: slowing down near a junction
            maneuvers.append(GiveWay)

        # Stop is applicable when not in a junction
        # Use flexible detection: also consider low velocity states
        if Stop.applicable(agent_state, scenario_map):
            # Include Stop if we have a StoppingGoal or if agent is nearly stopped
            if isinstance(goal, StoppingGoal) or agent_state.speed < STOP_VELOCITY_THRESHOLD:
                maneuvers.append(Stop)

        return maneuvers

    @classmethod
    def _approaching_junction(cls, agent_state: AgentState, scenario_map: Map,
                               max_distance: float) -> bool:
        """Check if agent is approaching a junction within max_distance.

        This is more lenient than Turn.applicable() which requires being
        right at the junction entrance.
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return False

        # Check if currently in junction
        if current_lane.parent_road.junction is not None:
            return True

        # Check distance to lane end (junction entrance)
        lane_midline = current_lane.midline
        current_point = Point(agent_state.position)
        current_lon = lane_midline.project(current_point)
        distance_to_end = lane_midline.length - current_lon

        if distance_to_end > max_distance:
            return False

        # Check if successor lanes are in a junction
        next_lanes = current_lane.link.successor
        if next_lanes is not None:
            return any(ll.parent_road.junction is not None for ll in next_lanes)

        return False

    @classmethod
    def _is_giving_way(cls, agent_state: AgentState, scenario_map: Map) -> bool:
        """Check if agent appears to be giving way (slowing near junction).

        Flexible detection based on:
        - Approaching a junction within GIVE_WAY_APPROACH_DISTANCE
        - Moving slower than GIVE_WAY_VELOCITY_THRESHOLD
        """
        # Must be moving slowly
        if agent_state.speed > GIVE_WAY_VELOCITY_THRESHOLD:
            return False

        # Must be approaching a junction
        return cls._approaching_junction(agent_state, scenario_map, GIVE_WAY_APPROACH_DISTANCE)

    @classmethod
    def get_possible_args(cls, maneuver_class: Type[Maneuver],
                          agent_state: AgentState, scenario_map: Map,
                          goal: Goal = None) -> List[Dict]:
        """Get parameter variations for a maneuver type.

        Each maneuver can have multiple parametrizations (e.g., different
        termination points, different junction lanes to turn into).

        Args:
            maneuver_class: The maneuver class
            agent_state: Current agent state
            scenario_map: The road layout
            goal: Optional goal to consider

        Returns:
            List of argument dictionaries, each defining one parametrization
        """
        if maneuver_class == FollowLane:
            return cls._get_follow_lane_args(agent_state, scenario_map, goal)
        elif maneuver_class == SwitchLaneLeft:
            return cls._get_switch_lane_args(agent_state, scenario_map, goal, left=True)
        elif maneuver_class == SwitchLaneRight:
            return cls._get_switch_lane_args(agent_state, scenario_map, goal, left=False)
        elif maneuver_class == Turn:
            return cls._get_turn_args(agent_state, scenario_map, goal)
        elif maneuver_class == GiveWay:
            return cls._get_give_way_args(agent_state, scenario_map, goal)
        elif maneuver_class == Stop:
            return cls._get_stop_args(agent_state, scenario_map, goal)
        else:
            return [{}]

    @classmethod
    def _get_follow_lane_args(cls, agent_state: AgentState, scenario_map: Map,
                               goal: Goal = None) -> List[Dict]:
        """Get FollowLane parameter variations.

        Returns termination points at:
        - Goal if on current lane
        - End of current lane
        - Intermediate points for long lanes
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        args_list = []

        # If goal is on current lane, terminate at goal
        if goal is not None:
            goal_point = goal.point_on_lane(current_lane)
            if goal_point is not None:
                args_list.append({
                    "type": "follow-lane",
                    "termination_point": goal_point
                })
                return args_list  # If goal is on lane, just go there

        # Otherwise, terminate at end of lane
        endpoint = np.array(current_lane.midline.coords[-1])
        args_list.append({
            "type": "follow-lane",
            "termination_point": endpoint
        })

        return args_list

    @classmethod
    def _get_switch_lane_args(cls, agent_state: AgentState, scenario_map: Map,
                               goal: Goal = None, left: bool = True) -> List[Dict]:
        """Get SwitchLane parameter variations."""
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        # Get target lane
        adjacent_lanes = scenario_map.get_adjacent_lanes(current_lane)
        target_lane = None
        for lane in adjacent_lanes:
            if left and lane.id > current_lane.id:
                target_lane = lane
                break
            elif not left and lane.id < current_lane.id:
                target_lane = lane
                break

        if target_lane is None:
            return []

        # Termination point at end of target lane
        endpoint = np.array(target_lane.midline.coords[-1])

        maneuver_type = "switch-left" if left else "switch-right"
        return [{
            "type": maneuver_type,
            "termination_point": endpoint
        }]

    @classmethod
    def _get_turn_args(cls, agent_state: AgentState, scenario_map: Map,
                        goal: Goal = None) -> List[Dict]:
        """Get Turn parameter variations.

        Returns one argument dict for each possible turn direction at the junction.
        Turn is only applicable when the agent is inside a junction.
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        # Turn is only applicable when inside a junction
        if current_lane.parent_road.junction is None:
            return []

        args_list = []


        # Current lane is a junction lane - find where it leads
        if not current_lane.link.predecessor is None and len(current_lane.link.predecessor) == 1:
            for succ_lane in current_lane.link.predecessor[0].link.successor:
                endpoint = np.array(succ_lane.midline.coords[-1])
                args_list.append({
                    "type": "turn",
                    "junction_road_id": current_lane.parent_road.id,
                    "junction_lane_id": current_lane.id,
                    "termination_point": endpoint
                })
        else:
            raise RuntimeError(f"Junction road {current_lane.parent_road.id} had "
            f"zero or more than one predecessor road.")
            # # No successor - terminate at end of junction lane
            # endpoint = np.array(current_lane.midline.coords[-1])
            # args_list.append({
            #     "type": "turn",
            #     "junction_road_id": current_lane.parent_road.id,
            #     "junction_lane_id": current_lane.id,
            #     "termination_point": endpoint
            # })

        return args_list

    @classmethod
    def _get_give_way_args(cls, agent_state: AgentState, scenario_map: Map,
                            goal: Goal = None) -> List[Dict]:
        """Get GiveWay parameter variations.

        GiveWay is typically followed by Turn, so we return args for each
        possible turn direction at the junction.

        Uses flexible detection to find junction lanes even when not
        immediately at the junction entrance.
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        # Find junction lanes - either direct successors or through lane following
        junction_lanes = cls._find_upcoming_junction_lanes(current_lane, scenario_map)

        if not junction_lanes:
            return []

        args_list = []
        for junction_lane in junction_lanes:
            junction_successor = junction_lane.link.successor
            if junction_successor:
                # Termination point is start of post-junction lane
                endpoint = np.array(junction_successor[0].midline.coords[0])
            else:
                endpoint = np.array(junction_lane.midline.coords[-1])

            args_list.append({
                "type": "give-way",
                "junction_road_id": junction_lane.parent_road.id,
                "junction_lane_id": junction_lane.id,
                "termination_point": endpoint,
                "stop": True
            })

        return args_list

    @classmethod
    def _find_upcoming_junction_lanes(cls, current_lane: Lane, scenario_map: Map,
                                       max_depth: int = 3) -> List[Lane]:
        """Find junction lanes reachable from current lane.

        Searches through lane successors up to max_depth to find junction lanes.
        This allows detection of GiveWay even when not immediately at junction.
        """
        junction_lanes = []

        # Check direct successors first
        successor_lanes = current_lane.link.successor if current_lane.link.successor else []
        for succ in successor_lanes:
            if succ.parent_road.junction is not None:
                junction_lanes.append(succ)

        if junction_lanes:
            return junction_lanes

        # If no direct junction successors, search deeper (but limit depth)
        if max_depth > 1:
            for succ in successor_lanes:
                # Don't search through junctions
                if succ.parent_road.junction is None:
                    deeper_junctions = cls._find_upcoming_junction_lanes(
                        succ, scenario_map, max_depth - 1
                    )
                    junction_lanes.extend(deeper_junctions)

        return junction_lanes

    @classmethod
    def _get_stop_args(cls, agent_state: AgentState, scenario_map: Map,
                        goal: Goal = None) -> List[Dict]:
        """Get Stop parameter variations.

        Handles:
        - Explicit StoppingGoal
        - Low velocity states (agent appears to be stopping)
        """
        args_list = []

        if isinstance(goal, StoppingGoal):
            args_list.append({
                "type": "stop",
                "termination_point": goal.center,
                "stop_duration": Stop.DEFAULT_STOP_DURATION
            })
            return args_list

        # For low velocity states, generate stop at current/nearby position
        direction = np.array([np.cos(agent_state.heading), np.sin(agent_state.heading)])

        # Stop point depends on current velocity
        if agent_state.speed < 1.0:
            # Nearly stopped - stop at current position
            stop_point = agent_state.position + direction * 0.5
        else:
            # Slowing down - stop a bit ahead (approximate stopping distance)
            stopping_distance = min(agent_state.speed * 1.0, 5.0)  # 1s of travel or 5m max
            stop_point = agent_state.position + direction * stopping_distance

        args_list.append({
            "type": "stop",
            "termination_point": stop_point,
            "stop_duration": Stop.DEFAULT_STOP_DURATION
        })

        return args_list
