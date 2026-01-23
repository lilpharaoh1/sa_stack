"""
Factory for creating and configuring maneuvers for A* search.

This provides similar functionality to MacroActionFactory but operates
at the maneuver level, returning applicable maneuver types and their
possible argument variations.
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

        # Turn is applicable when approaching/in a junction
        if Turn.applicable(agent_state, scenario_map):
            maneuvers.append(Turn)

        # GiveWay is applicable when approaching a junction
        if GiveWay.applicable(agent_state, scenario_map):
            maneuvers.append(GiveWay)

        # Stop is applicable when not in a junction
        if Stop.applicable(agent_state, scenario_map):
            # Only include Stop if we have a StoppingGoal or if explicitly needed
            if isinstance(goal, StoppingGoal):
                maneuvers.append(Stop)

        return maneuvers

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
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        # Check if approaching junction
        successor_lanes = current_lane.link.successor if current_lane.link.successor else []
        junction_lanes = [l for l in successor_lanes if l.parent_road.junction is not None]

        if not junction_lanes:
            return []

        args_list = []
        for junction_lane in junction_lanes:
            # Find where this junction lane leads
            junction_successor = junction_lane.link.successor
            if junction_successor:
                endpoint = np.array(junction_successor[0].midline.coords[-1])
            else:
                endpoint = np.array(junction_lane.midline.coords[-1])

            args_list.append({
                "type": "turn",
                "junction_road_id": junction_lane.parent_road.id,
                "junction_lane_id": junction_lane.id,
                "termination_point": endpoint
            })

        return args_list

    @classmethod
    def _get_give_way_args(cls, agent_state: AgentState, scenario_map: Map,
                            goal: Goal = None) -> List[Dict]:
        """Get GiveWay parameter variations.

        GiveWay is typically followed by Turn, so we return args for each
        possible turn direction at the junction.
        """
        current_lane = scenario_map.best_lane_at(agent_state.position, agent_state.heading)
        if current_lane is None:
            return []

        successor_lanes = current_lane.link.successor if current_lane.link.successor else []
        junction_lanes = [l for l in successor_lanes if l.parent_road.junction is not None]

        if not junction_lanes:
            return []

        args_list = []
        for junction_lane in junction_lanes:
            junction_successor = junction_lane.link.successor
            if junction_successor:
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
    def _get_stop_args(cls, agent_state: AgentState, scenario_map: Map,
                        goal: Goal = None) -> List[Dict]:
        """Get Stop parameter variations."""
        if isinstance(goal, StoppingGoal):
            return [{
                "type": "stop",
                "termination_point": goal.center,
                "stop_duration": Stop.DEFAULT_STOP_DURATION
            }]

        # Otherwise, stop at current position
        direction = np.array([np.cos(agent_state.heading), np.sin(agent_state.heading)])
        stop_point = agent_state.position + direction * 2.0  # 2m ahead
        return [{
            "type": "stop",
            "termination_point": stop_point,
            "stop_duration": Stop.DEFAULT_STOP_DURATION
        }]
