"""
A* search over maneuvers (instead of macro-actions).

This provides the same functionality as igp2/recognition/astar.py but
operates at the maneuver level, searching directly over primitive
maneuvers like FollowLane, Turn, GiveWay, Stop, etc.
"""

import traceback
import numpy as np
import heapq
import logging
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple, Type

from scipy.spatial import distance_matrix

from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.core.trajectory import VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.core.util import Circle, add_offset_point
from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, Stop
from igp2.epistemic.maneuver_factory import ManeuverFactory

logger = logging.getLogger(__name__)


class ManeuverAStar:
    """A* search over maneuver sequences to goals.

    This is analogous to the AStar class in recognition module, but searches
    over maneuvers directly instead of macro-actions. This enables finer-grained
    goal recognition at the maneuver level.

    The search returns sequences of maneuvers (e.g., [FollowLane, GiveWay, Turn])
    instead of macro-action sequences (e.g., [Continue, Exit]).
    """

    NEXT_LANE_OFFSET = 0.01

    def __init__(self,
                 cost_function: Callable[[VelocityTrajectory, Goal], float] = None,
                 heuristic_function: Callable[[VelocityTrajectory, Goal], float] = None,
                 next_lane_offset: float = None,
                 max_iter: int = 150):
        """Initialize maneuver-level A* search.

        Args:
            cost_function: The cost function g (default: trajectory duration)
            heuristic_function: The heuristic function h (default: time to goal)
            next_lane_offset: Offset used to reach next lane
            max_iter: Maximum iterations before stopping search
        """
        self.next_lane_offset = ManeuverAStar.NEXT_LANE_OFFSET if next_lane_offset is None else next_lane_offset
        self.max_iter = max_iter

        self._g = ManeuverAStar.trajectory_duration if cost_function is None else cost_function
        self._h = ManeuverAStar.time_to_goal if heuristic_function is None else heuristic_function
        self._f = self.cost_function

    def search(self,
               agent_id: int,
               frame: Dict[int, AgentState],
               goal: Goal,
               scenario_map: Map,
               n_trajectories: int = 1,
               open_loop: bool = True,
               debug: bool = False,
               visible_region: Circle = None) -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Run A* search to find maneuver sequences to the goal.

        Args:
            agent_id: The agent to plan for
            frame: State of the environment
            goal: The target goal
            scenario_map: The road layout
            n_trajectories: Number of trajectories to return
            open_loop: Whether to generate open-loop maneuvers
            debug: If True, plot evolution of frontier
            visible_region: Region visible to ego vehicle

        Returns:
            Tuple of:
            - List of VelocityTrajectories ordered by increasing cost
            - List of maneuver sequences (List[Maneuver]) that generated them
        """
        solutions = []
        frontier = [(0.0, ([], frame))]  # (cost, (maneuvers, frame))
        iterations = 0

        while frontier and len(solutions) < n_trajectories and iterations < self.max_iter:
            iterations += 1
            cost, (maneuvers, current_frame) = heapq.heappop(frontier)

            # Check termination
            trajectory = self._full_trajectory(maneuvers, offset_point=False)
            if self.goal_reached(goal, trajectory) and \
                    (not isinstance(goal, StoppingGoal) or
                     trajectory.duration >= Stop.DEFAULT_STOP_DURATION - 0.01):
                if not maneuvers:
                    logger.info(f"\tAID {agent_id} at {goal} already.")
                else:
                    # Skip duplicates
                    maneuvers_key = tuple(str(m) for m in maneuvers)
                    existing_keys = [tuple(str(m) for m in sol) for sol in solutions]
                    if maneuvers_key not in existing_keys:
                        logger.info(f"\tManeuver solution found for AID {agent_id} to {goal}: {maneuvers}")
                        solutions.append(maneuvers)
                continue

            # Check if current position is valid
            if not scenario_map.roads_at(current_frame[agent_id].position):
                continue

            # Check for looping
            if maneuvers:
                if self._check_looping(trajectory, maneuvers[-1]):
                    continue

                if debug:
                    plot_map(scenario_map, midline=True, hide_road_bounds_in_junction=True)
                    for aid, state in current_frame.items():
                        plt.plot(*state.position, marker="o")
                        plt.text(*state.position, aid)
                    plt.scatter(trajectory.path[:, 0], trajectory.path[:, 1],
                                c=trajectory.velocity, cmap=plt.cm.get_cmap('Reds'),
                                vmin=-4, vmax=20, s=8)
                    plt.plot(*goal.center, marker="x")
                    plt.title(f"agent {agent_id} -> {goal}: {maneuvers}")
                    plt.show()

            # Expand: try all applicable maneuvers
            for maneuver_class in ManeuverFactory.get_applicable_maneuvers(
                    current_frame[agent_id], scenario_map, goal):
                for man_args in ManeuverFactory.get_possible_args(
                        maneuver_class, current_frame[agent_id], scenario_map, goal):
                    try:
                        # Create maneuver
                        config = ManeuverConfig(man_args)
                        new_maneuver = maneuver_class(config, agent_id=agent_id,
                                                       frame=current_frame,
                                                       scenario_map=scenario_map)

                        new_maneuvers = maneuvers + [new_maneuver]
                        new_trajectory = self._full_trajectory(new_maneuvers)

                        # Check if trajectory stays in visible region
                        if not self._check_in_region(new_trajectory, visible_region, goal):
                            continue

                        # Advance frame using maneuver's play_forward
                        new_frame = Maneuver.play_forward_maneuver(
                            agent_id, scenario_map, current_frame, new_maneuver
                        )
                        new_frame[agent_id] = new_trajectory.final_agent_state
                        new_cost = self._f(new_trajectory, goal)

                        # Avoid duplicate costs
                        if any(new_cost == c for c, _ in frontier):
                            new_cost += np.random.uniform(0.0, 1e-6)

                        heapq.heappush(frontier, (new_cost, (new_maneuvers, new_frame)))

                    except Exception as e:
                        logger.debug(f"Failed to create maneuver: {e}")
                        logger.debug(traceback.format_exc())
                        continue

        trajectories = [self._full_trajectory(mans, offset_point=False) for mans in solutions]
        return trajectories, solutions

    def cost_function(self, trajectory: VelocityTrajectory, goal: Goal) -> float:
        """Compute f = g + h for A* search."""
        return self._g(trajectory, goal) + self._h(trajectory, goal)

    @staticmethod
    def trajectory_duration(trajectory: VelocityTrajectory, goal: Goal) -> float:
        """Cost function g: duration of trajectory so far."""
        return trajectory.duration

    @staticmethod
    def time_to_goal(trajectory: VelocityTrajectory, goal: Goal) -> float:
        """Heuristic h: estimated time to reach goal."""
        return goal.distance(trajectory.path[-1]) / Maneuver.MAX_SPEED

    @staticmethod
    def goal_reached(goal: Goal, trajectory: VelocityTrajectory) -> bool:
        """Check if trajectory has reached the goal."""
        if trajectory is None:
            return False

        if goal.reached(trajectory.path[-1]):
            return True
        elif not isinstance(goal, StoppingGoal):
            return goal.passed_through_goal(trajectory)
        return False

    def _full_trajectory(self, maneuvers: List[Maneuver],
                          offset_point: bool = True) -> VelocityTrajectory:
        """Concatenate trajectories from a sequence of maneuvers.

        Args:
            maneuvers: List of maneuvers to concatenate
            offset_point: Whether to add an offset point at the end

        Returns:
            Combined VelocityTrajectory
        """
        if not maneuvers:
            return None

        path = np.empty((0, 2), float)
        velocity = np.empty((0,), float)

        for maneuver in maneuvers:
            trajectory = maneuver.trajectory
            # Remove duplicate point at junction between maneuvers
            path = np.concatenate([path[:-1] if len(path) > 0 else path,
                                   trajectory.path], axis=0)
            velocity = np.concatenate([velocity[:-1] if len(velocity) > 0 else velocity,
                                        trajectory.velocity])

        full_trajectory = VelocityTrajectory(path, velocity)
        if offset_point:
            add_offset_point(full_trajectory, self.next_lane_offset)

        return full_trajectory

    def _check_looping(self, trajectory: VelocityTrajectory,
                        final_maneuver: Maneuver) -> bool:
        """Check if the final maneuver brought us back to a visited location."""
        final_path = final_maneuver.trajectory.path[::-1]
        previous_path = trajectory.path[:-len(final_path)]
        if len(previous_path) == 0:
            return False

        ds = distance_matrix(previous_path, final_path)
        close_points = np.sum(np.isclose(ds, 0.0, atol=2 * Maneuver.POINT_SPACING), axis=1)
        return np.count_nonzero(close_points) > 2 / Maneuver.POINT_SPACING

    def _check_in_region(self, trajectory: VelocityTrajectory,
                          visible_region: Circle, goal: Goal = None) -> bool:
        """Check if trajectory stays within visible region."""
        if visible_region is None:
            return True

        # If goal is outside visible region, allow trajectory to leave
        if goal is not None and not visible_region.contains(goal.center):
            return True

        dists = np.linalg.norm(trajectory.path[:-1] - visible_region.center, axis=1)
        in_region = dists <= visible_region.radius + 1  # 1m tolerance

        if True in in_region:
            first_in_idx = np.nonzero(in_region)[0][0]
            in_region = in_region[first_in_idx:]
            return np.all(in_region)

        return True
