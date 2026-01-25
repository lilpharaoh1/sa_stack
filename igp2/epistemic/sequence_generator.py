"""
Maneuver sequence generator using breadth-first search.

Generates candidate maneuver sequences prioritized by fewest maneuvers,
which is more interpretable than time-optimal A* search for recognition.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np

from igp2.opendrive.map import Map
from igp2.core.trajectory import VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.core.util import Circle
from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, Stop
from igp2.epistemic.maneuver_factory import ManeuverFactory

logger = logging.getLogger(__name__)


class ManeuverSequenceGenerator:
    """Generate candidate maneuver sequences using breadth-first search.

    Unlike A* which finds time-optimal paths, this generator finds sequences
    with the fewest maneuvers first. This is more natural for recognition
    since simpler plans are generally more likely.

    The generator uses BFS to explore maneuver combinations level by level:
    - Level 1: All 1-maneuver sequences that reach the goal
    - Level 2: All 2-maneuver sequences that reach the goal
    - etc.
    """

    def __init__(self,
                 max_depth: int = 6,
                 max_candidates: int = 10,
                 max_iterations: int = 500):
        """Initialize the sequence generator.

        Args:
            max_depth: Maximum number of maneuvers in a sequence
            max_candidates: Maximum number of candidate sequences to return
            max_iterations: Maximum BFS iterations before stopping
        """
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations

    def generate(self,
                 agent_id: int,
                 frame: Dict[int, AgentState],
                 goal: Goal,
                 scenario_map: Map,
                 visible_region: Circle = None) -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Generate candidate maneuver sequences to reach the goal.

        Uses BFS to find sequences with fewest maneuvers first.

        Args:
            agent_id: The agent to plan for
            frame: State of the environment
            goal: The target goal
            scenario_map: The road layout
            visible_region: Region visible to ego vehicle

        Returns:
            Tuple of:
            - List of VelocityTrajectories (ordered by number of maneuvers)
            - List of maneuver sequences that generated them
        """
        solutions = []  # List of (maneuver_sequence, trajectory)

        # BFS queue: (maneuvers, frame, depth)
        queue = deque([([], frame, 0)])
        iterations = 0
        visited_states = set()  # Track visited (position, heading) tuples to avoid loops

        while queue and len(solutions) < self.max_candidates and iterations < self.max_iterations:
            iterations += 1
            maneuvers, current_frame, depth = queue.popleft()

            print(f"\nChecking maneuvers {maneuvers} {current_frame[0].position}")

            # Check termination
            # print(f"\nbefore: {maneuvers}")
            trajectory = self._full_trajectory(maneuvers)
            # print(f"after: {trajectory}")
            if trajectory is not None and self._goal_reached(goal, trajectory):
                # Valid solution found
                solutions.append((maneuvers, trajectory))
                logger.debug(f"Found {len(maneuvers)}-maneuver sequence: "
                            f"{[type(m).__name__ for m in maneuvers]}")
                print("Exiting 1")
                continue

            # Depth limit
            if depth >= self.max_depth:
                print("Exiting 2")
                continue

            # # Check for loops using position hash
            # state = current_frame[agent_id]
            # state_key = (round(state.position[0], 1), round(state.position[1], 1),
            #             round(state.heading, 2))
            # if state_key in visited_states:
            #     print("Exiting 3")
            #     continue
            # visited_states.add(state_key)

            # Expand: try all applicable maneuvers
            for maneuver_class in ManeuverFactory.get_applicable_maneuvers(
                    current_frame[agent_id], scenario_map, goal):
                for man_args in ManeuverFactory.get_possible_args(
                        maneuver_class, current_frame[agent_id], scenario_map, goal):
                    # try:
                    # Create maneuver
                    config = ManeuverConfig(man_args)
                    new_maneuver = maneuver_class(config, agent_id=agent_id,
                                                    frame=current_frame,
                                                    scenario_map=scenario_map)

                    new_maneuvers = maneuvers + [new_maneuver]
                    new_trajectory = self._full_trajectory(new_maneuvers)

                    print(f"Adding maneuver: {new_maneuver}")

                    if new_trajectory is None:
                        print("Exiting 3")
                        continue

                    # Check if trajectory stays in visible region
                    if not self._check_in_region(new_trajectory, visible_region, goal):
                        print("Exiting 4")
                        continue

                    # Advance frame
                    new_frame = Maneuver.play_forward_maneuver(
                        agent_id, scenario_map, current_frame, new_maneuver
                    )
                    new_frame[agent_id] = new_trajectory.final_agent_state

                    print(f"prev_frame: {current_frame[0].position}")
                    print(f"new_frame: {new_frame[0].position}")
                    # print("---------------------------")

                    # Add to queue
                    queue.append((new_maneuvers, new_frame, depth + 1))

                    # except Exception as e:
                    #     logger.debug(f"Failed to create maneuver: {e}")
                    #     continue

        print("\n")

        # Sort solutions by number of maneuvers (should already be sorted due to BFS)
        solutions.sort(key=lambda x: len(x[0]))

        trajectories = [sol[1] for sol in solutions]
        plans = [sol[0] for sol in solutions]

        logger.info(f"Generated {len(solutions)} candidate sequences for agent {agent_id}")
        for i, (plan, _) in enumerate(solutions[:5]):
            logger.debug(f"  Candidate {i}: {[type(m).__name__ for m in plan]}")

        return trajectories, plans

    def generate_from_observed(self,
                                agent_id: int,
                                frame: Dict[int, AgentState],
                                goal: Goal,
                                scenario_map: Map,
                                observed_trajectory,
                                visible_region: Circle = None) -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Generate candidate sequences starting from the end of observed trajectory.

        This is used for recognition: we generate what maneuvers COULD complete
        the trajectory from the current position to the goal.

        Args:
            agent_id: The agent to plan for
            frame: Current state of the environment
            goal: The target goal
            scenario_map: The road layout
            observed_trajectory: The trajectory observed so far
            visible_region: Region visible to ego vehicle

        Returns:
            Tuple of (trajectories, maneuver_plans)
        """
        # Generate from current position
        trajectories, plans = self.generate(
            agent_id, frame, goal, scenario_map, visible_region
        )

        # Prepend observed trajectory to each candidate
        combined_trajectories = []
        for traj in trajectories:
            if observed_trajectory is not None and len(observed_trajectory.path) > 0:
                combined = self._join_trajectories(observed_trajectory, traj)
                combined_trajectories.append(combined)
            else:
                combined_trajectories.append(traj)

        return combined_trajectories, plans

    def _full_trajectory(self, maneuvers: List[Maneuver]) -> Optional[VelocityTrajectory]:
        """Concatenate trajectories from a sequence of maneuvers."""
        if not maneuvers:
            return None

        path = np.empty((0, 2), float)
        velocity = np.empty((0,), float)

        for maneuver in maneuvers:
            trajectory = maneuver.trajectory
            if trajectory is None or len(trajectory.path) == 0:
                return None

            # Remove duplicate point at junction between maneuvers
            path = np.concatenate([path[:-1] if len(path) > 0 else path,
                                   trajectory.path], axis=0)
            velocity = np.concatenate([velocity[:-1] if len(velocity) > 0 else velocity,
                                        trajectory.velocity])

        if len(path) == 0:
            return None

        return VelocityTrajectory(path, velocity)

    def _join_trajectories(self, observed: VelocityTrajectory,
                           candidate: VelocityTrajectory) -> VelocityTrajectory:
        """Join observed trajectory with candidate continuation."""
        if len(observed.path) == 0:
            return candidate
        if len(candidate.path) == 0:
            return observed

        # Concatenate, removing duplicate point at junction
        path = np.concatenate([observed.path, candidate.path[1:]], axis=0)
        velocity = np.concatenate([observed.velocity, candidate.velocity[1:]])

        return VelocityTrajectory(path, velocity)

    def _goal_reached(self, goal: Goal, trajectory: VelocityTrajectory) -> bool:
        """Check if trajectory has reached the goal."""
        if trajectory is None or len(trajectory.path) == 0:
            return False

        if goal.reached(trajectory.path[-1]):
            return True
        elif not isinstance(goal, StoppingGoal):
            return goal.passed_through_goal(trajectory)
        return False

    def _check_in_region(self, trajectory: VelocityTrajectory,
                          visible_region: Circle, goal: Goal = None) -> bool:
        """Check if trajectory stays within visible region."""
        if visible_region is None:
            return True

        # If goal is outside visible region, allow trajectory to leave
        if goal is not None and not visible_region.contains(goal.center):
            return True

        dists = np.linalg.norm(trajectory.path - visible_region.center, axis=1)
        in_region = dists <= visible_region.radius + 1  # 1m tolerance

        if True in in_region:
            first_in_idx = np.nonzero(in_region)[0][0]
            in_region = in_region[first_in_idx:]
            return np.all(in_region)

        return True
