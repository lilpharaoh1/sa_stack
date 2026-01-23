"""
Maneuver-level goal recognition.

This is analogous to GoalRecognition but operates at the maneuver level,
using ManeuverAStar to search over maneuver sequences instead of macro-actions.

The probabilistic framework (Bayesian inference, Boltzmann distribution) is
identical to GoalRecognition - only the level of abstraction changes.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

from shapely.geometry import Point

from igp2.core.util import find_lane_sequence, Circle
from igp2.core.trajectory import Trajectory, VelocityTrajectory
from igp2.core.velocitysmoother import VelocitySmoother
from igp2.core.cost import Cost
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, Stop
from igp2.epistemic.maneuver_astar import ManeuverAStar
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities

logger = logging.getLogger(__name__)


class ManeuverRecognition:
    """Goal recognition using maneuver-level A* search.

    This class provides the same functionality as GoalRecognition but operates
    at the maneuver level. Instead of predicting macro-action plans like
    [Continue, Exit], it predicts maneuver plans like [FollowLane, GiveWay, Turn].

    The probabilistic framework is identical:
    - Generate optimum trajectory from initial position to goal
    - Generate current trajectories from current position to goal
    - Calculate likelihood using Boltzmann distribution over reward differences
    - Update goal probabilities using Bayes' rule

    The only difference is that A* searches over maneuvers instead of macro-actions.
    """

    def __init__(self, astar: ManeuverAStar, smoother: VelocitySmoother,
                 scenario_map: Map, cost: Cost = None,
                 n_trajectories: int = 1, beta: float = 1., gamma: float = 1.,
                 reward_as_difference: bool = True):
        """Initialize maneuver-level goal recognition.

        Args:
            astar: ManeuverAStar object for generating maneuver plans
            smoother: VelocitySmoother for making trajectories realistic
            scenario_map: The road layout
            cost: Cost function for trajectory evaluation
            n_trajectories: Number of alternative trajectories per goal
            beta: Temperature for goal likelihood Boltzmann distribution
            gamma: Temperature for trajectory probability Boltzmann distribution
            reward_as_difference: Use trajectory difference for reward calculation
        """
        self._n_trajectories = n_trajectories
        self._beta = beta
        self._gamma = gamma
        self._reward_as_difference = reward_as_difference
        self._astar = astar
        self._smoother = smoother
        self._cost = Cost() if cost is None else cost
        self._scenario_map = scenario_map

    def update_goals_probabilities(self,
                                    goals_probabilities: ManeuverProbabilities,
                                    observed_trajectory: Trajectory,
                                    agent_id: int,
                                    frame_ini: Dict[int, AgentState],
                                    frame: Dict[int, AgentState],
                                    visible_region: Circle = None,
                                    open_loop: bool = True,
                                    debug: bool = False) -> ManeuverProbabilities:
        """Update goal probabilities using maneuver-level recognition.

        This is identical to GoalRecognition.update_goals_probabilities()
        except it uses ManeuverAStar and returns maneuver plans.

        Args:
            goals_probabilities: ManeuverProbabilities object to update
            observed_trajectory: The observed trajectory so far
            agent_id: Agent ID
            frame_ini: Initial frame (start of trajectory)
            frame: Current frame
            visible_region: Visible region of the map
            open_loop: Whether to generate open-loop maneuvers
            debug: Enable debug plotting

        Returns:
            Updated ManeuverProbabilities object
        """
        norm_factor = 0.
        current_lane = self._scenario_map.best_lane_at(
            frame[agent_id].position, frame[agent_id].heading
        )
        if current_lane is None:
            raise RuntimeError(
                f"Could not find best lane for agent at position {frame[agent_id].position}."
            )

        logger.info(f"Agent ID {agent_id} maneuver-level goal recognition:")

        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            try:
                goal = goal_and_type[0]
                logger.info(f"  Maneuver recognition for {goal}")

                if goal.reached(frame_ini[agent_id].position) and not isinstance(goal, StoppingGoal):
                    raise RuntimeError(f"\tAgent {agent_id} reached goal at start.")

                # Check if goal is blocked
                self._check_blocked(agent_id, current_lane, frame, goal)

                # Generate optimum trajectory from initial position
                if goals_probabilities.optimum_trajectory[goal_and_type] is None:
                    logger.debug("\tGenerating optimum maneuver trajectory")
                    trajectories, plans = self._generate_trajectory(
                        1, agent_id, frame_ini, goal,
                        state_trajectory=None, visible_region=visible_region,
                        open_loop=open_loop, debug=debug
                    )
                    goals_probabilities.optimum_trajectory[goal_and_type] = trajectories[0]
                    goals_probabilities.optimum_plan[goal_and_type] = plans[0]

                    # Log the maneuver sequence
                    maneuver_names = [type(m).__name__ for m in plans[0]]
                    logger.debug(f"\tOptimum maneuver sequence: {maneuver_names}")

                opt_trajectory = goals_probabilities.optimum_trajectory[goal_and_type]

                # Generate trajectories from current position
                logger.debug(f"\tGenerating maneuver trajectories from current time step")
                all_trajectories, all_plans = self._generate_trajectory(
                    self._n_trajectories, agent_id, frame, goal, observed_trajectory,
                    visible_region=visible_region, debug=debug
                )

                # Log maneuver sequences
                for i, plan in enumerate(all_plans):
                    maneuver_names = [type(m).__name__ for m in plan]
                    logger.debug(f"\tPlan {i} maneuvers: {maneuver_names}")

                # Calculate optimum reward
                goals_probabilities.optimum_reward[goal_and_type] = self._reward(opt_trajectory, goal)
                logger.debug(f"\tOptimum costs: {self._cost.cost_components}")

                # Process each generated trajectory
                for i, trajectory in enumerate(all_trajectories):
                    # Join observed and generated trajectories
                    trajectory.insert(observed_trajectory)

                    # Calculate rewards and likelihood
                    reward = self._reward(trajectory, goal)
                    logger.debug(f"\tT{i} costs: {self._cost.cost_components}")
                    goals_probabilities.all_rewards[goal_and_type].append(reward)

                    reward_diff = self._reward_difference(opt_trajectory, trajectory, goal)
                    goals_probabilities.all_reward_differences[goal_and_type].append(reward_diff)

                # Calculate likelihood
                likelihood = self._likelihood(opt_trajectory, all_trajectories[0], goal)

                # Calculate trajectory probabilities
                goals_probabilities.trajectories_probabilities[goal_and_type] = \
                    self._trajectory_probabilities(goals_probabilities.all_rewards[goal_and_type])

                # Store results
                goals_probabilities.all_trajectories[goal_and_type] = all_trajectories
                goals_probabilities.all_plans[goal_and_type] = all_plans
                goals_probabilities.current_trajectory[goal_and_type] = all_trajectories[0]
                goals_probabilities.reward_difference[goal_and_type] = \
                    goals_probabilities.all_reward_differences[goal_and_type][0]
                goals_probabilities.current_reward[goal_and_type] = \
                    goals_probabilities.all_rewards[goal_and_type][0]

            except RuntimeError as e:
                logger.debug(str(e))
                likelihood = 0.
                goals_probabilities.current_trajectory[goal_and_type] = None

            # Update goal probabilities using Bayes' rule
            goals_probabilities.goals_probabilities[goal_and_type] = \
                goals_probabilities.goals_priors[goal_and_type] * likelihood
            goals_probabilities.likelihood[goal_and_type] = likelihood
            norm_factor += likelihood * goals_probabilities.goals_priors[goal_and_type]

        # Normalize probabilities
        for key, prob in goals_probabilities.goals_probabilities.items():
            try:
                goals_probabilities.goals_probabilities[key] = prob / norm_factor
            except ZeroDivisionError:
                logger.debug("\tAll goals unreachable. Setting all probabilities to 0.")
                break

        return goals_probabilities

    def _generate_trajectory(self,
                              n_trajectories: int,
                              agent_id: int,
                              frame: Dict[int, AgentState],
                              goal: Goal,
                              state_trajectory: Trajectory,
                              visible_region: Circle = None,
                              n_resample: int = 5,
                              open_loop: bool = True,
                              debug: bool = False) -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Generate trajectories using maneuver-level A* search.

        Returns:
            Tuple of (trajectories, maneuver_plans)
        """
        trajectories, plans = self._astar.search(
            agent_id, frame, goal, self._scenario_map,
            n_trajectories, open_loop=open_loop,
            visible_region=visible_region, debug=debug
        )

        if len(trajectories) == 0:
            raise RuntimeError(f"\t{goal} is unreachable via maneuvers")

        # Smooth trajectories
        for trajectory in trajectories:
            if state_trajectory is None:
                trajectory.velocity[0] = frame[agent_id].speed
            else:
                trajectory.velocity[0] = state_trajectory.velocity[-1]

            self._smoother.load_trajectory(trajectory)
            new_velocities = self._smoother.split_smooth()

            # Handle high initial acceleration
            initial_acc = np.abs(new_velocities[0] - new_velocities[1])
            if len(trajectory.velocity) > n_resample and \
                    initial_acc > frame[agent_id].metadata.max_acceleration:
                new_vels = Maneuver.get_const_acceleration_vel(
                    trajectory.velocity[0],
                    trajectory.velocity[n_resample - 1],
                    trajectory.path[:n_resample]
                )
                trajectory.velocity[:n_resample] = new_vels

                self._smoother.load_trajectory(trajectory)
                new_velocities = self._smoother.split_smooth()

            trajectory.velocity = new_velocities

        return trajectories, plans

    def _check_blocked(self, agent_id: int, current_lane, frame: Dict[int, AgentState], goal: Goal):
        """Check if goal is blocked by a stopped vehicle."""
        # Check if stopped vehicle at stopping goal
        if isinstance(goal, StoppingGoal):
            for aid, state in frame.items():
                if aid != agent_id and goal.reached(state.position) and state.speed < Stop.STOP_VELOCITY:
                    raise RuntimeError(f"\t{goal} is occupied by stopped vehicle.")

        # Check if path is blocked
        goal_lane = self._scenario_map.lanes_at(goal.center)
        if not goal_lane:
            return
        goal_lane = goal_lane[0]

        lanes_to_goal = find_lane_sequence(current_lane, goal_lane, goal)
        if lanes_to_goal:
            vehicle_in_front, distance, lane_ls = Maneuver.get_vehicle_in_front(
                agent_id, frame, lanes_to_goal
            )
            goal_distance = goal.distance(Point(frame[agent_id].position))
            if vehicle_in_front is not None and \
                    (np.isclose(goal_distance, distance, atol=goal.radius) or goal_distance > distance) and \
                    np.isclose(frame[vehicle_in_front].speed, Stop.STOP_VELOCITY, atol=0.05):
                raise RuntimeError(f"\tGoal {goal} is blocked by stopped vehicle {vehicle_in_front}.")

    def _trajectory_probabilities(self, rewards: List[float]) -> List[float]:
        """Calculate trajectory probabilities using Boltzmann distribution."""
        rewards = np.array(rewards)
        num = np.exp(self._gamma * rewards - np.max(rewards))
        return list(num / np.sum(num))

    def _likelihood(self, optimum_trajectory: Trajectory,
                    current_trajectory: Trajectory, goal: Goal) -> float:
        """Calculate goal likelihood using Boltzmann distribution."""
        difference = self._reward_difference(optimum_trajectory, current_trajectory, goal)
        return float(np.clip(np.exp(self._beta * difference), 1e-305, 1e305))

    def _reward(self, trajectory: Trajectory, goal: Goal) -> float:
        """Calculate trajectory reward (negative cost)."""
        return -self._cost.trajectory_cost(trajectory, goal)

    def _reward_difference(self, optimum_trajectory: Trajectory,
                            current_trajectory: Trajectory, goal: Goal) -> float:
        """Calculate reward difference between trajectories."""
        if self._reward_as_difference:
            return -self._cost.cost_difference_resampled(
                optimum_trajectory, current_trajectory, goal
            )
        else:
            return self._reward(current_trajectory, goal) - self._reward(optimum_trajectory, goal)

    def predict_maneuvers(self, goals_probabilities: ManeuverProbabilities,
                           goal: Goal = None) -> List[str]:
        """Get the predicted maneuver sequence for a goal.

        This is a convenience method for getting the most likely maneuver
        sequence being executed by the agent.

        Args:
            goals_probabilities: Updated ManeuverProbabilities
            goal: Specific goal (default: most likely goal)

        Returns:
            List of maneuver type names (e.g., ['FollowLane', 'GiveWay', 'Turn'])
        """
        return goals_probabilities.get_maneuver_sequence(goal)
