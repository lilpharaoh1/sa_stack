"""
Maneuver-level goal recognition.

This module provides maneuver-level goal recognition with two approaches:

1. Cost-based (original): Uses trajectory cost differences with Boltzmann distribution
   - Compares observed trajectory cost vs optimal trajectory cost
   - Good for time-optimality comparisons

2. Similarity-based (new): Uses trajectory similarity with BFS sequence generation
   - Generates candidate sequences prioritized by fewest maneuvers
   - Compares trajectory shapes/velocities directly
   - More interpretable - directly measures how well candidates match observed behavior
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

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
from igp2.epistemic.sequence_generator import ManeuverSequenceGenerator
from igp2.epistemic.trajectory_similarity import (
    path_similarity, velocity_similarity, combined_similarity,
    trajectory_overlap_similarity
)

logger = logging.getLogger(__name__)


class ManeuverRecognition:
    """Goal recognition using maneuver-level planning.

    This class provides maneuver-level goal recognition with two modes:

    1. "cost" mode (original):
       - Uses A* to find time-optimal maneuver sequences
       - Compares trajectory costs using Boltzmann distribution
       - P(goal) ∝ exp(β * (reward_current - reward_optimal))

    2. "similarity" mode (new):
       - Uses BFS to find sequences with fewest maneuvers first
       - Compares trajectory similarity (path shape, velocity profile)
       - P(goal) ∝ exp(β * similarity_score)
       - More interpretable and natural for recognition

    The similarity mode is recommended for recognition as it:
    - Prioritizes simpler plans (fewer maneuvers)
    - Directly measures how well candidates match observed behavior
    - Is more robust to timing differences
    """

    def __init__(self, astar: ManeuverAStar, smoother: VelocitySmoother,
                 scenario_map: Map, cost: Cost = None,
                 n_trajectories: int = 5, beta: float = 1., gamma: float = 1.,
                 reward_as_difference: bool = True,
                 recognition_mode: str = "similarity",
                 similarity_method: str = "average_distance"):
        """Initialize maneuver-level goal recognition.

        Args:
            astar: ManeuverAStar object for generating maneuver plans
            smoother: VelocitySmoother for making trajectories realistic
            scenario_map: The road layout
            cost: Cost function for trajectory evaluation
            n_trajectories: Number of alternative trajectories per goal
            beta: Temperature for likelihood Boltzmann distribution
            gamma: Temperature for trajectory probability Boltzmann distribution
            reward_as_difference: Use trajectory difference for reward calculation
            recognition_mode: "cost" for original approach, "similarity" for new approach
            similarity_method: Method for path similarity ("average_distance", "hausdorff", "frechet_approx")
        """
        self._n_trajectories = n_trajectories
        self._beta = beta
        self._gamma = gamma
        self._reward_as_difference = reward_as_difference
        self._astar = astar
        self._smoother = smoother
        self._cost = Cost() if cost is None else cost
        self._scenario_map = scenario_map
        self._recognition_mode = recognition_mode
        self._similarity_method = similarity_method

        # Initialize sequence generator for similarity mode
        self._sequence_generator = ManeuverSequenceGenerator(
            max_depth=6,
            max_candidates=n_trajectories,
            max_iterations=300
        )

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

        Dispatches to either cost-based or similarity-based recognition
        depending on the recognition_mode setting.

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
        if self._recognition_mode == "similarity":
            return self._update_goals_similarity(
                goals_probabilities, observed_trajectory, agent_id,
                frame_ini, frame, visible_region, debug
            )
        else:
            return self._update_goals_cost_based(
                goals_probabilities, observed_trajectory, agent_id,
                frame_ini, frame, visible_region, open_loop, debug
            )

    def _update_goals_similarity(self,
                                  goals_probabilities: ManeuverProbabilities,
                                  observed_trajectory: Trajectory,
                                  agent_id: int,
                                  frame_ini: Dict[int, AgentState],
                                  frame: Dict[int, AgentState],
                                  visible_region: Circle = None,
                                  debug: bool = False) -> ManeuverProbabilities:
        """Update goal probabilities using trajectory similarity.

        This approach:
        1. Generates candidate sequences using BFS (fewest maneuvers first)
        2. Compares each candidate's trajectory to the observed trajectory
        3. Computes likelihood based on trajectory similarity

        Args:
            goals_probabilities: ManeuverProbabilities object to update
            observed_trajectory: The observed trajectory so far
            agent_id: Agent ID
            frame_ini: Initial frame (start of trajectory)
            frame: Current frame
            visible_region: Visible region of the map
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

        logger.info(f"Agent ID {agent_id} maneuver-level goal recognition (similarity mode):")

        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            try:
                goal = goal_and_type[0]
                logger.info(f"  Maneuver recognition for {goal}")

                if goal.reached(frame_ini[agent_id].position) and not isinstance(goal, StoppingGoal):
                    raise RuntimeError(f"\tAgent {agent_id} reached goal at start.")

                # Check if goal is blocked
                self._check_blocked(agent_id, current_lane, frame, goal)

                # Generate candidate sequences using BFS (fewest maneuvers first)
                logger.debug("\tGenerating candidate maneuver sequences")
                all_trajectories, all_plans = self._sequence_generator.generate(
                    agent_id, frame, goal, self._scenario_map, visible_region
                )

                if len(all_trajectories) == 0:
                    raise RuntimeError(f"\t{goal} is unreachable via maneuvers")

                print("\n")
                print("Generated plans:")
                for plan in all_plans:
                    print(f"    {plan}")
                print("\n")

                # Smooth trajectories
                for trajectory in all_trajectories:
                    trajectory.velocity[0] = frame[agent_id].speed
                    self._smoother.load_trajectory(trajectory)
                    trajectory.velocity = self._smoother.split_smooth()

                # Log maneuver sequences
                for i, plan in enumerate(all_plans[:5]):
                    maneuver_names = [type(m).__name__ for m in plan]
                    logger.debug(f"\tCandidate {i}: {maneuver_names} ({len(plan)} maneuvers)")

                # Compute similarity scores for each candidate
                similarities = []
                for i, candidate_traj in enumerate(all_trajectories):
                    # Compare observed trajectory to the START of the candidate
                    # (since candidate is from current position to goal)
                    sim = self._compute_trajectory_similarity(
                        observed_trajectory, candidate_traj, all_plans[i]
                    )
                    similarities.append(sim)
                    logger.debug(f"\tCandidate {i} similarity: {sim:.4f}")

                # Find best matching candidate
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]

                # Compute likelihood based on similarity
                # Higher similarity = higher likelihood
                # Add bonus for fewer maneuvers (simpler plans more likely)
                maneuver_bonus = -0.1 * len(all_plans[best_idx])  # Small penalty for complexity
                adjusted_similarity = best_similarity + maneuver_bonus
                likelihood = float(np.clip(np.exp(self._beta * adjusted_similarity), 1e-305, 1e305))

                # Store results
                # Use best trajectory as optimum, include all candidates
                goals_probabilities.optimum_trajectory[goal_and_type] = all_trajectories[best_idx]
                goals_probabilities.optimum_plan[goal_and_type] = all_plans[best_idx]
                goals_probabilities.all_trajectories[goal_and_type] = all_trajectories
                goals_probabilities.all_plans[goal_and_type] = all_plans
                goals_probabilities.current_trajectory[goal_and_type] = all_trajectories[best_idx]

                # Store similarity scores as rewards for compatibility
                goals_probabilities.all_rewards[goal_and_type] = similarities
                goals_probabilities.optimum_reward[goal_and_type] = best_similarity
                goals_probabilities.current_reward[goal_and_type] = best_similarity
                goals_probabilities.reward_difference[goal_and_type] = 0.0  # Not used in similarity mode

                # Calculate trajectory probabilities from similarities
                goals_probabilities.trajectories_probabilities[goal_and_type] = \
                    self._similarity_to_probabilities(similarities)

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

    def _update_goals_cost_based(self,
                                  goals_probabilities: ManeuverProbabilities,
                                  observed_trajectory: Trajectory,
                                  agent_id: int,
                                  frame_ini: Dict[int, AgentState],
                                  frame: Dict[int, AgentState],
                                  visible_region: Circle = None,
                                  open_loop: bool = True,
                                  debug: bool = False) -> ManeuverProbabilities:
        """Update goal probabilities using cost-based comparison (original approach).

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

        logger.info(f"Agent ID {agent_id} maneuver-level goal recognition (cost mode):")

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

    def _compute_trajectory_similarity(self,
                                        observed: Trajectory,
                                        candidate: VelocityTrajectory,
                                        candidate_plan: List[Maneuver]) -> float:
        """Compute similarity between observed trajectory and a candidate.

        The similarity score considers:
        1. Path shape similarity (how close are the paths?)
        2. Velocity profile similarity (are they moving similarly?)
        3. Plan complexity (simpler plans get slight bonus)

        Args:
            observed: The observed trajectory so far
            candidate: The candidate trajectory (from current position to goal)
            candidate_plan: The maneuver sequence for the candidate

        Returns:
            Similarity score (higher = more similar)
        """
        if observed is None or len(observed.path) < 2:
            # No observed trajectory yet - use plan simplicity as proxy
            return -0.1 * len(candidate_plan)

        if candidate is None or len(candidate.path) < 2:
            return -np.inf

        # Compare paths using the configured method
        path_sim = path_similarity(observed, candidate, method=self._similarity_method)

        # Compare velocities if available
        if hasattr(observed, 'velocity') and hasattr(candidate, 'velocity'):
            vel_sim = velocity_similarity(observed, candidate)
        else:
            vel_sim = 0.0

        # Combined similarity (path weighted more heavily)
        # Path similarity is typically negative (distance), velocity is [-1, 1]
        # Normalize path similarity to similar scale
        path_sim_normalized = path_sim / 10.0  # Typical distances 0-50m -> 0 to -5

        combined = 0.7 * path_sim_normalized + 0.3 * vel_sim

        logger.debug(f"\t  Path similarity: {path_sim:.2f}, Velocity similarity: {vel_sim:.2f}, "
                    f"Combined: {combined:.4f}")

        return combined

    def _similarity_to_probabilities(self, similarities: List[float]) -> List[float]:
        """Convert similarity scores to probabilities using softmax.

        Args:
            similarities: List of similarity scores

        Returns:
            Probability distribution over trajectories
        """
        if not similarities:
            return []

        similarities = np.array(similarities)

        # Handle all-negative or all-same values
        if np.all(similarities == similarities[0]):
            return list(np.ones(len(similarities)) / len(similarities))

        # Softmax with temperature
        exp_sim = np.exp(self._gamma * (similarities - np.max(similarities)))
        return list(exp_sim / np.sum(exp_sim))

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
