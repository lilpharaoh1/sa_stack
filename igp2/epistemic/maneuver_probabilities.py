"""
Data container for maneuver-level goal probabilities.

This is analogous to GoalsProbabilities but stores maneuver sequences
instead of macro-action sequences.
"""

import random
import numpy as np
import logging
from copy import copy
from operator import itemgetter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

from igp2.core.goal import Goal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.core.trajectory import VelocityTrajectory
from igp2.core.cost import Cost
from igp2.planlibrary.maneuver import Maneuver

logger = logging.getLogger(__name__)


class GoalWithType:
    """Tuple of a Goal object and a goal_type string.

    Same as in GoalsProbabilities - kept for compatibility.
    """

    def __init__(self):
        pass

    def __new__(cls, goal: Goal = None, goal_type: str = None):
        return goal, goal_type


class ManeuverProbabilities:
    """Container for maneuver-level goal probabilities.

    This is the maneuver-level equivalent of GoalsProbabilities.
    Instead of storing macro-action plans (List[MacroAction]),
    it stores maneuver plans (List[Maneuver]).

    The probability calculations and Bayesian updates work identically
    to GoalsProbabilities - only the plan representation changes.
    """

    def __init__(self, goals: List[Goal] = None, goal_types: List[List[str]] = None,
                 priors: List[float] = None):
        """Create a new ManeuverProbabilities object.

        Args:
            goals: List of goal objects
            goal_types: Optional goal type refinements
            priors: Optional prior probabilities (default: uniform)
        """
        self._goals_and_types = []
        if goal_types is None:
            for goal in goals:
                goal_and_type = GoalWithType(goal, None)
                self._goals_and_types.append(goal_and_type)
        else:
            for goal, goal_type_arr in zip(goals, goal_types):
                for goal_type in goal_type_arr:
                    goal_and_type = GoalWithType(goal, goal_type)
                    self._goals_and_types.append(goal_and_type)

        if priors is None:
            self._goals_priors = dict.fromkeys(self._goals_and_types, self.uniform_distribution())
        else:
            self._goals_priors = dict(zip(self._goals_and_types, priors))

        # Probability distributions
        self._goals_probabilities = copy(self._goals_priors)
        self._trajectories_probabilities = dict.fromkeys(self._goals_and_types, [])

        # Trajectory storage
        self._optimum_trajectory = dict.fromkeys(self._goals_and_types, None)
        self._current_trajectory = copy(self._optimum_trajectory)
        self._all_trajectories = {key: [] for key in self._goals_and_types}

        # Maneuver plan storage (this is the key difference from GoalsProbabilities)
        # Instead of List[MacroAction], we store List[Maneuver]
        self._optimum_plan = dict.fromkeys(self._goals_and_types, None)
        self._all_plans = {key: [] for key in self._goals_and_types}

        # Reward data
        self._optimum_reward = copy(self._optimum_trajectory)
        self._current_reward = copy(self._optimum_trajectory)
        self._all_rewards = {key: [] for key in self._goals_and_types}

        # Reward difference data
        self._reward_difference = copy(self._optimum_trajectory)
        self._all_reward_differences = {key: [] for key in self._goals_and_types}

        # Likelihoods
        self._likelihood = copy(self._optimum_trajectory)

    def uniform_distribution(self) -> float:
        """Generate uniform probability across all goals."""
        return float(1 / len(self._goals_and_types))

    def sample_goals(self, k: int = 1) -> List[GoalWithType]:
        """Randomly sample goals according to probability distribution."""
        goals = list(self.goals_probabilities.keys())
        weights = list(self.goals_probabilities.values())
        return random.choices(goals, weights=weights, k=k)

    def sample_trajectories_to_goal(self, goal: GoalWithType, k: int = 1) \
            -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Randomly sample trajectories to a goal.

        Returns:
            Tuple of (trajectories, maneuver_plans)
        """
        assert goal in self.trajectories_probabilities, f"Goal {goal} not in trajectories_probabilities!"
        assert goal in self.all_trajectories, f"Goal {goal} not in all_trajectories!"

        trajectories = self._all_trajectories[goal]
        if trajectories:
            weights = self._trajectories_probabilities[goal]
            sampled_trajectories = random.choices(trajectories, weights=weights, k=k)
            plans = [self.trajectory_to_plan(goal, traj) for traj in sampled_trajectories]
            return sampled_trajectories, plans
        return [], []

    def trajectory_to_plan(self, goal: GoalWithType, trajectory: VelocityTrajectory) -> List[Maneuver]:
        """Return the maneuver plan that generated a trajectory."""
        idx = self.all_trajectories[goal].index(trajectory)
        return self.all_plans[goal][idx]

    def map_prediction(self) -> Tuple[GoalWithType, VelocityTrajectory, List[Maneuver]]:
        """Return the MAP (maximum a posteriori) goal, trajectory, and maneuver plan.

        Returns:
            Tuple of (most likely goal, most likely trajectory, maneuver plan)
        """
        goal = max(self.goals_probabilities, key=self.goals_probabilities.get)

        if not self.all_trajectories[goal]:
            return goal, None, None

        trajectory, p_trajectory = max(
            zip(self.all_trajectories[goal], self.trajectories_probabilities[goal]),
            key=itemgetter(1)
        )
        plan = self.trajectory_to_plan(goal, trajectory)
        return goal, trajectory, plan

    def get_maneuver_sequence(self, goal: GoalWithType = None) -> List[str]:
        """Get the most likely maneuver sequence for a goal.

        Args:
            goal: The goal to get maneuvers for (default: most likely goal)

        Returns:
            List of maneuver type names (e.g., ['FollowLane', 'GiveWay', 'Turn'])
        """
        if goal is None:
            goal = max(self.goals_probabilities, key=self.goals_probabilities.get)

        if not self.all_plans[goal]:
            return []

        # Get most likely plan
        if self.trajectories_probabilities[goal]:
            best_idx = np.argmax(self.trajectories_probabilities[goal])
            plan = self.all_plans[goal][best_idx]
        else:
            plan = self.all_plans[goal][0] if self.all_plans[goal] else []

        return [type(m).__name__ for m in plan]

    def add_smoothing(self, alpha: float = 1., uniform_goals: bool = False):
        """Perform add-alpha smoothing on the probability distribution."""
        n_reachable = sum(map(lambda x: len(x) > 0, self.trajectories_probabilities.values()))

        for goal, trajectory_prob in self.trajectories_probabilities.items():
            trajectory_len = len(trajectory_prob)
            if trajectory_len > 0:
                if uniform_goals:
                    self.goals_probabilities[goal] = 1 / n_reachable
                else:
                    self.goals_probabilities[goal] = \
                        (self.goals_probabilities[goal] + alpha) / (1 + n_reachable * alpha)
                self.trajectories_probabilities[goal] = \
                    [(prob + alpha) / (1 + trajectory_len * alpha) for prob in trajectory_prob]

    def plot(self, scenario_map: Map = None, max_n_trajectories: int = 1,
             cost: Cost = None) -> plt.Axes:
        """Plot optimal and predicted trajectories."""

        def plot_trajectory(traj, ax_, cmap, goal_, title=""):
            plot_map(scenario_map, markings=True, ax=ax_)
            path, vel = traj.path, traj.velocity
            ax_.scatter(path[:, 0], path[:, 1], c=vel, cmap=cmap, vmin=-4, vmax=20, s=8)
            if cost is not None:
                cost.trajectory_cost(traj, goal_)
                plt.rc('axes', titlesize=8)
                t = str(cost.cost_components)
                t = t[:len(t) // 2] + "\n" + t[len(t) // 2:]
                ax_.set_title(t)
            else:
                ax_.set_title(title)

        color_map_optimal = plt.cm.get_cmap('Reds')
        color_map = plt.cm.get_cmap('Blues')

        valid_goals = [g for g, ts in self._all_trajectories.items() if len(ts) > 0]
        if not valid_goals:
            return None

        fig, axes = plt.subplots(len(valid_goals), 2, figsize=(12, 9))
        if len(valid_goals) == 1:
            axes = axes.reshape(1, -1)

        for gid, goal in enumerate(valid_goals):
            if self._optimum_trajectory[goal] is not None:
                plot_trajectory(self._optimum_trajectory[goal], axes[gid, 0],
                                color_map_optimal, goal[0])
            for tid, trajectory in enumerate(self._all_trajectories[goal][:max_n_trajectories]):
                plot_trajectory(trajectory, axes[gid, 1], color_map, goal[0])

        return axes

    def log(self, lgr: logging.Logger):
        """Log probabilities and maneuver plans."""
        for key, pg_z in self.goals_probabilities.items():
            if pg_z != 0.0:
                lgr.info(f"{key}: {np.round(pg_z, 3)}")
                for i, (plan, prob) in enumerate(
                        zip(self.all_plans[key], self.trajectories_probabilities[key])):
                    lgr.info(f"\tTrajectory {i}: {np.round(prob, 3)}")
                    # Log maneuver sequence
                    maneuver_names = [type(m).__name__ for m in plan]
                    lgr.info(f"\t\tManeuvers: {maneuver_names}")

    # Properties (same interface as GoalsProbabilities)

    @property
    def goals_probabilities(self) -> Dict[GoalWithType, float]:
        """Current goal probabilities."""
        return self._goals_probabilities

    @property
    def goals_priors(self) -> Dict[GoalWithType, float]:
        """Goal priors."""
        return self._goals_priors

    @property
    def trajectories_probabilities(self) -> Dict[GoalWithType, List[float]]:
        """Trajectory probability distributions for each goal."""
        return self._trajectories_probabilities

    @property
    def optimum_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        """Optimum trajectory from initial position to each goal."""
        return self._optimum_trajectory

    @property
    def optimum_plan(self) -> Dict[GoalWithType, List[Maneuver]]:
        """Optimum maneuver plan to each goal."""
        return self._optimum_plan

    @property
    def current_trajectory(self) -> Dict[GoalWithType, VelocityTrajectory]:
        """Current trajectory (observed + generated) to each goal."""
        return self._current_trajectory

    @property
    def all_trajectories(self) -> Dict[GoalWithType, List[VelocityTrajectory]]:
        """All generated trajectories to each goal."""
        return self._all_trajectories

    @property
    def all_plans(self) -> Dict[GoalWithType, List[List[Maneuver]]]:
        """All maneuver plans to each goal."""
        return self._all_plans

    @property
    def optimum_reward(self) -> Dict[GoalWithType, float]:
        """Optimum trajectory reward for each goal."""
        return self._optimum_reward

    @property
    def current_reward(self) -> Dict[GoalWithType, float]:
        """Current trajectory reward for each goal."""
        return self._current_reward

    @property
    def all_rewards(self) -> Dict[GoalWithType, List[float]]:
        """All trajectory rewards for each goal."""
        return self._all_rewards

    @property
    def reward_difference(self) -> Dict[GoalWithType, float]:
        """Reward difference for each goal."""
        return self._reward_difference

    @property
    def all_reward_differences(self) -> Dict[GoalWithType, List[float]]:
        """All reward differences for each goal."""
        return self._all_reward_differences

    @property
    def likelihood(self) -> Dict[GoalWithType, float]:
        """Computed likelihoods for each goal."""
        return self._likelihood

    @property
    def goals_and_types(self) -> List[GoalWithType]:
        """All goals with their types."""
        return self._goals_and_types
