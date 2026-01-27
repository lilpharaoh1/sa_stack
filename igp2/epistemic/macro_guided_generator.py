"""
Macro-action guided maneuver sequence generator.

This generator uses macro-action structure to guide maneuver expansion:
1. Gets applicable macro-actions at current state (Continue, Exit, ChangeLane, etc.)
2. Expands each macro-action into its maneuvers (via get_maneuvers())
3. Queues maneuvers individually for BFS exploration
4. Tracks progress within macro-actions to avoid infinite loops

This approach:
- Uses macro-action applicability logic (which is well-tested)
- Generates plans at maneuver-level granularity
- Allows for variations (with/without optional maneuver types)
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from dataclasses import dataclass
from copy import copy, deepcopy

import numpy as np

from igp2.opendrive.map import Map
from igp2.core.trajectory import VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.core.util import Circle
from igp2.planlibrary.maneuver import Maneuver, ManeuverConfig, GiveWay
from igp2.planlibrary.macro_action import (
    MacroAction, MacroActionConfig, MacroActionFactory,
    Continue, Exit, ChangeLaneLeft, ChangeLaneRight
)

logger = logging.getLogger(__name__)


@dataclass
class ManeuverStep:
    """Represents a single maneuver step in a plan."""
    maneuver: Maneuver
    macro_action_type: str  # Which macro-action this came from
    step_index: int  # Index within the macro-action's maneuver list
    total_steps: int  # Total maneuvers in the macro-action


@dataclass
class QueueEntry:
    """Entry in the BFS queue."""
    maneuver_steps: List[ManeuverStep]  # Accumulated maneuver steps
    frame: Dict[int, AgentState]  # Current frame after executing maneuvers
    depth: int  # Number of macro-actions expanded (not maneuvers)
    completed_macro_actions: List[str]  # Types of completed macro-actions


class MacroGuidedSequenceGenerator:
    """Generate maneuver sequences guided by macro-action structure.

    This generator uses macro-actions to determine what maneuvers are applicable,
    but tracks and queues at the maneuver level. This provides:
    - Robust applicability checks (from macro-action logic)
    - Fine-grained maneuver-level plans
    - Ability to generate variations (with/without optional maneuvers)

    The BFS explores:
    1. Get applicable macro-actions at current state
    2. For each, create the maneuver sequence
    3. Queue each complete maneuver sequence as a potential path
    4. Continue until goal is reached
    """

    # Maneuver types considered optional (can be removed to create variations).
    # Extend this set to add more optional types (e.g., Stop).
    OPTIONAL_MANEUVER_TYPES = {GiveWay}

    def __init__(self,
                 scenario_map: Map,
                 max_depth: int = 4,
                 max_candidates: int = 10,
                 max_iterations: int = 100,
                 generate_variations: bool = True):
        """Initialize the generator.

        Args:
            scenario_map: The road network map
            max_depth: Maximum number of macro-actions to chain
            max_candidates: Maximum number of candidate sequences to return
            max_iterations: Maximum BFS iterations
            generate_variations: Whether to generate with/without GiveWay variations
        """
        self._scenario_map = scenario_map
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.generate_variations = generate_variations

    def generate(self,
                 agent_id: int,
                 frame: Dict[int, AgentState],
                 goal: Goal,
                 visible_region: Circle = None,
                 seed_plans: List[Tuple[List[Maneuver], VelocityTrajectory]] = None
                 ) -> Tuple[List[VelocityTrajectory], List[List[Maneuver]]]:
        """Generate candidate maneuver sequences to reach the goal.

        Uses macro-action structure to guide BFS exploration at maneuver level.

        Args:
            agent_id: The agent to plan for
            frame: Current state of the environment
            goal: The target goal
            visible_region: Region visible to ego vehicle
            seed_plans: Optional list of (maneuver_list, trajectory) tuples to
                seed the BFS. Seeds that reach the goal are added directly as
                solutions; others are added to the BFS queue for expansion.

        Returns:
            Tuple of:
            - List of VelocityTrajectories
            - List of maneuver sequences (List[Maneuver])
        """
        solutions = []  # List of (maneuver_list, trajectory)
        seen_signatures = set()  # Track maneuver-type signatures to avoid duplicates

        # BFS queue
        initial_entry = QueueEntry(
            maneuver_steps=[],
            frame=frame.copy(),
            depth=0,
            completed_macro_actions=[]
        )
        queue = deque([initial_entry])

        # Add seed plans to queue/solutions
        if seed_plans:
            for maneuver_list, trajectory in seed_plans:
                steps = [
                    ManeuverStep(
                        maneuver=m,
                        macro_action_type="seed",
                        step_index=i,
                        total_steps=len(maneuver_list)
                    )
                    for i, m in enumerate(maneuver_list)
                ]
                if trajectory is not None and self._goal_reached(goal, trajectory):
                    sig = self._plan_signature(maneuver_list)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        solutions.append((maneuver_list, trajectory))
                        logger.debug(f"Seed plan added as solution: "
                                     f"{[type(m).__name__ for m in maneuver_list]}")
                else:
                    seed_entry = QueueEntry(
                        maneuver_steps=steps,
                        frame=frame.copy(),
                        depth=0,
                        completed_macro_actions=["seed"]
                    )
                    queue.appendleft(seed_entry)
                    logger.debug(f"Seed plan added to BFS queue: "
                                 f"{[type(m).__name__ for m in maneuver_list]}")

        iterations = 0
        visited = set()  # Track visited states to avoid loops

        while queue and len(solutions) < self.max_candidates and iterations < self.max_iterations:
            iterations += 1
            entry = queue.popleft()

            # Check if we've reached the goal
            trajectory = self._build_trajectory(entry.maneuver_steps)
            if trajectory is not None and self._goal_reached(goal, trajectory):
                maneuvers = [step.maneuver for step in entry.maneuver_steps]
                sig = self._plan_signature(maneuvers)
                if sig in seen_signatures:
                    logger.debug(f"Skipping duplicate solution: "
                                 f"{[type(m).__name__ for m in maneuvers]}")
                    continue
                seen_signatures.add(sig)
                solutions.append((maneuvers, trajectory))
                logger.debug(f"Found solution with {len(maneuvers)} maneuvers: "
                           f"{[type(m).__name__ for m in maneuvers]}")
                continue

            # Depth limit (on macro-actions, not maneuvers)
            if entry.depth >= self.max_depth:
                continue

            # Create state key to avoid revisiting
            state = entry.frame[agent_id]
            state_key = self._make_state_key(state, entry.completed_macro_actions)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Get applicable macro-actions
            applicable_mas = self._get_applicable_macro_actions(state)

            for ma_class in applicable_mas:
                # Get possible argument variations for this macro-action
                try:
                    possible_args = ma_class.get_possible_args(state, self._scenario_map, goal)
                except Exception as e:
                    logger.debug(f"Failed to get args for {ma_class.__name__}: {e}")
                    continue

                for ma_args in possible_args:
                    try:
                        # Create the macro-action to get its maneuvers
                        # Copy args to avoid mutating the original
                        args = ma_args.copy()
                        args['type'] = ma_class.__name__
                        args['open_loop'] = True
                        config = MacroActionConfig(args)
                        macro_action = MacroActionFactory.create(
                            config, agent_id, entry.frame, self._scenario_map
                        )

                        if macro_action is None:
                            continue

                        # Get maneuvers from this macro-action
                        maneuvers = macro_action.maneuvers
                        if not maneuvers:
                            continue

                        # Create maneuver steps
                        new_steps = []
                        for i, man in enumerate(maneuvers):
                            step = ManeuverStep(
                                maneuver=man,
                                macro_action_type=ma_class.__name__,
                                step_index=i,
                                total_steps=len(maneuvers)
                            )
                            new_steps.append(step)

                        # Build new entry with all maneuvers from this macro-action
                        all_steps = entry.maneuver_steps + new_steps

                        # Get the frame after executing this macro-action
                        new_frame = MacroAction.play_forward_macro_action(
                            agent_id, self._scenario_map, entry.frame, macro_action
                        )

                        new_entry = QueueEntry(
                            maneuver_steps=all_steps,
                            frame=new_frame,
                            depth=entry.depth + 1,
                            completed_macro_actions=entry.completed_macro_actions + [ma_class.__name__]
                        )
                        queue.append(new_entry)

                        # Generate variations (e.g., without GiveWay)
                        if self.generate_variations:
                            variations = self._generate_maneuver_variations(
                                entry, new_steps, agent_id, ma_class.__name__
                            )
                            for var_entry in variations:
                                queue.append(var_entry)

                    except Exception as e:
                        logger.debug(f"Failed to expand {ma_class.__name__}: {e}")
                        continue

        # Extract results
        trajectories = [sol[1] for sol in solutions]
        plans = [sol[0] for sol in solutions]

        logger.info(f"Generated {len(solutions)} candidate sequences for agent {agent_id}")                                                                                                      
        for i, (plan, _) in enumerate(solutions[:5]):                                                                                                                                            
            logger.debug(f"  Candidate {i}: {[type(m).__name__ for m in plan]}")
        return trajectories, plans

    def _generate_maneuver_variations(self,
                                      base_entry: QueueEntry,
                                      new_steps: List[ManeuverStep],
                                      agent_id: int,
                                      ma_type: str) -> List[QueueEntry]:
        """Generate variations by removing optional maneuver types.

        For each optional maneuver type present in new_steps (as defined by
        OPTIONAL_MANEUVER_TYPES), builds a variation with that type removed.
        When a maneuver is removed, the preceding maneuver's trajectory is
        extended to absorb the removed maneuver's path segment, preventing
        spatial gaps in the combined trajectory.

        Args:
            base_entry: The entry before adding new_steps
            new_steps: The new maneuver steps to potentially modify
            agent_id: Agent ID
            ma_type: Macro-action type name

        Returns:
            List of QueueEntry variations
        """
        variations = []

        # Find which optional types are present in the new steps
        optional_types_present = (
            {type(step.maneuver) for step in new_steps}
            & self.OPTIONAL_MANEUVER_TYPES
        )

        for opt_type in optional_types_present:
            # Build variation without this optional type, extending the
            # preceding maneuver's trajectory to cover the gap left by removal.
            filtered_steps = []

            for step in new_steps:
                if isinstance(step.maneuver, opt_type):
                    # Extend the preceding step's trajectory to cover this gap
                    removed_traj = step.maneuver.trajectory
                    if filtered_steps and removed_traj is not None and len(removed_traj.path) > 1:
                        prev_step = filtered_steps[-1]
                        prev_traj = prev_step.maneuver.trajectory

                        if prev_traj is not None:
                            extended_path = np.concatenate(
                                [prev_traj.path, removed_traj.path[1:]])
                            extended_vel = np.concatenate(
                                [prev_traj.velocity, removed_traj.velocity[1:]])
                            extended_traj = VelocityTrajectory(
                                extended_path, extended_vel)

                            # Shallow-copy the maneuver so the original is unaffected
                            extended_maneuver = copy(prev_step.maneuver)
                            extended_maneuver.trajectory = extended_traj

                            filtered_steps[-1] = ManeuverStep(
                                maneuver=extended_maneuver,
                                macro_action_type=prev_step.macro_action_type,
                                step_index=prev_step.step_index,
                                total_steps=prev_step.total_steps,
                            )
                    # else: removed maneuver is first step with no predecessor — drop it
                else:
                    filtered_steps.append(ManeuverStep(
                        maneuver=step.maneuver,
                        macro_action_type=step.macro_action_type,
                        step_index=step.step_index,
                        total_steps=step.total_steps,
                    ))

            if filtered_steps:  # Only if there are remaining steps
                # Re-index the steps
                for i, step in enumerate(filtered_steps):
                    step.step_index = i
                    step.total_steps = len(filtered_steps)

                all_steps = base_entry.maneuver_steps + filtered_steps

                # Build trajectory to get new frame
                trajectory = self._build_trajectory(all_steps)
                if trajectory is not None:
                    # Get final state from trajectory
                    new_frame = base_entry.frame.copy()
                    new_frame[agent_id] = trajectory.final_agent_state

                    opt_type_name = opt_type.__name__
                    var_entry = QueueEntry(
                        maneuver_steps=all_steps,
                        frame=new_frame,
                        depth=base_entry.depth + 1,
                        completed_macro_actions=base_entry.completed_macro_actions + [f"{ma_type}_no{opt_type_name}"]
                    )
                    variations.append(var_entry)

        return variations

    def _get_applicable_macro_actions(self, state: AgentState) -> List[type]:
        """Get applicable macro-action types for the current state."""
        applicable = []

        # Check each macro-action type
        ma_types = [Continue, Exit, ChangeLaneLeft, ChangeLaneRight]

        for ma_class in ma_types:
            try:
                if ma_class.applicable(state, self._scenario_map):
                    applicable.append(ma_class)
            except Exception as e:
                logger.debug(f"Applicability check failed for {ma_class.__name__}: {e}")
                continue

        return applicable

    def _build_trajectory(self, steps: List[ManeuverStep]) -> Optional[VelocityTrajectory]:
        """Build a combined trajectory from maneuver steps."""
        if not steps:
            return None

        paths = []
        velocities = []

        for step in steps:
            traj = step.maneuver.trajectory
            if traj is None or len(traj.path) == 0:
                return None

            if paths:
                # Skip first point to avoid duplicates at junctions
                paths.append(traj.path[1:])
                velocities.append(traj.velocity[1:])
            else:
                paths.append(traj.path)
                velocities.append(traj.velocity)

        if not paths:
            return None

        combined_path = np.concatenate(paths, axis=0)
        combined_velocity = np.concatenate(velocities)

        if len(combined_path) < 2:
            return None

        return VelocityTrajectory(combined_path, combined_velocity)

    def _goal_reached(self, goal: Goal, trajectory: VelocityTrajectory) -> bool:
        """Check if trajectory has reached the goal."""
        if trajectory is None or len(trajectory.path) == 0:
            return False

        if goal.reached(trajectory.path[-1]):
            return True
        elif not isinstance(goal, StoppingGoal):
            return goal.passed_through_goal(trajectory)
        return False

    @staticmethod
    def _plan_signature(maneuvers: List[Maneuver]) -> tuple:
        """Create a hashable signature from a maneuver list for deduplication."""
        return tuple(type(m).__name__ for m in maneuvers)

    def _make_state_key(self, state: AgentState, completed_mas: List[str]) -> tuple:
        """Create a hashable key for state tracking."""
        return (
            round(state.position[0], 1),
            round(state.position[1], 1),
            round(state.heading, 2),
            tuple(completed_mas[-2:])  # Only track last 2 to allow some revisiting
        )
