from typing import List
import logging
import copy

from igp2.agents.macro_agent import MacroAgent
from igp2.core.agentstate import AgentState
from igp2.core.vehicle import Action, Observation, TrajectoryPrediction
from igp2.core.goal import Goal
from igp2.planlibrary.macro_action import MacroAction, Continue, Exit
from igp2.recognition.astar import AStar

logger = logging.getLogger(__name__)


class TrafficAgent(MacroAgent):
    """ Agent that follows a list of MAs, optionally calculated using A*.

    Args:
        agent_id: ID of the agent
        initial_state: Starting state of the agent
        goal: Optional final goal of the agent
        fps: Execution rate of the environment simulation
        macro_actions: Optional pre-specified macro actions to follow
        open_loop: If True, follow pre-computed trajectory without feedback control.
                   If False, use closed-loop maneuver control. Default: False.
                   NOTE: open_loop=True requires MacroAction to support trajectory-based
                   execution, which is not fully implemented. Use closed-loop for now.
    """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: "Goal" = None,
                 fps: int = 20,
                 macro_actions: List[MacroAction] = None,
                 open_loop: bool = False,
                 pgp_drive: bool = False,
                 pgp_control: bool = False):
        super(TrafficAgent, self).__init__(agent_id, initial_state, goal, fps)
        self._astar = AStar(max_iter=1000)
        self._macro_actions = []
        self._open_loop = open_loop
        if macro_actions is not None and macro_actions:
            self.set_macro_actions(macro_actions)
        self._current_macro_id = 0
        self._pgp_drive = pgp_drive
        self._pgp_control = pgp_control

    def __repr__(self) -> str:
        return f"TrafficAgent(ID={self.agent_id})"

    def __deepcopy__(self, memo):
        """Custom deepcopy to avoid cycles through scenario and frame."""
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        dup = cls.__new__(cls)
        memo[id(self)] = dup

        # Copy primitive and small attributes
        for k, v in self.__dict__.items():
            # Skip cyclic or environment-heavy fields
            if k in ("_frame", "_scenario_map", "_planner", "_macro_actions", "_current_macro"):
                setattr(dup, k, None)
            elif k == "_astar":
                setattr(dup, k, v)  # shallow copy is fine
            else:
                setattr(dup, k, copy.deepcopy(v, memo))

        return dup

    def set_macro_actions(self, new_macros: List[MacroAction]):
        """ Specify a new set of macro actions to follow. """
        assert len(new_macros) > 0, "Empty macro list given!"
        if not self._open_loop:
            for macro in new_macros:
                macro.to_closed_loop()
        self._macro_actions = new_macros
        self._current_macro = new_macros[0]

    def set_destination(self, observation: Observation, goal: Goal = None):
        """ Set the current destination of this vehicle and calculate the shortest path to it using A*.

            Args:
                observation: The current observation.
                goal: Optional new goal to override the current one.
        """
        if goal is not None:
            self._goal = goal

        logger.info(f"Finding path for TrafficAgent ID {self.agent_id} (open_loop={self._open_loop})")
        _, actions = self._astar.search(self.agent_id,
                                        observation.frame,
                                        self._goal,
                                        observation.scenario_map,
                                        open_loop=True)

        if len(actions) == 0:
            print("len actions == 0 so failing due to astar")
            raise RuntimeError(f"Couldn't find path to goal {self.goal} for TrafficAgent {self.agent_id}.")
        self._macro_actions = actions[0]
        # Convert to closed-loop if configured
        if not self._open_loop:
            for macro in self._macro_actions:
                macro.to_closed_loop()
        self._current_macro = self._macro_actions[0]

    def done(self, observation: Observation) -> bool:
        """ Returns true if there are no more actions on the macro list and the current macro is finished,
        OR if the goal has been reached while on the final macro action. """
        macro_done = self._current_macro_id + 1 >= len(self._macro_actions) and super(TrafficAgent, self).done(observation)
        goal_reached = self.goal.reached(observation.frame[self.agent_id].position) if self.goal is not None else True

        # Check if we're on the final macro action and have reached the goal
        # This allows early completion when the goal is reached, without waiting
        # for the maneuver to report done (which requires driving past the endpoint)
        on_final_macro = self._current_macro_id + 1 >= len(self._macro_actions)
        done = macro_done or (on_final_macro and goal_reached)

        if macro_done and not goal_reached:
            try:
                logger.debug("set_destination in done")
                self.set_destination(observation)
            except RuntimeError as e:
                # If goal can't be reached, then just follow lane until end of episode.
                logger.info(f"Agent {self.agent_id} at {observation.frame[self.agent_id].position} couldn't reach goal {self.goal}.")
                logger.info(e)
                state = observation.frame[self.agent_id]
                scenario_map = observation.scenario_map
                if Continue.applicable(state, scenario_map):
                    self.update_macro_action(Continue, {}, observation)
                elif Exit.applicable(state, scenario_map):
                    for args in Exit.get_possible_args(state, scenario_map, self.goal):
                        self.update_macro_action(Exit, args, observation)
                        break
                else:
                    return True
                self._current_macro_id = 0
                self._macro_actions = [self._current_macro]
            return False
        return done

    def next_action(self, observation: Observation, prediction: TrajectoryPrediction = None) -> Action:
        if self.current_macro is None:
            if len(self._macro_actions) == 0:
                logger.debug("set_destination in next_action")
                self.set_destination(observation)

        if self._current_macro.done(observation):
            # logger.debug(f"Macro actions for Agent {self.agent_id}) self._macro_actions, self._current_macro: {self._macro_actions, self._current_macro, self._current_macro_id}")
            if self._current_macro_id < len(self._macro_actions):
                try:
                    self._advance_macro(observation)
                except:
                    return Action(0, 0)
            else:
                logger.warning(f"TrafficAgent {self.agent_id} has no macro actions!")
                logger.debug("in TrafficAgent.next_action -> Returning Action(0, 0)")
                return Action(0, 0)

        # if self._current_macro_id >= len(self._macro_actions):
        #     logger.warning(f"TrafficAgent {self.agent_id} has no macro actions!")
        #     logger.debug("in TrafficAgent.next_action -> Second returning Action(0, 0)")
        #     return Action(0, 0)
        return self._current_macro.next_action(observation)

    def reset(self):
        super(TrafficAgent, self).reset()
        self._macro_actions = []
        self._current_macro_id = 0

    def _advance_macro(self, observation: Observation):
        if not self._macro_actions:
            raise RuntimeError("TrafficAgent has no macro actions.")

        self._current_macro_id += 1
        if self._current_macro_id >= len(self._macro_actions):
            # logger.debug(f"self._current_macro, self._current_macro_id, len(self._macro_actions): {self._current_macro, self._current_macro_id, len(self._macro_actions)}")
            # logger.debug(f"Agent {self.agent_id} has no more macro actions to execute. Setting self._current_macro to None.")
            # self._current_macro = None
            raise RuntimeError(f"Agent {self.agent_id} has no more macro actions to execute.")
        else:
            self._current_macro = self._macro_actions[self._current_macro_id]

    @property
    def macro_actions(self) -> List[MacroAction]:
        """ The current macro actions to be executed by the agent. """
        return self._macro_actions

    @property
    def open_loop(self) -> bool:
        """ Whether the agent executes in open-loop (trajectory following) or closed-loop (feedback control). """
        return self._open_loop
