"""
Keyboard-controlled belief agent.

Subclasses BeliefAgent so that the **human policy** (TwoStagePolicy) is
**not run** — the keyboard action from a ``KeyboardAgent`` is used instead.

The FrenetFrame (needed by inference) is created in ``TwoStagePolicy.__init__``
from the reference waypoints, so it is always available without running the
human NLP.  The rest of the ``next_action() -> _run_inference()`` pipeline is
unchanged: inference sees the keyboard action as "what the human did" and may
override it with an intervention.

In **longitudinal** planning mode, keyboard steering is ignored and a PID
lane-following controller steers the vehicle along the reference waypoints.
The human only controls acceleration (W/S keys).
"""

import logging
import time as _time
from typing import Dict, Any

import numpy as np

from igp2.agents.belief_agent import BeliefAgent
from igp2.agents.keyboard_agent import KeyboardAgent
from igp2.beliefcontrol.two_stage_policy import TwoStagePolicy
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.vehicle import Action, Observation
from igp2.opendrive.map import Map

logger = logging.getLogger(__name__)

# Lane-following PID gains
_LATERAL_KP = 1.5
_LATERAL_KD = 0.3
_LOOKAHEAD_MIN = 5.0   # metres
_LOOKAHEAD_SCALE = 0.5  # lookahead = max(min, speed * scale)


class KeyboardBeliefAgent(BeliefAgent):
    """BeliefAgent whose human actions come from keyboard input.

    The internal human policy (TwoStagePolicy) is kept for its FrenetFrame
    and ``_state_to_frenet`` utility, but ``select_action()`` is never called.
    The true policy still runs if needed (dual relevance, policy_only, etc.).

    In longitudinal mode the keyboard only controls acceleration; a PID
    controller tracks the reference waypoints for steering.
    """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: Goal = None,
                 fps: int = 20,
                 scenario_map: Map = None,
                 agent_beliefs: Dict[int, Dict[str, Any]] = None,
                 policy_type: str = "two_stage_opt",
                 plot_interval: bool = True,
                 human: bool = True,
                 intervention_type: str = 'none',
                 inference_type: str = 'naive',
                 relevance_method: str = 'naive',
                 planning_mode: str = '2d',
                 ref_controls: str = 'opt',
                 human_type: str = 'static',
                 **policy_kwargs):
        super().__init__(
            agent_id=agent_id,
            initial_state=initial_state,
            goal=goal,
            fps=fps,
            scenario_map=scenario_map,
            agent_beliefs=agent_beliefs,
            policy_type=policy_type,
            plot_interval=plot_interval,
            human=human,
            intervention_type=intervention_type,
            inference_type=inference_type,
            relevance_method=relevance_method,
            planning_mode=planning_mode,
            ref_controls=ref_controls,
            human_type=human_type,
            **policy_kwargs,
        )
        self._keyboard_agent = KeyboardAgent(agent_id, initial_state, goal, fps)
        self._longitudinal = (planning_mode == "longitudinal")
        self._wp_idx = 0  # waypoint tracker index for lane-following PID
        self._prev_heading_error = 0.0
        logger.info("KeyboardBeliefAgent %d: keyboard overlay active "
                     "(human policy SKIPPED, longitudinal=%s)",
                     agent_id, self._longitudinal)

    def _lane_follow_steer(self, ego_state: AgentState) -> float:
        """Compute a PID steering angle to follow the reference waypoints."""
        pos = np.array(ego_state.position[:2], dtype=float)
        heading = float(ego_state.heading)
        speed = float(ego_state.speed)
        wp = self._reference_waypoints

        if len(wp) < 2:
            return 0.0

        # Advance waypoint index past points already passed
        lookahead = max(_LOOKAHEAD_MIN, speed * _LOOKAHEAD_SCALE)
        while self._wp_idx < len(wp) - 1:
            if np.linalg.norm(wp[self._wp_idx] - pos) < lookahead * 0.5:
                self._wp_idx += 1
            else:
                break

        # Find the lookahead target
        target_idx = self._wp_idx
        for i in range(self._wp_idx, len(wp)):
            if np.linalg.norm(wp[i] - pos) >= lookahead:
                target_idx = i
                break
        else:
            target_idx = len(wp) - 1

        target = wp[target_idx]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        target_heading = np.arctan2(dy, dx)

        # Heading error (wrapped to [-pi, pi])
        error = (target_heading - heading + np.pi) % (2 * np.pi) - np.pi

        # PD control
        d_error = error - self._prev_heading_error
        self._prev_heading_error = error

        max_steer = 0.7
        steer = np.clip(_LATERAL_KP * error + _LATERAL_KD * d_error,
                         -max_steer, max_steer)
        return float(steer)

    def policy(self, observation: Observation) -> Action:
        """Skip the human policy, optionally run true policy, return keyboard.

        In longitudinal mode, steering comes from a lane-following PID and the
        keyboard only controls acceleration.  In 2d mode, the full keyboard
        action (accel + steer) is used.
        """
        ego_state = observation.frame.get(self.agent_id)
        if ego_state is None or self._scenario_map is None:
            return Action(acceleration=0.0, steer_angle=0.0, target_speed=0.0)

        if len(self._reference_waypoints) == 0:
            return Action(acceleration=0.0, steer_angle=0.0, target_speed=0.0)

        # Store all other agents (needed by collect_step diagnostics)
        true_other_agents = {aid: s for aid, s in observation.frame.items()
                             if aid != self.agent_id}
        self._last_all_other_agents = true_other_agents

        # --- Run true (ground-truth) policy only if needed ---
        true_action = None
        true_policy_time = 0.0
        if self._need_true_policy:
            t0 = _time.perf_counter()
            if isinstance(self._true_policy, TwoStagePolicy):
                true_action, _, _ = self._true_policy.select_action(
                    ego_state,
                    other_agents=true_other_agents or None,
                    agent_trajectories=self._true_agent_trajectories or None)
            else:
                true_action, _, _ = self._true_policy.select_action(ego_state)
            true_policy_time = _time.perf_counter() - t0

        self.last_step_timing = {
            'human_policy': 0.0,
            'true_policy': true_policy_time,
            'plotting': 0.0,
        }
        self._last_true_action = true_action

        # Get keyboard action
        kb_action = self._keyboard_agent.next_action(observation)

        if self._longitudinal:
            # Longitudinal mode: keyboard controls acceleration only,
            # steering comes from lane-following PID
            steer = self._lane_follow_steer(ego_state)
            return Action(acceleration=kb_action.acceleration,
                          steer_angle=steer,
                          target_speed=kb_action.target_speed)

        return kb_action
