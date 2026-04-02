"""Per-step data collection for belief experiments."""

import os
import time
import logging
from typing import Dict, Optional

import numpy as np
from shapely.geometry import Polygon

from .data import StepRecord


def _agent_obb(position, heading, length, width) -> Polygon:
    """Build an oriented bounding box polygon for an agent."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    corners = np.array([
        [+hl, +hw],
        [+hl, -hw],
        [-hl, -hw],
        [-hl, +hw],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    world_corners = (rot @ corners.T).T + position[:2]
    return Polygon(world_corners)


def _check_ego_collision(ego_state, frame, ego_id):
    """Check if the ego vehicle's OBB overlaps any other agent/obstacle.

    Returns (colliding: bool, collider_id: int or None).
    """
    if ego_state is None or frame is None:
        return False, None

    ego_pos = np.array(ego_state.position[:2], dtype=float)
    ego_poly = _agent_obb(
        ego_pos, ego_state.heading,
        ego_state.metadata.length, ego_state.metadata.width)

    for aid, state in frame.items():
        if aid == ego_id:
            continue
        other_pos = np.array(state.position[:2], dtype=float)
        other_poly = _agent_obb(
            other_pos, state.heading,
            state.metadata.length, state.metadata.width)
        if ego_poly.intersects(other_poly):
            return True, aid
    return False, None


def collect_step(step: int, t0: float, ego_agent, ego_goal, frame,
                 prev_true_trajectories: Optional[Dict[int, np.ndarray]] = None,
                 prev_action: Optional[tuple] = None,
                 ) -> StepRecord:
    """Collect diagnostics for one simulation step.

    Args:
        step: Current step index.
        t0: Wall-clock start time of the experiment.
        ego_agent: The ego BeliefAgent instance.
        ego_goal: Ego goal (used for goal-reached check).
        frame: Current observation frame.
        prev_true_trajectories: True-policy predicted trajectories from the
            *previous* step.  Used to compute 1-step-ahead prediction error.
        prev_action: (acceleration, steer_angle) from the previous step's
            executed action.  Used to compute jerk and steering rate violations.
    """
    ego_id = ego_agent.agent_id
    ego_state = frame.get(ego_id) if frame else None

    human_policy = getattr(ego_agent, '_human_policy', None)
    true_policy = getattr(ego_agent, '_true_policy', None)

    # Human (belief) policy
    human_rollout = getattr(human_policy, 'last_rollout', None) if human_policy else None
    human_milp = getattr(human_policy, 'last_milp_rollout', None) if human_policy else None
    human_nlp_converged = None
    if human_policy is not None:
        human_nlp_converged = getattr(human_policy, '_prev_nlp_states', None) is not None
    human_obstacles = getattr(human_policy, 'last_obstacles', None) if human_policy else None
    human_other_agents = getattr(human_policy, 'last_other_agents', None) if human_policy else None
    human_trajectories = dict(ego_agent._human_agent_trajectories)

    # Reference waypoints from human policy's FrenetFrame
    ref_waypoints = None
    human_frenet = getattr(human_policy, 'frenet_frame', None) if human_policy else None
    if human_frenet is not None:
        ref_waypoints = getattr(human_frenet, '_waypoints', None)

    # True (ground-truth) policy
    true_rollout = getattr(true_policy, 'last_rollout', None) if true_policy else None
    true_milp = getattr(true_policy, 'last_milp_rollout', None) if true_policy else None
    true_nlp_converged = None
    if true_policy is not None:
        true_nlp_converged = getattr(true_policy, '_prev_nlp_states', None) is not None
    true_obstacles = getattr(true_policy, 'last_obstacles', None) if true_policy else None
    true_other_agents = getattr(true_policy, 'last_other_agents', None) if true_policy else None
    true_trajectories = dict(ego_agent._true_agent_trajectories)

    # Frenet state from whichever policy ran (prefer true, fallback to human)
    ego_frenet_state = None
    fs = getattr(true_policy, 'last_frenet_state', None) if true_policy else None
    if fs is None:
        fs = getattr(human_policy, 'last_frenet_state', None) if human_policy else None
    if fs is not None:
        ego_frenet_state = np.array(fs, dtype=float)

    dynamic_agents = {aid: s for aid, s in frame.items()
                      if aid != ego_id and aid >= 0} if frame else {}
    static_obstacles = {aid: s for aid, s in frame.items()
                        if aid < 0} if frame else {}

    # Goal reached: based on actual ego position only
    goal_reached = False
    if ego_goal is not None and ego_state is not None:
        goal_reached = ego_goal.reached(np.array(ego_state.position[:2]))

    # Actual collision check (OBB overlap)
    ego_collision, ego_collision_id = _check_ego_collision(
        ego_state, frame, ego_id)

    # Human policy constraint diagnostics
    human_diag = getattr(human_policy, 'last_diagnostics', None) if human_policy else None

    # True policy constraint diagnostics
    true_diag = getattr(true_policy, 'last_diagnostics', None) if true_policy else None

    # Ego timing breakdown
    ego_timing = dict(ego_agent.last_step_timing) if ego_agent.last_step_timing else None

    # 1-step-ahead prediction error: compare previous step's predicted
    # positions (index 1 of each trajectory) with actual positions now.
    # NOTE: uses TRUE trajectories. When belief inference is added, compute
    # a separate error from human (belief) trajectories.
    prediction_error = None
    if prev_true_trajectories and frame:
        errors = []
        for aid, pred_traj in prev_true_trajectories.items():
            actual_state = frame.get(aid)
            if actual_state is None or len(pred_traj) < 2:
                continue
            predicted_pos = pred_traj[1]  # 1-step-ahead prediction
            actual_pos = actual_state.position
            errors.append(float(np.linalg.norm(predicted_pos - actual_pos)))
        if errors:
            prediction_error = float(np.mean(errors))

    # --- Belief inference & intervention metrics ---
    belief_accuracy = None
    belief_marginals = None
    belief_ground_truth = None
    belief_config_results = None
    believed_config = None
    action_deviation = None
    intervention_active = False
    intervention_controls = None
    intervention_opt_states = None
    intervention_opt_controls = None
    intervention_ref_states = None
    intervention_ref_controls = None
    intervention_success = None

    belief_inference = getattr(ego_agent, '_belief_inference', None)
    agent_beliefs = getattr(ego_agent, 'agent_beliefs', {})

    # Ground-truth visibility from configured beliefs
    if agent_beliefs:
        belief_ground_truth = {
            aid: belief.visible for aid, belief in agent_beliefs.items()
        }

    # Kalman awareness traces (human + vehicle)
    human_awareness = None
    vehicle_awareness = None
    vehicle_awareness_bounds = None
    if belief_inference is not None:
        hk = getattr(belief_inference, '_human_kalman', None)
        vk = getattr(belief_inference, '_kalman_awareness', None)
        if hk is not None:
            h_phi = hk.phi
            human_awareness = {
                aid: float(h_phi[i])
                for i, aid in enumerate(hk.agent_ids)
            }
        if vk is not None:
            v_phi = vk.phi
            vehicle_awareness = {
                aid: float(v_phi[i])
                for i, aid in enumerate(vk.agent_ids)
            }
            vehicle_awareness_bounds = vk.phi_bounds()

    # Velocity particle state
    velocity_particles_snap = None
    velocity_kappa_gt = None
    if belief_inference is not None:
        vp_dict = getattr(belief_inference, '_velocity_particles', {})
        if vp_dict:
            velocity_particles_snap = {}
            for aid, vp in vp_dict.items():
                velocity_particles_snap[aid] = {
                    'kappa_values': list(vp.kappa_values),
                    'weights': list(vp.weights),
                    'mean': vp.weighted_mean(),
                    'std': vp.weighted_std(),
                    'ess': vp.effective_sample_size(),
                }
        gt_k = getattr(belief_inference, '_gt_kappa', {})
        if gt_k:
            velocity_kappa_gt = dict(gt_k)

    # Marginals, per-config costs, and accuracy from belief inference
    if belief_inference is not None:
        marginals = belief_inference.last_marginals
        if marginals:
            belief_marginals = dict(marginals)

            # Compute accuracy: for each agent in marginals, check if
            # the inferred classification matches ground truth
            hidden_threshold = getattr(belief_inference, '_hidden_threshold', 0.6)
            correct = 0
            total = 0
            for aid, p_hidden in marginals.items():
                gt_belief = agent_beliefs.get(aid)
                if gt_belief is None:
                    continue
                gt_visible = gt_belief.visible
                inferred_visible = p_hidden <= hidden_threshold
                if inferred_visible == gt_visible:
                    correct += 1
                total += 1
            if total > 0:
                belief_accuracy = correct / total

        # Per-configuration inference detail (costs, energies, probs)
        inf_results = belief_inference.last_results
        inf_energies = belief_inference.last_energies
        inf_probs = belief_inference.last_config_probs
        if inf_results and inf_energies and inf_probs:
            belief_config_results = []
            for i, r in enumerate(inf_results):
                belief_config_results.append({
                    'config': dict(r.config),
                    'pos_cost': float(r.pos_cost),
                    'vel_cost': float(r.vel_cost),
                    'energy': float(inf_energies[i]),
                    'prob': float(inf_probs[i]),
                    'milp_ok': r.milp_ok,
                    'nlp_ok': r.nlp_ok,
                })

    # Action deviation: compare human policy action to executed action
    last_human = getattr(ego_agent, '_last_human_action', None)
    last_executed = getattr(ego_agent, '_last_executed_action', None)
    action_deviation_accel = None
    action_deviation_steer = None
    if last_human is not None and last_executed is not None:
        da = last_executed.acceleration - last_human.acceleration
        dd = last_executed.steer_angle - last_human.steer_angle
        action_deviation = float(np.sqrt(da**2 + dd**2))
        action_deviation_accel = float(abs(da))
        action_deviation_steer = float(abs(dd))

    # Intervention detail
    if belief_inference is not None:
        interv = belief_inference.last_intervention
        if interv is not None:
            intervention_success = interv.get('success', False)
            believed_config = interv.get('believed_config')
            if intervention_success:
                intervention_active = True
                intervention_controls = interv.get('intervention')
                intervention_opt_states = interv.get('opt_states')
                intervention_opt_controls = interv.get('opt_controls')
                intervention_ref_states = interv.get('ref_states')
                intervention_ref_controls = interv.get('ref_controls')

    # For always_policy: every step is an override if actions differ
    intervention_type = getattr(ego_agent, '_intervention_type', None)
    if intervention_type == 'always_policy' and action_deviation is not None:
        if action_deviation > 1e-6:
            intervention_active = True

    # --- Per-step ego violations and cost ---
    ego_acceleration = None
    ego_steer_angle = None
    ego_accel_violated = False
    ego_steering_violated = False
    ego_jerk_violated = False
    ego_steer_rate_violated = False
    ego_jerk = None
    ego_steer_rate = None
    ego_step_cost = None
    ego_cost_lateral = None
    ego_cost_speed = None
    ego_cost_accel = None
    ego_cost_steering = None
    ego_cost_heading = None

    # Get NLP parameters from whichever policy is available
    _policy = human_policy or true_policy
    if _policy is not None and last_executed is not None:
        from igp2.beliefcontrol.second_stage import SecondStagePlanner
        params = dict(SecondStagePlanner.DEFAULTS)
        nlp_params = getattr(_policy, '_params', None)
        if nlp_params:
            params.update(nlp_params)

        a_exec = float(last_executed.acceleration)
        delta_exec = float(last_executed.steer_angle)
        ego_acceleration = a_exec
        ego_steer_angle = delta_exec

        # Control violations
        ego_accel_violated = abs(a_exec) > params['a_max'] + 1e-6
        ego_steering_violated = abs(delta_exec) > params['delta_max'] + 1e-6

        # Comfort metrics (rate-of-change magnitudes)
        fps = getattr(ego_agent, '_fps', 20)
        dt_sim = 1.0 / fps
        if prev_action is not None:
            prev_a, prev_delta = prev_action
            jerk = abs(a_exec - prev_a) / dt_sim
            steer_rate = abs(delta_exec - prev_delta) / dt_sim
            ego_jerk = jerk
            ego_steer_rate = steer_rate
            ego_jerk_violated = jerk > params['jerk_max'] + 1e-6
            ego_steer_rate_violated = steer_rate > params['delta_rate_max'] + 1e-6

        # Per-step cost (NLP objective terms)
        if ego_frenet_state is not None:
            s, d, phi, v = ego_frenet_state[:4]
            target_speed = getattr(_policy, 'target_speed', params['v_max'])
            # Reference s: how far the ego should have travelled at target speed
            s_ref = s  # self-referencing (no absolute reference available per step)
            ego_cost_lateral = float(params['w_d'] * d ** 2)
            ego_cost_speed = float(params['w_v'] * (v - target_speed) ** 2)
            ego_cost_accel = float(params['w_a'] * a_exec ** 2)
            ego_cost_steering = float(params['w_delta'] * delta_exec ** 2)
            ego_cost_heading = float(params['w_phi'] * phi ** 2)
            ego_step_cost = (ego_cost_lateral + ego_cost_speed
                             + ego_cost_accel + ego_cost_steering
                             + ego_cost_heading)

    return StepRecord(
        step=step,
        wall_time=time.time() - t0,
        ego_position=np.array(ego_state.position) if ego_state else None,
        ego_speed=float(ego_state.speed) if ego_state else None,
        ego_heading=float(ego_state.heading) if ego_state else None,
        ego_frenet_state=ego_frenet_state,
        human_rollout=human_rollout,
        human_milp_rollout=human_milp,
        human_nlp_converged=human_nlp_converged,
        human_obstacles=human_obstacles,
        human_other_agents=human_other_agents,
        human_trajectories=human_trajectories,
        true_rollout=true_rollout,
        true_milp_rollout=true_milp,
        true_nlp_converged=true_nlp_converged,
        true_obstacles=true_obstacles,
        true_other_agents=true_other_agents,
        true_trajectories=true_trajectories,
        dynamic_agents=dynamic_agents,
        static_obstacles=static_obstacles,
        goal_reached=goal_reached,
        # Constraint diagnostics from human policy
        human_diag_milp_ok=human_diag.get('milp_ok') if human_diag else None,
        human_diag_nlp_ok=human_diag.get('nlp_ok') if human_diag else None,
        # Constraint diagnostics from true policy
        true_diag_milp_ok=true_diag.get('milp_ok') if true_diag else None,
        true_diag_nlp_ok=true_diag.get('nlp_ok') if true_diag else None,
        true_diag_velocity_violated=true_diag.get('velocity_violated') if true_diag else None,
        true_diag_acceleration_violated=true_diag.get('acceleration_violated') if true_diag else None,
        true_diag_steering_violated=true_diag.get('steering_violated') if true_diag else None,
        true_diag_jerk_violated=true_diag.get('jerk_violated') if true_diag else None,
        true_diag_steer_rate_violated=true_diag.get('steer_rate_violated') if true_diag else None,
        true_diag_road_violations=len(true_diag.get('road_boundary_violations', [])) if true_diag else 0,
        true_diag_collision_violations=len(true_diag.get('collision_violations', [])) if true_diag else 0,
        # Timing and prediction
        ego_timing=ego_timing,
        prediction_error=prediction_error,
        # Belief inference & intervention
        belief_accuracy=belief_accuracy,
        belief_marginals=belief_marginals,
        belief_ground_truth=belief_ground_truth,
        human_awareness=human_awareness,
        vehicle_awareness=vehicle_awareness,
        vehicle_awareness_bounds=vehicle_awareness_bounds,
        belief_config_results=belief_config_results,
        believed_config=believed_config,
        action_deviation=action_deviation,
        intervention_active=intervention_active,
        intervention_controls=intervention_controls,
        intervention_opt_states=intervention_opt_states,
        intervention_opt_controls=intervention_opt_controls,
        intervention_ref_states=intervention_ref_states,
        intervention_ref_controls=intervention_ref_controls,
        intervention_success=intervention_success,
        reference_waypoints=ref_waypoints,
        ego_collision=ego_collision,
        ego_collision_id=ego_collision_id,
        # Actual ego violations
        ego_accel_violated=ego_accel_violated,
        ego_steering_violated=ego_steering_violated,
        ego_jerk_violated=ego_jerk_violated,
        ego_steer_rate_violated=ego_steer_rate_violated,
        ego_jerk=ego_jerk,
        ego_steer_rate=ego_steer_rate,
        ego_acceleration=ego_acceleration,
        ego_steer_angle=ego_steer_angle,
        # Decomposed action deviation
        action_deviation_accel=action_deviation_accel,
        action_deviation_steer=action_deviation_steer,
        velocity_particles=velocity_particles_snap,
        velocity_kappa_gt=velocity_kappa_gt,
        # Per-step ego cost
        ego_step_cost=ego_step_cost,
        ego_cost_lateral=ego_cost_lateral,
        ego_cost_speed=ego_cost_speed,
        ego_cost_accel=ego_cost_accel,
        ego_cost_steering=ego_cost_steering,
        ego_cost_heading=ego_cost_heading,
    )
