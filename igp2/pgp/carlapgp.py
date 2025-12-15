import logging
import numpy as np
import copy
from typing import Tuple, List, Dict
from shapely.geometry import Point, LineString, Polygon
from collections import deque
import yaml
import torch

from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road_lanes import LeftLanes, CenterLanes, RightLanes
from igp2.opendrive.map import Map
from igp2.vector_map import Dataset
from igp2.pgp.train_eval.initialization import initialize_prediction_model
from igp2.vector_map.plot_vector_map import plot_vector_map
from igp2.core.vehicle import Observation

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CarlaPGP:
    def __init__(self, xodr_path):
        self.__xodr_path = xodr_path
        self.__dataset = Dataset.parse_from_opendrive(xodr_path)
    
        self.t_h, self.t_f, self.fps = 2, 6, 15
        self.__dataset.t_h, self.__dataset.t_f, self.__dataset.fps = \
            self.t_h, self.t_f, self.fps
        self.__dataset.interval = self.interval 
        self.__agent_history = {}
        self.__trajectories = None
        self.__probabilities = None
        self.__traversals = None

        yaml_path = "/home/emran/IGP2/igp2/pgp/configs/pgp_gatx2_lvm_traversal.yml"
        # Load model config file
        with open(yaml_path, 'r') as yaml_file:
            cfg = yaml.safe_load(yaml_file)

        # Initialize model
        self.__model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                                 cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.__model = self.__model.float().to(device)
        self.__model.eval()

        # Load checkpoint
        checkpoint_path = "/home/emran/IGP2/igp2/pgp/weights/PGP_lr-scheduler.tar" 
        checkpoint = torch.load(checkpoint_path)
        self.__model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print("Model:", self.__model)

    def update(self, observation: Observation):
        for agent_id, agent_state in observation.frame.items():
            if not agent_id in self.__agent_history:
                self.__agent_history[agent_id] = deque([self.state2vector(agent_id, agent_state, first=True)], maxlen=self.interval)
            else:
                self.__agent_history[agent_id].append(self.state2vector(agent_id, agent_state))
            
    def predict_trajectories(self):
        agent_ids, agent_inputs = [], []
        for agent_id, agent_history in self.__agent_history.items():
            if len(agent_history) >= self.interval:
                agent_state = agent_history[-1]
                self.__dataset.generate_graph( \
                    agent_pose=[*agent_state[:2], heading_from_history(agent_history)])

                # plot_vector_map(self.__dataset, self.__dataset, markings=True, agent=True)

                target_agent_representation = transform_states_to_vehicle_frame(np.array(list(agent_history)))
                map_representation = self.__dataset.get_map_representation()
                surrounding_agent_representation = self.__dataset.get_surrounding_agent_representation(agent_id, self.__agent_history) # This could be done in a batch
                agent_node_masks = self.__dataset.get_agent_node_masks(map_representation, surrounding_agent_representation)

                # Add to inputs
                agent_ids.append(agent_id)
                agent_inputs.append({
                    "target_agent_representation": target_agent_representation,
                    "map_representation": map_representation,
                    "surrounding_agent_representation": surrounding_agent_representation,
                    "agent_node_masks": agent_node_masks,
                    "init_node": self.__dataset.get_initial_node(map_representation)
                })
        
        if len(agent_inputs) > 0:
            batched_inputs = stack_dicts(agent_inputs)
            trajectories = self.__model(batched_inputs)

            self.__trajectories = {}
            self.__probabilities = {}
            self.__traversals = {}
            for traj_id, agent_id in enumerate(agent_ids):
                # Assign trajectories to self.__trajectories                    
                self.__trajectories[agent_id] = self.transform_trajectories(trajectories['traj'][traj_id].cpu().detach().numpy())
                self.__probabilities[agent_id] = trajectories['probs'][traj_id].cpu().detach().numpy()
                self.__traversals[agent_id] = trajectories['traversals'][traj_id].cpu().detach().numpy()

    def transform_trajectories(self, traj_preds):
        out_trajs = np.zeros_like(traj_preds)
        for traj_idx, traj_pred in enumerate(traj_preds):
            out_trajs[traj_idx, :, 0] = (traj_pred[:, 1] - 2.0) * 0.5 # EMRAN don't know why they appear ahead of car?
            out_trajs[traj_idx, :, 1]= -traj_pred[:, 0] * 0.5

        return out_trajs

    def state2vector(self, agent_id, agent_state, first=False, yaw_window=5):
        # --- local helpers (so no NameError) ---
        def wrap_to_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        def unwrap_angles(angles):
            out = np.array(angles, dtype=float)
            for i in range(1, len(out)):
                d = wrap_to_pi(out[i] - out[i-1])
                out[i] = out[i-1] + d
            return out

        def fit_slope(t, y):
            t = np.asarray(t, dtype=float)
            y = np.asarray(y, dtype=float)
            t0 = t.mean()
            denom = np.dot(t - t0, t - t0)
            if denom < 1e-9:
                return 0.0
            return float(np.dot(t - t0, y - y.mean()) / denom)

        # --- current state ---
        x, y = agent_state.position
        v = np.array(agent_state.velocity, dtype=float)
        a = np.array(agent_state.acceleration, dtype=float)

        speed = float(np.linalg.norm(v))
        acceleration = float(np.linalg.norm(a))

        dt = 1.0 / float(self.fps)
        MIN_SPEED = 0.5       # m/s
        MAX_YAWRATE = 3.0     # rad/s (tune if needed)

        # --- build window (history + current) ---
        hist = self.__agent_history.get(agent_id, [])
        hist_list = list(hist)  # hist might be deque, so no slicing directly

        xs, ys = [], []
        headings = []

        if (not first) and len(hist_list) > 0:
            start = max(0, len(hist_list) - (yaw_window - 1))
            for row in hist_list[start:]:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
                # If you ever store heading in history later, put it at idx 5.
                headings.append(float(row[5]) if len(row) >= 6 else None)

        xs.append(float(x))
        ys.append(float(y))
        headings.append(float(agent_state.heading))

        n = len(xs)
        yaw_rate = 0.0

        if n >= 3:
            t = np.arange(n) * dt

            have_all_headings = all(h is not None for h in headings)
            if have_all_headings:
                psi = unwrap_angles(headings)
                yaw_rate = fit_slope(t, psi)
            else:
                dx = np.diff(xs)
                dy = np.diff(ys)
                step_dist = np.hypot(dx, dy)

                # If movement is tiny, travel-direction yaw is meaningless
                if float(step_dist.sum()) >= MIN_SPEED * (n - 1) * dt:
                    psi_steps = np.arctan2(dy, dx)
                    psi = np.concatenate([[psi_steps[0]], psi_steps])
                    psi = unwrap_angles(psi)
                    yaw_rate = fit_slope(t, psi)
                else:
                    yaw_rate = 0.0

        yaw_rate = float(np.clip(yaw_rate, -MAX_YAWRATE, MAX_YAWRATE))
        return [x, y, speed, acceleration, yaw_rate]

    def remove(self, agent_id):
        del self.__agent_history[agent_id]
        del self.__trajectories[agent_id]
        del self.__probabilities[agent_id]
        del self.__traversals[agent_id]

    @property
    def dataset(self):
        return self.__dataset

    @property
    def agent_history(self):
        return self.__agent_history

    @property
    def trajectories(self):
        return self.__trajectories
    
    @property   
    def probabilities(self):
        return self.__probabilities
    
    @property
    def traversals(self):
        return self.__traversals

    @property
    def interval(self):
        return int(self.t_h*self.fps)

def transform_states_to_vehicle_frame(states: np.ndarray) -> np.ndarray:
    """
    Transforms a sequence of vehicle states into the ego (vehicle) frame,
    where the most recent state (last in the array) is at the origin, facing +y (forward),
    and +x points to the right of the vehicle.

    Args:
        states (np.ndarray): Array of shape (T, 5) where each row is
                             [x, y, speed, acceleration, yaw_rate].

    Returns:
        np.ndarray: Transformed states of shape (T, 5) in ego frame.
    """
    assert states.shape[1] == 5, "Each state must have 5 elements: [x, y, speed, acceleration, yaw_rate]"

    # Get last two positions to estimate heading (in world frame)
    x0, y0 = states[-1, 0], states[-1, 1]
    x1, y1 = states[-2, 0], states[-2, 1]
    
    # Estimate heading angle (yaw) in world frame
    heading = np.arctan2(y0 - y1, x0 - x1)  # in radians

    # Translate positions relative to most recent position
    dx = states[:, 0] - x0
    dy = states[:, 1] - y0

    # Rotate positions into ego frame (+y forward, +x right)
    # So rotate by -(heading - π/2)
    theta = -(heading - np.pi / 2)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_ego = cos_theta * dx - sin_theta * dy
    y_ego = sin_theta * dx + cos_theta * dy

    # Copy rest of features (optional: you can also rotate velocity vector if needed)
    transformed_states = states.copy()
    transformed_states[:, 0] = x_ego
    transformed_states[:, 1] = y_ego

    return transformed_states



def heading_from_history(agent_history):
    return np.arctan2(agent_history[-1][1] - agent_history[-2][1], agent_history[-1][0] - agent_history[-2][0])

def stack_dicts(dict_list):
    if not dict_list:
        raise ValueError("dict_list cannot be empty")

    first = dict_list[0]

    if isinstance(first, dict):
        return {
            key: stack_dicts([d[key] for d in dict_list])
            for key in first
        }

    # Force float32
    tensor_list = [torch.from_numpy(arr).float() for arr in dict_list]

    if len(tensor_list) == 1:
        return tensor_list[0].unsqueeze(0).to(device)
    else:
        return torch.stack(tensor_list, dim=0).to(device)
