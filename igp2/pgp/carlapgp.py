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

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CarlaPGP:
    def __init__(self, xodr_path):
        self.__xodr_path = xodr_path
        self.__dataset = Dataset.parse_from_opendrive(xodr_path)
    
        self.t_h, self.t_f, self.fps = 2, 6, 20
        self.__dataset.t_h, self.__dataset.t_f, self.__dataset.fps = \
            self.t_h, self.t_f, self.fps
        self.__dataset.interval = self.interval 
        self.__agent_history = {}
        self.__trajectories = None
        self.__probabilities = None

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

    def update(self, frame):
        for agent_id, agent_state in frame.items():
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
            for traj_id, agent_id in enumerate(agent_ids):
                # Assign trajectories to self.__trajectories                    
                self.__trajectories[agent_id] = trajectories['traj'][traj_id].cpu().detach().numpy()
                self.__probabilities[agent_id] = trajectories['probs'][traj_id].cpu().detach().numpy()

    def state2vector(self, agent_id, agent_state, first=False):
        x, y = agent_state.position
        speed = np.linalg.norm(agent_state.velocity) # Might need to be scalar # Actually I think so
        acceleration = np.linalg.norm(agent_state.acceleration) # Might need to be scalar # Actually I think so
        yaw_rate = 0.0 if first else agent_state.heading - self.__agent_history[agent_id][-1][4]

        return [x, y, speed, acceleration, yaw_rate]

    def remove(self, agent_id):
        del self.__agent_history[agent_id]
        del self.__trajectories[agent_id]
        del self.__probabilities[agent_id]

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
    def interval(self):
        return int(self.t_h*self.fps)

def transform_states_to_vehicle_frame(states: np.ndarray) -> np.ndarray:
    """
    Transforms a sequence of vehicle states into the ego (vehicle) frame,
    where the most recent state (last in the array) is at the origin, facing +x.

    Args:
        states (np.ndarray): Array of shape (T, 5) where each row is
                             [x, y, speed, acceleration, yaw_rate].

    Returns:
        np.ndarray: Transformed states of shape (T, 5) in ego frame.
    """
    assert states.shape[1] == 5, "Each state must have 5 elements: [x, y, speed, acceleration, yaw_rate]"
    
    # Extract the reference state (latest state)
    x0, y0, _, _, yaw0 = states[-1]

    # Shift positions
    dx = states[:, 0] - x0
    dy = states[:, 1] - y0

    # Rotate positions by -yaw0 to align with ego heading
    cos_yaw = np.cos(-yaw0)
    sin_yaw = np.sin(-yaw0)
    x_ego = cos_yaw * dx - sin_yaw * dy
    y_ego = sin_yaw * dx + cos_yaw * dy

    # Copy other values unchanged
    transformed_states = states.copy()
    transformed_states[:, 0] = x_ego
    transformed_states[:, 1] = y_ego

    return transformed_states


def heading_from_history(agent_history):
    return np.arctan2(agent_history[-1][0] - agent_history[-2][0], agent_history[-1][1] - agent_history[-2][1])

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
