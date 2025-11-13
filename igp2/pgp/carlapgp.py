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
        self.__agent_history = {}
        self.__trajectories = {}

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
        batched_inputs, agent_inputs = None, []
        for agent_id, agent_state in frame.items():
            if not agent_id in self.__agent_history:
                self.__agent_history[agent_id] = deque([self.state2vector(agent_id, agent_state, first=True)], maxlen=16) # int(self.t_h*self.fps))
            else:
                self.__agent_history[agent_id].append(self.state2vector(agent_id, agent_state))
            
        for agent_id, agent_state in frame.items():
            if len(self.__agent_history[agent_id]) == 16: # self.t_h * self.fps:
                self.__dataset.generate_graph(agent_pose=[*agent_state.position, agent_state.heading])
                # Add to inputs
                agent_inputs.append({
                    "target_agent_representation": np.array(list(self.__agent_history[agent_id])),
                    "map_representation": self.__dataset.get_map_representation(),
                    "surrounding_agent_representation": self.__dataset.get_surrounding_agent_representation(agent_id, self.__agent_history) # This could be done in a batch
                })
        
        if len(agent_inputs) > 0:
            batched_inputs = stack_dicts(agent_inputs)
            trajectories = self.__model(batched_inputs)

        for agent_id, agent_state in frame.items():
            # Assign trajectories to self.__trajectories
            pass
        
    def state2vector(self, agent_id, agent_state, first=False):
        x, y = agent_state.position
        speed = np.linalg.norm(agent_state.velocity) # Might need to be scalar # Actually I think so
        acceleration = np.linalg.norm(agent_state.acceleration) # Might need to be scalar # Actually I think so
        yaw_rate = 0.0 if first else agent_state.heading - self.__agent_history[agent_id][-1][4]

        return [x, y, speed, acceleration, yaw_rate]

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
