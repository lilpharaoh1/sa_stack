import logging
import numpy as np
import copy
from typing import Tuple, List, Dict
from shapely.geometry import Point, LineString, Polygon


from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road_lanes import LeftLanes, CenterLanes, RightLanes
from igp2.opendrive.elements.geometry import normalise_angle
from igp2.opendrive.map import Map


logger = logging.getLogger(__name__)


class Dataset(Map):
    POLYLINE_LENGTH = 20
    DEFAULT_BOUNDS = [-50, 50, -20, 80]
    # DEFAULT_BOUNDS = [-5, 5, -20, 70]
    # DEFAULT_BOUNDS = [-15, 15, -10, 60]
    # DEFAULT_BOUNDS = [-15, 15, -10, 40]
    # DEFAULT_BOUNDS = [-100, 100, -100, 100]
    # DEFAULT_BOUNDS = [-10, 10, -5, 18]

    def __init__(self, opendrive: OpenDrive = None, process_graph=False, bounds=None):
        super().__init__(opendrive)

        # Save original OpenDrive state
        self.__orig_opendrive = OpenDrive()
        self.__orig_opendrive.header = opendrive.header
        self.__orig_opendrive._roads = list(opendrive.roads)
        self.__orig_opendrive._junctions = list(opendrive.junctions)
        self.__orig_opendrive._junction_groups = list(opendrive.junction_groups)

        self.agent = None
        self.bounds = bounds if bounds is not None else self.DEFAULT_BOUNDS

        self.max_nodes = 128 # EMRAN make some compute_stats-esque function for this
        self.max_vehicles = 64 # EMRAN make some compute_stats-esque function for this
        self.max_pedestrians = 64 # EMRAN make some compute_stats-esque function for this

        self.t_h, self.t_f, self.fps = None, None, None
        self.interval = None

        if process_graph:
            self.__process_graph()

    def generate_graph(self, agent_pose: Tuple[float, float, float] = None):
        if agent_pose:
            self.agent = agent_pose
            self.filter_opendrive_around_point(agent_pose)
        self.__process_graph()

    def __process_graph(self):
        nodes, edges = {}, []


        for road_id, road in self.roads.items():
            for lane_section in road.lanes.lane_sections:
                # Process nodes
                lanes = [lane for lane in lane_section.all_lanes if not lane.id == 0 and lane.type == "driving"]
                lane_ids = [lane.id for lane in lanes]

                lane_feats = self.get_lane_feats(road, lanes)


                lane_nodes, internal_edges = self.split_lanes(
                    lanes, lane_ids, lane_feats, road_id, self.POLYLINE_LENGTH
                )


                nodes.update(lane_nodes)
                edges.extend(internal_edges)
                edges.extend(self.get_edges(road, lanes, lane_ids))


        edges.extend(self.add_lane_change_edges(nodes))

        self.__graph = {"nodes": nodes, "edges": edges}

    def get_lane_feats(self, road, lanes):
        return [{"type": lane.type, "junction": road.junction != -1} for lane in lanes]

    def split_lanes(self, lanes, lane_ids, lane_feats, road_id, max_length):
        lane_segments, internal_edges = {}, []

        for idx, lane in enumerate(lanes):
            x_world, y_world, yaw_world = self.resample_midline(
            lane.midline.xy[0], lane.midline.xy[1], max_length
            )
            x_ego, y_ego, yaw_ego = self.world_to_ego_frame(x_world, y_world, yaw_world)

            previous_seg = None
            for seg_id, (x, y, yaw) in enumerate(zip(x_ego, y_ego, yaw_ego)):
                if seg_id == len(x_ego) - 1 and self.__has_successor(lane):
                    continue

                segment_name = f"{road_id}:{lane_ids[idx]}:{seg_id}"
                lane_segments[segment_name] = {"pose": (x, y), "feats": [x, y, yaw, 0.0, 0.0, seg_id < len(x_ego) - 1 or self.__has_successor(lane)]}

                if previous_seg:
                    internal_edges.append((previous_seg, segment_name))

                previous_seg = segment_name

        return lane_segments, internal_edges

    def resample_midline(self, x: List[float], y: List[float], max_dist: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.stack([x, y], axis=1)
        dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)

        if cum_dists[-1] == 0.0:
            # Single point: no yaw possible
            return np.array(x), np.array(y), np.array([0.0])

        num_points = int(np.ceil(cum_dists[-1] / max_dist)) + 1
        new_d = np.linspace(0.0, cum_dists[-1], num_points)

        x_new = np.interp(new_d, cum_dists, x)
        y_new = np.interp(new_d, cum_dists, y)

        # Compute yaw from one point to the next
        dx = np.diff(x_new)
        dy = np.diff(y_new)
        yaw = np.arctan2(dy, dx)

        # Append last yaw by repeating the second last one
        yaw = np.append(yaw, yaw[-1])

        return x_new, y_new, yaw

    def world_to_ego_frame(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform (x, y, yaw) world-frame coordinates to ego-frame coordinates.

        Args:
            x (np.ndarray): World x coordinates.
            y (np.ndarray): World y coordinates.
            yaw (np.ndarray): World yaw angles (in radians).

        Returns:
            Tuple of (x_ego, y_ego, yaw_ego)
        """
        assert self.agent is not None, "Ego pose (self.agent) must be set before calling this method."

        cx, cy, chead = self.agent
        # Translate to ego position
        dx = x - cx
        dy = y - cy

        # Rotate into ego frame
        cos_h, sin_h = np.cos(-chead), np.sin(-chead)
        x_ego = cos_h * dx - sin_h * dy
        y_ego = sin_h * dx + cos_h * dy

        # Adjust yaw to ego frame
        yaw_ego = normalise_angle(yaw - chead)

        return x_ego, y_ego, yaw_ego

    def get_edges(self, road, lanes, lane_ids):
        edges = []
        for idx, lane in enumerate(lanes):
            if lane.link and lane.link.successor:
                for succ in lane.link.successor:
                    last_seg_id = self.get_last_seg_id(*lane.midline.xy, self.POLYLINE_LENGTH) - 1
                    edges.append((
                        f"{road.id}:{lane_ids[idx]}:{last_seg_id}",
                        f"{succ.parent_road.id}:{succ.id}:0"
                    ))
        return edges

    def get_last_seg_id(self, x: List[float], y: List[float], max_dist: float) -> int:
        coords = np.stack([x, y], axis=1)

        # Compute cumulative arc-length
        deltas = np.diff(coords, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        total_length = np.sum(dists)

        if total_length == 0.0:
            return 0  # Single point

        # Compute number of resampled points
        num_points = int(np.ceil(total_length / max_dist)) + 1
        return num_points - 1

    def add_lane_change_edges(self, nodes: Dict[str, dict]):
        edges = []
        for node_id in nodes:
            road_id, lane_id, seg_id = map(int, node_id.split(":"))
            for delta_lane in [-1, 1]:
                if np.sign(lane_id + delta_lane) != np.sign(lane_id):
                    continue
                neighbor_id = f"{road_id}:{lane_id + delta_lane}:{seg_id + 1}"
                if neighbor_id in nodes:
                    edges.append((node_id, neighbor_id))
        return edges

    def filter_opendrive_around_point(self, agent: Tuple[float, float, float]):
        import copy
        from shapely.geometry import Polygon, LineString

        cx, cy, heading = agent
        left, right, back, front = self.bounds

        # --- Build ROI polygon ---
        corners_local = np.array([
            [front, left], [front, right],
            [back, right], [back, left],
        ])
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)],
        ])
        corners_world = (rot @ corners_local.T).T + np.array([cx, cy])
        roi_poly = Polygon(corners_world)

        filtered_roads, clipped_midlines = [], {}

        for road in self.__orig_opendrive.roads:
            road_copy = copy.copy(road)
            road_copy._lanes = copy.copy(road.lanes)
            road_copy._lanes._lane_sections = []

            keep_road = False
            for section in road.lanes.lane_sections:
                section_copy = self._filter_lane_section_by_roi(section, road.id, roi_poly, clipped_midlines)
                if len(section_copy.all_lanes) > 1:
                    road_copy._lanes._lane_sections.append(section_copy)
                    keep_road = True

            if keep_road:
                filtered_roads.append(road_copy)



        kept_ids = {r.id for r in filtered_roads}

        # Clean road-level and lane-level links
        for road in filtered_roads:
            if road.link.predecessor and road.link.predecessor.element.id not in kept_ids:
                road.link.predecessor = None
            if road.link.successor and road.link.successor.element.id not in kept_ids:
                road.link.successor = None

            for section in road.lanes.lane_sections:
                for lane in section.all_lanes:
                    if lane.link:
                        if lane.link.successor:
                            lane.link.successor = [
                                s for s in lane.link.successor if s.parent_road.id in kept_ids
                            ] or None
                        if lane.link.predecessor:
                            lane.link.predecessor = [
                                p for p in lane.link.predecessor if p.parent_road.id in kept_ids
                            ] or None

        filtered_junctions = [
            j for j in self.__orig_opendrive.junctions
            if any(
                conn.incoming_road.id in kept_ids or conn.connecting_road.id in kept_ids
                for conn in j.connections
            )
        ]

        # Rebuild and assign filtered map
        filtered_map = OpenDrive()
        filtered_map.header = self.__orig_opendrive.header
        filtered_map._roads = filtered_roads
        filtered_map._junctions = filtered_junctions

        self._Map__opendrive = filtered_map
        super()._Map__process_header()
        super()._Map__process_road_layout()

        # --- Re-apply clipped midlines (important!) ---
        for road in self.roads.values():
            for section in road.lanes.lane_sections:
                # Keep only the lanes that have been clipped
                kept_lanes = []
                for lane in section.all_lanes:
                    key = (road.id, lane.id)
                    if key in clipped_midlines:
                        lane._midline = clipped_midlines[key]
                        kept_lanes.append(lane)
                section._lanes = kept_lanes  # Replace original lanes with filtered ones

    def _filter_lane_section_by_roi(self, section, road_id, roi_poly, clipped_midlines):
        import copy
        from shapely.geometry import LineString

        section_copy = copy.copy(section)

        # Prepare fresh containers
        left = LeftLanes()
        center = CenterLanes()
        right = RightLanes()

        # Inside _filter_lane_section_by_roi
        has_center = False

        for lane in section.all_lanes:
            coords = list(zip(*lane.midline.xy))
            if not coords:
                continue

            midline_geom = LineString(coords)
            clipped = midline_geom.intersection(roi_poly)

            if isinstance(clipped, LineString) and len(clipped.coords) >= 2:
                lane_copy = copy.copy(lane)
                lane_copy._midline = LineString(clipped.coords)
                clipped_midlines[(road_id, lane.id)] = lane_copy._midline

                if lane.id < 0:
                    left._lanes.append(lane_copy)
                elif lane.id == 0:
                    center._lanes.append(lane_copy)
                    has_center = True
                else:
                    right._lanes.append(lane_copy)
            elif lane.id == 0:
                # Always keep center lane (even if it's outside ROI)
                lane_copy = copy.copy(lane)
                center._lanes.append(lane_copy)
                has_center = True

        # Ensure at least one center lane exists (fallback)
        if not has_center:
            print(f"WARNING: LaneSection on road {road_id} had no center lane. Inserting dummy.")
            dummy_center = copy.copy(section.center_lanes[0]) if section.center_lanes else Lane()
            center._lanes.append(dummy_center)

        # Replace original lane containers
        section_copy._left_lanes = left
        section_copy._center_lanes = center
        section_copy._right_lanes = right

        return section_copy

    def get_map_representation(self):
        node_id_mapping = {}
        lane_node_feats = np.array([node["feats"] for node in self.nodes.values()])
        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, Dataset.POLYLINE_LENGTH, 6)
        
        s_next, edge_type = self.get_edge_lookup()

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            's_next': s_next,
            'edge_type': edge_type
        }

        return map_representation

    def get_edge_lookup(self):
        node_ids = list(self.nodes.keys())
        id_to_idx = {seg_id: idx for idx, seg_id in enumerate(node_ids)}
        N = len(node_ids)

        s_next = np.zeros((self.max_nodes, self.max_nodes + 1))
        edge_type = np.zeros((self.max_nodes, self.max_nodes + 1), dtype=int)

        for edge in self.edges:
            start, end = edge
            start_road, start_lane, start_seg = start.split(":")
            end_road, end_lane, end_seg = end.split(":")

            nbr_idx = self.first_zero_index(s_next[id_to_idx[start]])
            s_next[id_to_idx[start], nbr_idx] = id_to_idx[end]
            edge_type[id_to_idx[start], nbr_idx] = 1 if start_road == end_road else 2

        s_next[:len(self.edges), -1] = np.arange(len(self.edges)) + self.max_nodes
        edge_type[:len(self.edges), -1] = 3

        return s_next, edge_type

    def get_surrounding_agent_representation(self, agent_id, agent_history):
        # Discard poses outside map extent
        vehicles = self.get_poses_inside_bounds(agent_history)
        pedestrians = [] # self.get_poses_inside_bounds(agent_history)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, self.interval + 1, 5)
        pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, self.interval * 2 + 1, 5)

        surrounding_agent_representation = {
            'vehicles': vehicles,
            'vehicle_masks': vehicle_masks,
            'pedestrians': pedestrians,
            'pedestrian_masks': pedestrian_masks
        }

        return surrounding_agent_representation

    def get_poses_inside_bounds(self, agent_history, ids=None):
        updated_pose_set = []
        updated_ids = []

        for m, (agent_id, agent_states) in enumerate(agent_history.items()):
            flag = False
            for n, agent_state in enumerate(agent_states):
                pose = agent_state[:2]
                if self.bounds[0] <= pose[0] <= self.bounds[1] and \
                        self.bounds[2] <= pose[1] <= self.bounds[3]:
                    flag = True

            if flag:
                updated_pose_set.append(list(agent_states))
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set


    @staticmethod
    def get_agent_node_masks(map_representation, surrounding_agent_representation, dist_thresh=10) -> Dict:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        lane_node_feats = map_representation['lane_node_feats']
        lane_node_masks = map_representation['lane_node_masks']
        vehicle_feats = surrounding_agent_representation['vehicles']
        vehicle_masks = surrounding_agent_representation['vehicle_masks']
        ped_feats = surrounding_agent_representation['pedestrians']
        ped_masks = surrounding_agent_representation['pedestrian_masks']

        vehicle_node_masks = np.ones((len(lane_node_feats), len(vehicle_feats)))
        ped_node_masks = np.ones((len(lane_node_feats), len(ped_feats)))

        for i, node_feat in enumerate(lane_node_feats):
            if (lane_node_masks[i] == 0).any():
                node_pose_idcs = np.where(lane_node_masks[i][:, 0] == 0)[0]
                node_locs = node_feat[node_pose_idcs, :2]

                for j, vehicle_feat in enumerate(vehicle_feats):
                    if (vehicle_masks[j] == 0).any():
                        vehicle_loc = vehicle_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - vehicle_loc, axis=1))
                        if dist <= dist_thresh:
                            vehicle_node_masks[i, j] = 0

                for j, ped_feat in enumerate(ped_feats):
                    if (ped_masks[j] == 0).any():
                        ped_loc = ped_feat[-1, :2]
                        dist = np.min(np.linalg.norm(node_locs - ped_loc, axis=1))
                        if dist <= dist_thresh:
                            ped_node_masks[i, j] = 0

        agent_node_masks = {'vehicles': vehicle_node_masks, 'pedestrians': ped_node_masks}
        return agent_node_masks

    def get_initial_node(self, map_representation) -> np.ndarray:
        """
        Returns initial node probabilities for initializing the graph traversal policy
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        """

        # Unpack lane node poses
        node_feats = map_representation['lane_node_feats']
        node_feat_lens = np.sum(1 - map_representation['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[:int(node_feat_lens[i]), :3])

        assigned_nodes = self.assign_pose_to_node(node_poses, np.asarray([0, 0, 0]), dist_thresh=3,
                                                  yaw_thresh=np.pi / 4, return_multiple=True)

        init_node = np.zeros(self.max_nodes)
        init_node[assigned_nodes] = 1/len(assigned_nodes)
        return init_node

    @staticmethod
    def assign_pose_to_node(node_poses, query_pose, dist_thresh=5, yaw_thresh=np.pi/3, return_multiple=False):
        """
        Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
        direction of motion.
        """
        dist_vals = []
        yaw_diffs = []

        for i in range(len(node_poses)):
            distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
            dist_vals.append(np.min(distances))
            idx = np.argmin(distances)
            yaw_lane = node_poses[i][idx, 2]
            yaw_query = query_pose[2]
            yaw_diffs.append(np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query)))

        idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
        idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
        idcs = np.intersect1d(idcs_dist, idcs_yaw)

        if len(idcs) > 0:
            if return_multiple:
                return idcs
            assigned_node_id = idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
        else:
            assigned_node_id = np.argmin(np.asarray(dist_vals))
            if return_multiple:
                assigned_node_id = np.asarray([assigned_node_id])

        return assigned_node_id

    @staticmethod
    def first_zero_index(arr):
        for idx, val in enumerate(arr):
            if val == 0:
                return idx
        return None  # if no zero found

    @staticmethod
    def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    def reset_opendrive(self):
        self._Map__opendrive = self.__orig_opendrive
        super()._Map__process_header()
        super()._Map__process_road_layout()

    def set_bounds(self, bounds: List[float]):
        assert bounds is not None
        self.bounds = bounds

    def __has_successor(self, lane):
        return lane.link is not None and lane.link.successor is not None

    @property
    def original_opendrive(self):
        return self.__orig_opendrive

    @property
    def graph(self):
        return self.__graph

    @property
    def nodes(self):
        return self.__graph["nodes"]

    @property
    def edges(self):
        return self.__graph["edges"]
                