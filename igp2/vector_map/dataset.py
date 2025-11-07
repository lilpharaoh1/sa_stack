import logging
import numpy as np
import copy

from typing import Union, Tuple, List, Dict, Optional
from shapely.geometry import Point, LineString, Polygon

from lxml import etree

from igp2.opendrive.elements.geometry import normalise_angle
from igp2.opendrive.elements.junction import Junction, JunctionGroup
from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road import Road
from igp2.opendrive.elements.road_lanes import Lane, LaneTypes
from igp2.opendrive.parser import parse_opendrive
from igp2.opendrive.map import Map

logger = logging.getLogger(__name__)


class Dataset(Map):
    POLYLINE_LENGTH = 10
    DEFAULT_BOUNDS = [-50, 50, -20, 80]
    # DEFAULT_BOUNDS = [-15, 15, -10, 40]
    # DEFAULT_BOUNDS = [-100, 100, -100, 100]
    # DEFAULT_BOUNDS = [-10, 10, -5, 18]

    def __init__(self, opendrive: OpenDrive = None, process_graph=False, bounds=None):
        """ Create a map object given the parsed OpenDrive file

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file
        """
        super().__init__(opendrive)
        self.__orig_opendrive = OpenDrive()
        self.__orig_opendrive.header = opendrive.header
        self.__orig_opendrive._roads = list(opendrive.roads)  # Shallow copy of roads
        self.__orig_opendrive._junctions = list(opendrive.junctions)
        self.__orig_opendrive._junction_groups = list(opendrive.junction_groups)

        self.agent = None
        self.bounds = bounds if not bounds is None else Dataset.DEFAULT_BOUNDS
    
        if process_graph:
            self.__process_graph()

    def generate_graph(
        self, 
        agent: Tuple[float, float, float] = None,
        ):
        if not agent is None:
            self.agent = agent
            self.filter_opendrive_around_point(agent)
        self.__process_graph()

    def __process_graph(self):
        nodes = {}
        edges = []

        for road_id, road in self.roads.items():
            for lane_section in road.lanes.lane_sections:
                # Process nodes
                lane_ids = [lane.id for lane in lane_section.all_lanes if lane.id != 0]
                lanes = [lane for lane in lane_section.all_lanes if lane.id != 0]

                lane_feats = self.get_lane_feats(road, lanes)
                lane_nodes, internal_edges = self.split_lanes(lanes, lane_ids, lane_feats, road_id, max_length=Dataset.POLYLINE_LENGTH) 
                
                nodes = nodes | lane_nodes  
                edges.extend(internal_edges)

                # Process external edges
                external_edges = self.get_edges(road, lanes, lane_ids)
                edges.extend(external_edges)

                # Process lane change eges
                lane_change_edges = self.add_lane_change_edges(nodes)
                edges.extend(lane_change_edges)

        self.__graph = {
            "nodes": nodes,
            "edges": edges
        }
            
    def get_lane_feats(self, road, lanes):
        flags = []
        for lane_num, lane in enumerate(lanes):
            flags.append({
                "type": lane.type,
                "junction": road.junction != -1,
            })
        
        return flags

    def split_lanes(self, lanes, lane_ids, lane_feats, road_id, max_length): # EMRAN split length might need to be the same size as PGP
        lane_segments = {}
        internal_edges = []
        for idx, lane in enumerate(lanes):
            previous_seg = None
            # Resample lanes with a uniform-ish distance between them
            x_resampled, y_resampled = self.resample_midline(lane.midline.xy[0], lane.midline.xy[1], max_length)
            for seg_id, (x, y) in enumerate(zip(x_resampled, y_resampled)):
                # If last node in lane and successor exisst, forget the last node as we will link to the successor anyways
                if seg_id == len(x_resampled) - 1 and self.__has_successor(lane):
                    continue

                # Add nodes
                segment_name = f"{road_id}:{lane_ids[idx]}:{seg_id}"
                lane_segments[segment_name] = {
                        "pose": (x, y),
                        "feats": lane_feats[idx]
                    }

                # Add straight internal edges
                if not previous_seg is None:
                    internal_edges.append((previous_seg, segment_name))
                previous_seg = segment_name
                
        return lane_segments, internal_edges

    def resample_midline(self, x: List[float], y: List[float], max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample a polyline so that distance between points does not exceed `max_dist`.

        Args:
            x: list of x coordinates of the midline
            y: list of y coordinates of the midline
            max_dist: maximum allowed distance between resampled points

        Returns:
            Tuple of (resampled_x, resampled_y) as numpy arrays
        """
        coords = np.stack([x, y], axis=1)
        # Compute distances between points
        deltas = np.diff(coords, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)

        total_length = cum_dists[-1]
        if total_length == 0.0:
            return np.array(x), np.array(y)  # single-point lane

        # Number of points needed
        num_points = int(np.ceil(total_length / max_dist)) + 1
        new_distances = np.linspace(0.0, total_length, num_points)

        # Interpolate new points
        new_x = np.interp(new_distances, cum_dists, x)
        new_y = np.interp(new_distances, cum_dists, y)

        return new_x, new_y


    def get_edges(self, road, lanes, lane_ids):
        predecessor_road_id = road.link.predecessor.element.id if not road.link.predecessor is None else None 
        successor_road_id = road.link.successor.element.id if not road.link.successor is None else None

        edges = []
        for idx, lane in enumerate(lanes):
            if not lane.link is None:
                if not predecessor_road_id is None:
                    pass
                if not lane.link.successor is None:
                    for succ in lane.link.successor:
                        last_seg_id = self.get_last_seg_id(*lane.midline.xy, Dataset.POLYLINE_LENGTH) - 1
                        successor_road_id, successor_lane_id = succ.parent_road.id, succ.id
                        edges.append((f"{road.id}:{lane_ids[idx]}:{last_seg_id}", f"{successor_road_id}:{successor_lane_id}:0"))

        return edges

    def get_last_seg_id(self, x: List[float], y: List[float], max_dist: float) -> int:
        """
        Given a midline (x, y) and a max segment length, compute the last segment ID
        after resampling to ensure even spacing.

        Args:
            x: list of x coordinates of the midline
            y: list of y coordinates of the midline
            max_dist: maximum allowed distance between resampled points

        Returns:
            int: The last segment index after resampling
        """
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
        """
        Adds lateral connections to allow lane changes between adjacent same-direction lanes.
        """
        lane_change_edges = []
        for node_id, node_data in nodes.items():
            road_id, lane_id, seg_id = node_id.split(":")
            road_id = int(road_id)
            lane_id = int(lane_id)
            seg_id = int(seg_id)

            # Only check adjacent lanes (±1) in same direction
            for delta_lane in [-1, 1]:
                neighbor_lane_id = lane_id + delta_lane

                # Check if direction matches (same sign)
                if np.sign(neighbor_lane_id) != np.sign(lane_id):
                    continue

                # Check segment ahead and same segment
                # for delta_seg in [0, 1]:
                for delta_seg in [1]:
                    neighbor_seg_id = seg_id + delta_seg
                    neighbor_node_id = f"{road_id}:{neighbor_lane_id}:{neighbor_seg_id}"

                    if neighbor_node_id in nodes:
                        lane_change_edges.append((node_id, neighbor_node_id))

        return lane_change_edges

    def filter_opendrive_around_point(self, agent: Tuple[float, float, float]):
        """
        Filters an OpenDrive object using a vehicle-centric bounding box.
        Keeps roads only if at least one lane has part of its midline inside the box,
        and clips **lane midlines** to only include points inside.
        """
        import copy
        from shapely.geometry import Polygon, LineString, Point

        cx, cy, heading = agent
        left, right, back, front = self.bounds

        # Define ROI polygon
        corners_local = np.array([
            [front,  left],
            [front,  right],
            [back,   right],
            [back,   left],
        ])
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)],
        ])
        corners_world = (rot @ corners_local.T).T + np.array([cx, cy])
        roi_poly = Polygon(corners_world)

        # --- Filter roads by clipping lanes ---
        filtered_roads = []
        clipped_midlines = {}
        for road in self.__orig_opendrive.roads:
            road_copy = copy.deepcopy(road)  # <-- this is key

            keep_road = False
            for lane_section in road_copy.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    coords = list(zip(*lane.midline.xy))
                    if not coords:
                        continue

                    midline_geom = LineString(coords)
                    clipped = midline_geom.intersection(roi_poly)

                    if isinstance(clipped, LineString) and len(clipped.coords) >= 2:
                        clipped_midlines[(road.id, lane.id)] = LineString(clipped.coords)
                        keep_road = True

            if keep_road:
                filtered_roads.append(road_copy)

        kept_road_ids = {r.id for r in filtered_roads}

        # Clean road/lane links
        for road in filtered_roads:
            if road.link.predecessor and road.link.predecessor.element.id not in kept_road_ids:
                road.link.predecessor = None
            if road.link.successor and road.link.successor.element.id not in kept_road_ids:
                road.link.successor = None

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.link:
                        if lane.link.successor:
                            lane.link.successor = [
                                succ for succ in lane.link.successor
                                if succ.parent_road.id in kept_road_ids
                            ] or None
                        if lane.link.predecessor:
                            lane.link.predecessor = [
                                pred for pred in lane.link.predecessor
                                if pred.parent_road.id in kept_road_ids
                            ] or None

        # Filter junctions based on kept roads
        filtered_junctions = [
            j for j in self.__orig_opendrive.junctions
            if any(
                conn.incoming_road.id in kept_road_ids or conn.connecting_road.id in kept_road_ids
                for conn in j.connections
            )
        ]

        # Rebuild filtered OpenDrive
        filtered_map = OpenDrive()
        filtered_map.header = self.__orig_opendrive.header
        filtered_map._roads = filtered_roads
        filtered_map._junctions = filtered_junctions

        # Set and reprocess
        self._Map__opendrive = filtered_map
        super()._Map__process_header()
        super()._Map__process_road_layout()

        for (road_id, lane_id), clipped_midline in clipped_midlines.items():
            road = self.roads.get(road_id)
            if not road:
                continue
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == lane_id:
                        lane._midline = clipped_midline

    def reset_opendrive(self):
        self._Map__opendrive = self.__orig_opendrive
        super()._Map__process_header()
        super()._Map__process_road_layout()

    def set_bounds(self, bounds: List[float]):
        assert not bounds is None
        self.bounds = bounds

    def __has_successor(self, lane):
        return not lane.link is None and not lane.link.successor is None

    @property
    def graph(self):
        """ Whole lane graph """
        return self.__graph

    @property
    def nodes(self):
        """ Nodes in lane graph """
        return self.__graph["nodes"]

    @property
    def edges(self):
        """ Edges in lane graph """
        return self.__graph["edges"]
                