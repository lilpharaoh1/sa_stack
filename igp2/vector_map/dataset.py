import logging
import numpy as np
import copy
from typing import Tuple, List, Dict
from shapely.geometry import Point, LineString, Polygon


from igp2.opendrive.elements.opendrive import OpenDrive
from igp2.opendrive.elements.road_lanes import LeftLanes, CenterLanes, RightLanes
from igp2.opendrive.map import Map


logger = logging.getLogger(__name__)


class Dataset(Map):
    POLYLINE_LENGTH = 10
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
            x_resampled, y_resampled = self.resample_midline(
            lane.midline.xy[0], lane.midline.xy[1], max_length
            )

            previous_seg = None
            for seg_id, (x, y) in enumerate(zip(x_resampled, y_resampled)):
                if seg_id == len(x_resampled) - 1 and self.__has_successor(lane):
                    continue

                segment_name = f"{road_id}:{lane_ids[idx]}:{seg_id}"
                lane_segments[segment_name] = {"pose": (x, y), "feats": lane_feats[idx]}

                if previous_seg:
                    internal_edges.append((previous_seg, segment_name))

                previous_seg = segment_name

        return lane_segments, internal_edges

    def resample_midline(self, x: List[float], y: List[float], max_dist: float):
        coords = np.stack([x, y], axis=1)
        dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)


        if cum_dists[-1] == 0.0:
            return np.array(x), np.array(y)


        num_points = int(np.ceil(cum_dists[-1] / max_dist)) + 1
        new_d = np.linspace(0.0, cum_dists[-1], num_points)


        return np.interp(new_d, cum_dists, x), np.interp(new_d, cum_dists, y)


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
                