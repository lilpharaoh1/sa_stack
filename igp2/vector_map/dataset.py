import logging
import numpy as np

from typing import Union, Tuple, List, Dict, Optional
from shapely.geometry import Point

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
    POLYLINE_LENGTH = 1

    def __init__(self, opendrive: OpenDrive = None):
        """ Create a map object given the parsed OpenDrive file

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file
        """
        super().__init__(opendrive)
        self.__process_graph()

    def __process_graph(self):
        nodes = {}
        edges = []

        for road_id, road in self.roads.items():
            # print(f"Road {road_id}: {road}")
            # print(f"Lanes: {road.lanes}")
            # print(f"Lane Sections: {road.lanes.lane_sections}")     
            # print(f"Midline: {road.midline.xy}")    
            # print("-------------------------------------------------------")
            for lane_section in road.lanes.lane_sections:
                # Process nodes
                lane_ids = [lane.id for lane in lane_section.all_lanes if lane.id != 0]
                lanes = [lane for lane in lane_section.all_lanes if lane.id != 0]

                lane_flags = self.get_lane_flags(road, lanes)
                lane_nodes, internal_edges = self.split_lanes(lanes, lane_ids, lane_flags, road_id, max_length=Dataset.POLYLINE_LENGTH) 
                
                nodes = nodes | lane_nodes  
                edges.extend(internal_edges)

                # Process edges
                external_edges = self.get_edges(road, lanes, lane_ids)
                # external_edges = self.clean_external_edges(external_edges, internal_edges)
                edges.extend(external_edges)
        
        # for edge in edges:
        #     print(edge)

        self.__graph = {
            "nodes": nodes,
            "edges": edges
        }
            
    def get_lane_flags(self, road, lanes):
        flags = []
        for lane_num, lane in enumerate(lanes):
            flags.append({
                "type": lane.type,
                "junction": road.junction != -1,
            })
        
        return flags

    def split_lanes(self, lanes, lane_ids, lane_feats, road_id, max_length=1.0): # EMRAN split length might need to be the same size as PGP
        lane_segments = {}
        internal_edges = []
        for idx, lane in enumerate(lanes):
            previous_seg = None
            for seg_id, (x, y) in enumerate(zip(lane.midline.xy[0], lane.midline.xy[1])):
                if seg_id == len(lane.midline.xy[0]) - 1 and self.__has_successor(lane): # If last node in lane
                    continue

                segment_name = f"{road_id}:{lane_ids[idx]}:{seg_id}"
                lane_segments[segment_name] = {
                        "pose": (x, y),
                        "feats": lane_feats[idx]
                    }

                if not previous_seg is None:
                    # Check if single connection lane with successor, because if so don't add the link
                    # if not (len(lane.midline.xy[0]) == 2 and self.__has_successor(lane)):
                    internal_edges.append((previous_seg, segment_name))
                previous_seg = segment_name
                
        return lane_segments, internal_edges

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
                        successor_road_id, successor_lane_id = succ.parent_road.id, succ.id
                        edges.append((f"{road.id}:{lane_ids[idx]}:{len(lane.midline.xy[0]) - 2}", f"{successor_road_id}:{successor_lane_id}:0"))

        return edges
    
    def clean_external_edges(self, external_edges, internal_edges):
        cleaned_edges = []
        for external_edge in external_edges:
            external_start_road, external_start_lane, external_start_segment = external_edge[0].split(':')
            external_end_road, external_end_lane, external_end_segment = external_edge[1].split(':')
            # Delete last link
            pass
            # Delete last node
            pass
            # Add connection
            new_external_edge = (f"{external_start_road}:{external_start_lane}:", external_edge[1])


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
                