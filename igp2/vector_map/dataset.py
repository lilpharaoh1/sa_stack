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

    def __init__(self, opendrive: OpenDrive = None):
        """ Create a map object given the parsed OpenDrive file

        Args:
            opendrive: A class describing the parsed contents of the OpenDrive file
        """
        super().__init__(opendrive)
