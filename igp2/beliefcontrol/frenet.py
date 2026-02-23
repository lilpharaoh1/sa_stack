"""Frenet (reference-path) coordinate frame.

Provides coordinate transforms between world (x, y) and Frenet (s, d)
frames, as well as road-boundary queries along the reference path.
"""

import logging
from typing import Tuple

import numpy as np
from shapely.geometry import LineString

from igp2.opendrive.map import Map

logger = logging.getLogger(__name__)


class FrenetFrame:
    """Coordinate transform between world (x, y) and Frenet (s, d) frames.

    The Frenet frame is defined by a reference path (sequence of waypoints):
      * **s** — arc-length distance along the path (longitudinal).
      * **d** — signed lateral offset from the path (positive = left).

    Precomputes cumulative arc lengths, unit tangents, and unit normals
    so that repeated transforms are fast.

    Args:
        waypoints: (N, 2) array of reference-path positions in world coords.
    """

    def __init__(self, waypoints: np.ndarray):
        if len(waypoints) < 2:
            raise ValueError("FrenetFrame requires at least 2 waypoints")
        self._waypoints = np.asarray(waypoints, dtype=float)
        n = len(self._waypoints)

        # Segment vectors and lengths
        diffs = np.diff(self._waypoints, axis=0)  # (N-1, 2)
        seg_lengths = np.linalg.norm(diffs, axis=1)  # (N-1,)
        seg_lengths = np.maximum(seg_lengths, 1e-9)

        # Cumulative arc length at each waypoint
        self._arc_lengths = np.zeros(n)
        self._arc_lengths[1:] = np.cumsum(seg_lengths)
        self._total_length = self._arc_lengths[-1]

        # Unit tangent per segment (N-1, 2)
        self._seg_tangents = diffs / seg_lengths[:, None]

        # Unit normal per segment (rotate tangent 90deg CCW -> points left)
        self._seg_normals = np.column_stack([
            -self._seg_tangents[:, 1],
            self._seg_tangents[:, 0],
        ])

        # Tangent angle per segment
        self._seg_angles = np.arctan2(
            self._seg_tangents[:, 1], self._seg_tangents[:, 0],
        )

    # ------------------------------------------------------------------
    # World -> Frenet
    # ------------------------------------------------------------------

    def world_to_frenet(self, x: float, y: float,
                        heading: float = None,
                        vx: float = None, vy: float = None):
        """Convert a world-frame point (and optionally heading / velocity)
        to Frenet coordinates.

        Args:
            x, y: World position.
            heading: World heading (rad). If given, a relative heading is
                returned.
            vx, vy: World velocity components. If given, longitudinal and
                lateral velocities in the Frenet frame are returned.

        Returns:
            A dict with keys ``'s'``, ``'d'``, and optionally
            ``'heading'``, ``'vs'``, ``'vd'``.
        """
        seg_idx, s, d = self._project(x, y)
        result = {'s': s, 'd': d}

        if heading is not None:
            angle = self._seg_angles[seg_idx]
            rel = (heading - angle + np.pi) % (2 * np.pi) - np.pi
            result['heading'] = rel

        if vx is not None and vy is not None:
            t = self._seg_tangents[seg_idx]
            n = self._seg_normals[seg_idx]
            result['vs'] = vx * t[0] + vy * t[1]
            result['vd'] = vx * n[0] + vy * n[1]

        return result

    def world_to_frenet_batch(self, positions: np.ndarray):
        """Convert an (M, 2) array of world positions to (M,) s and d arrays.

        Returns:
            (s_array, d_array) — each of shape (M,).
        """
        s_arr = np.empty(len(positions))
        d_arr = np.empty(len(positions))
        for i, (px, py) in enumerate(positions):
            _, s_arr[i], d_arr[i] = self._project(px, py)
        return s_arr, d_arr

    # ------------------------------------------------------------------
    # Frenet -> World
    # ------------------------------------------------------------------

    def frenet_to_world(self, s: float, d: float,
                        heading: float = None,
                        vs: float = None, vd: float = None):
        """Convert Frenet coordinates back to world frame.

        Args:
            s: Arc-length position along path.
            d: Signed lateral offset (positive = left of path).
            heading: Relative heading in Frenet frame. If given, world
                heading is returned.
            vs, vd: Longitudinal / lateral velocity. If given, world
                velocity components are returned.

        Returns:
            A dict with keys ``'x'``, ``'y'``, and optionally
            ``'heading'``, ``'vx'``, ``'vy'``.
        """
        seg_idx, ref_x, ref_y, angle = self._interpolate(s)
        t = self._seg_tangents[seg_idx]
        n = self._seg_normals[seg_idx]

        world_x = ref_x + d * n[0]
        world_y = ref_y + d * n[1]
        result = {'x': world_x, 'y': world_y}

        if heading is not None:
            result['heading'] = (heading + angle + np.pi) % (2 * np.pi) - np.pi

        if vs is not None and vd is not None:
            result['vx'] = vs * t[0] + vd * n[0]
            result['vy'] = vs * t[1] + vd * n[1]

        return result

    def frenet_to_world_batch(self, s_arr: np.ndarray,
                              d_arr: np.ndarray) -> np.ndarray:
        """Convert arrays of (s, d) to an (M, 2) world-position array."""
        out = np.empty((len(s_arr), 2))
        for i in range(len(s_arr)):
            r = self.frenet_to_world(s_arr[i], d_arr[i])
            out[i] = [r['x'], r['y']]
        return out

    # ------------------------------------------------------------------
    # Road boundaries
    # ------------------------------------------------------------------

    def road_boundaries(self, s_values: np.ndarray,
                        scenario_map: Map,
                        search_radius: float = 5.0):
        """Query drivable-lane boundaries at sampled arc-length positions.

        For each *s* value, casts a perpendicular ray through the
        corresponding path point and intersects it with drivable lane
        boundaries to find left and right offsets.

        Args:
            s_values: (K,) array of longitudinal positions along the path.
            scenario_map: Map with drivable lane information.
            search_radius: Half-length of the perpendicular ray (metres).

        Returns:
            (d_left, d_right) — each (K,) arrays.  ``d_left`` is positive,
            ``d_right`` is negative, defining the corridor
            ``d_right <= d <= d_left`` at each step.
        """
        k = len(s_values)
        d_left = np.full(k, search_radius)
        d_right = np.full(k, -search_radius)

        for i, s in enumerate(s_values):
            seg_idx, rx, ry, _ = self._interpolate(s)
            n = self._seg_normals[seg_idx]

            # Perpendicular ray endpoints
            p_left = np.array([rx + search_radius * n[0],
                               ry + search_radius * n[1]])
            p_right = np.array([rx - search_radius * n[0],
                                ry - search_radius * n[1]])
            ray = LineString([p_right, p_left])

            # Find drivable lanes near this point
            ref_pt = np.array([rx, ry])
            lanes = scenario_map.lanes_at(ref_pt, drivable_only=True,
                                          max_distance=search_radius)
            if not lanes:
                continue

            # Intersect ray with each lane boundary, track extremes
            best_left = 0.0
            best_right = 0.0
            for lane in lanes:
                if lane.boundary is None:
                    continue
                ring = lane.boundary.exterior
                inter = ray.intersection(ring)
                if inter.is_empty:
                    continue
                # Collect intersection points
                pts = []
                if inter.geom_type == 'Point':
                    pts = [np.array([inter.x, inter.y])]
                elif inter.geom_type == 'MultiPoint':
                    pts = [np.array([p.x, p.y]) for p in inter.geoms]
                elif inter.geom_type == 'LineString':
                    pts = [np.array(c) for c in inter.coords]
                elif inter.geom_type in ('GeometryCollection',
                                         'MultiLineString'):
                    for g in inter.geoms:
                        if g.geom_type == 'Point':
                            pts.append(np.array([g.x, g.y]))
                        elif g.geom_type == 'LineString':
                            pts.extend(np.array(c) for c in g.coords)

                for pt in pts:
                    # Signed distance along normal direction
                    offset = np.dot(pt - ref_pt, n)
                    if offset >= 0:
                        best_left = max(best_left, offset)
                    else:
                        best_right = min(best_right, offset)

            if best_left > 0:
                d_left[i] = best_left
            if best_right < 0:
                d_right[i] = best_right

        return d_left, d_right

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """Total arc length of the reference path."""
        return self._total_length

    @property
    def waypoints(self) -> np.ndarray:
        return self._waypoints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(self, x: float, y: float) -> Tuple[int, float, float]:
        """Project a world point onto the reference path.

        Returns:
            (seg_idx, s, d) — index of the closest segment, arc-length
            coordinate, and signed lateral offset.
        """
        pt = np.array([x, y])
        best_seg = 0
        best_s = 0.0
        best_d = 0.0
        best_dist_sq = float('inf')

        wps = self._waypoints
        for i in range(len(wps) - 1):
            a = wps[i]
            ab = wps[i + 1] - a
            seg_len_sq = np.dot(ab, ab)
            if seg_len_sq < 1e-18:
                t = 0.0
            else:
                t = np.clip(np.dot(pt - a, ab) / seg_len_sq, 0.0, 1.0)

            proj = a + t * ab
            diff = pt - proj
            dist_sq = np.dot(diff, diff)

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_seg = i
                seg_len = np.sqrt(seg_len_sq)
                best_s = self._arc_lengths[i] + t * seg_len
                # Signed lateral offset: positive = left of path
                n = self._seg_normals[i]
                best_d = np.dot(diff, n)

        return best_seg, best_s, best_d

    def _interpolate(self, s: float) -> Tuple[int, float, float, float]:
        """Interpolate a point on the path at arc-length *s*.

        Returns:
            (seg_idx, x, y, angle) — segment index, world position,
            and tangent angle at that point.
        """
        s = np.clip(s, 0.0, self._total_length)

        # Find segment: arc_lengths[seg] <= s < arc_lengths[seg+1]
        idx = np.searchsorted(self._arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self._waypoints) - 2)

        seg_start_s = self._arc_lengths[idx]
        seg_len = self._arc_lengths[idx + 1] - seg_start_s
        if seg_len < 1e-9:
            t = 0.0
        else:
            t = (s - seg_start_s) / seg_len

        a = self._waypoints[idx]
        b = self._waypoints[idx + 1]
        pt = a + t * (b - a)
        return int(idx), float(pt[0]), float(pt[1]), float(self._seg_angles[idx])
