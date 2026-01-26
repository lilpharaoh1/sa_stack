"""
Trajectory similarity metrics for maneuver recognition.

Provides functions to compare observed trajectories against candidate
maneuver trajectories, enabling more direct/interpretable recognition.
"""

import numpy as np
from typing import Tuple
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d

from igp2.core.trajectory import Trajectory, VelocityTrajectory


def path_similarity(observed: Trajectory, candidate: Trajectory,
                    method: str = "average_distance") -> float:
    """Compute similarity between two trajectory paths.

    Args:
        observed: The observed trajectory
        candidate: The candidate trajectory to compare against
        method: Similarity method - "average_distance", "hausdorff", or "endpoint"

    Returns:
        Similarity score (higher = more similar, range depends on method)
        For distance-based methods, returns negative distance (so higher = closer)
    """
    obs_path = observed.path
    cand_path = candidate.path

    if len(obs_path) < 2 or len(cand_path) < 2:
        return -np.inf

    if method == "average_distance":
        return -_average_point_distance(obs_path, cand_path)
    elif method == "hausdorff":
        return -_hausdorff_distance(obs_path, cand_path)
    elif method == "endpoint":
        return -_endpoint_distance(obs_path, cand_path)
    elif method == "frechet_approx":
        return -_frechet_distance_approx(obs_path, cand_path)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def velocity_similarity(observed: VelocityTrajectory, candidate: VelocityTrajectory) -> float:
    """Compute similarity between velocity profiles.

    Resamples both trajectories to same length and computes correlation.

    Args:
        observed: The observed trajectory with velocities
        candidate: The candidate trajectory with velocities

    Returns:
        Similarity score (higher = more similar, range [-1, 1] for correlation)
    """
    if len(observed.velocity) < 2 or len(candidate.velocity) < 2:
        return 0.0

    # Resample to common length
    n_points = min(len(observed.velocity), len(candidate.velocity), 50)

    obs_vel = _resample_1d(observed.velocity, n_points)
    cand_vel = _resample_1d(candidate.velocity, n_points)

    # Compute correlation
    if np.std(obs_vel) < 1e-6 or np.std(cand_vel) < 1e-6:
        # One or both have constant velocity - use distance instead
        return -np.mean(np.abs(obs_vel - cand_vel))

    correlation = np.corrcoef(obs_vel, cand_vel)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def combined_similarity(observed: VelocityTrajectory, candidate: VelocityTrajectory,
                        path_weight: float = 0.7, velocity_weight: float = 0.3,
                        path_method: str = "average_distance") -> float:
    """Compute combined path and velocity similarity.

    Args:
        observed: The observed trajectory
        candidate: The candidate trajectory
        path_weight: Weight for path similarity (0-1)
        velocity_weight: Weight for velocity similarity (0-1)
        path_method: Method for path similarity

    Returns:
        Combined similarity score (higher = more similar)
    """
    path_sim = path_similarity(observed, candidate, method=path_method)
    vel_sim = velocity_similarity(observed, candidate)

    # Normalize path similarity to roughly [-1, 0] range for combination
    # Typical path distances are 0-50m, so divide by 10 to get similar scale
    path_sim_normalized = path_sim / 10.0

    return path_weight * path_sim_normalized + velocity_weight * vel_sim


def trajectory_overlap_similarity(observed: Trajectory, candidate: Trajectory,
                                   tolerance: float = 2.0) -> float:
    """Compute what fraction of observed trajectory overlaps with candidate.

    Args:
        observed: The observed trajectory
        candidate: The candidate trajectory
        tolerance: Distance threshold for considering points as overlapping (meters)

    Returns:
        Overlap ratio (0-1, higher = more overlap)
    """
    obs_path = observed.path
    cand_path = candidate.path

    if len(obs_path) == 0 or len(cand_path) == 0:
        return 0.0

    # For each observed point, check if it's within tolerance of any candidate point
    overlapping = 0
    for obs_point in obs_path:
        distances = np.linalg.norm(cand_path - obs_point, axis=1)
        if np.min(distances) <= tolerance:
            overlapping += 1

    return overlapping / len(obs_path)


def _average_point_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    """Compute average distance between corresponding points on two paths.

    Resamples both paths to same length first.
    """
    n_points = min(len(path1), len(path2), 100)

    # Resample paths
    path1_resampled = _resample_path(path1, n_points)
    path2_resampled = _resample_path(path2, n_points)

    # Compute point-to-point distances
    distances = np.linalg.norm(path1_resampled - path2_resampled, axis=1)
    return float(np.mean(distances))


def _hausdorff_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    """Compute Hausdorff distance between two paths."""
    d1 = directed_hausdorff(path1, path2)[0]
    d2 = directed_hausdorff(path2, path1)[0]
    return max(d1, d2)


def _endpoint_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    """Compute distance between path endpoints."""
    start_dist = np.linalg.norm(path1[0] - path2[0])
    end_dist = np.linalg.norm(path1[-1] - path2[-1])
    return (start_dist + end_dist) / 2


def _frechet_distance_approx(path1: np.ndarray, path2: np.ndarray,
                              n_samples: int = 50) -> float:
    """Approximate Fréchet distance using discrete sampling.

    This is O(n^2) but provides a good approximation.
    """
    # Resample to fixed number of points
    p1 = _resample_path(path1, n_samples)
    p2 = _resample_path(path2, n_samples)

    # Compute distance matrix
    n = len(p1)
    m = len(p2)

    # Dynamic programming for discrete Fréchet
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = np.linalg.norm(p1[i] - p2[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        return ca[i, j]

    return c(n - 1, m - 1)


def _resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a path to a fixed number of points."""
    if len(path) < 2:
        return path

    # Compute cumulative distance along path
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-6:
        return np.tile(path[0], (n_points, 1))

    # Create interpolation functions
    t_original = cumulative_length / total_length
    fx = interp1d(t_original, path[:, 0], kind='linear', fill_value='extrapolate')
    fy = interp1d(t_original, path[:, 1], kind='linear', fill_value='extrapolate')

    # Sample at uniform intervals
    t_new = np.linspace(0, 1, n_points)
    return np.column_stack([fx(t_new), fy(t_new)])


def _resample_1d(values: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a 1D array to a fixed number of points."""
    if len(values) < 2:
        return np.full(n_points, values[0] if len(values) > 0 else 0)

    t_original = np.linspace(0, 1, len(values))
    f = interp1d(t_original, values, kind='linear', fill_value='extrapolate')
    t_new = np.linspace(0, 1, n_points)
    return f(t_new)
