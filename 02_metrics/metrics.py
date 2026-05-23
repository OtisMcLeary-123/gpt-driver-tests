"""
02_metrics — L2 Displacement Error & Collision Rate

Implements the two evaluation metrics from GPT-Driver Section 4.1:

1. L2 error (m): Average Euclidean distance between planned and ground-truth
   waypoints at each timestep (1s, 2s, 3s).

2. Collision rate (%): For each waypoint, place the ego-vehicle bounding box
   and check overlap with ground-truth object boxes. Report per-timestep rates.

Trajectory format: 6 waypoints at 0.5s intervals over 3 seconds.
Coordinate frame: ego-centric, X perpendicular, Y parallel to heading.
"""

import numpy as np


def l2_error(planned: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute per-timestep L2 displacement error.

    Args:
        planned: (6, 2) array of planned waypoints
        ground_truth: (6, 2) array of GT waypoints

    Returns:
        Dict with L2 at 1s, 2s, 3s and average.
    """
    # Waypoints at indices 1,3,5 correspond to 1s, 2s, 3s (0.5s intervals)
    dists = np.linalg.norm(planned - ground_truth, axis=1)
    return {
        "1s": float(dists[1]),
        "2s": float(dists[3]),
        "3s": float(dists[5]),
        "avg": float(np.mean(dists[[1, 3, 5]])),
    }


def box_corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    """Return 4 corners of a rotated bounding box as (4, 2) array."""
    cos, sin = np.cos(heading), np.sin(heading)
    rot = np.array([[cos, -sin], [sin, cos]])
    half = np.array([[length / 2, width / 2],
                     [-length / 2, width / 2],
                     [-length / 2, -width / 2],
                     [length / 2, -width / 2]])
    return (rot @ half.T).T + center


def boxes_collide(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """SAT (Separating Axis Theorem) collision check for two convex quads."""
    for corners in (corners_a, corners_b):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


def collision_rate(
    planned: np.ndarray,
    ego_heading: np.ndarray,
    ego_length: float,
    ego_width: float,
    objects: list[dict],
) -> dict:
    """
    Compute per-timestep collision rate.

    Args:
        planned: (6, 2) planned waypoints
        ego_heading: (6,) heading at each waypoint (radians)
        ego_length: ego vehicle length (m)
        ego_width: ego vehicle width (m)
        objects: list of dicts with keys:
            'boxes': (6, 2) predicted positions
            'headings': (6,) headings
            'length', 'width': box dimensions

    Returns:
        Dict with collision flag (0 or 1) at 1s, 2s, 3s and average.
    """
    timesteps = {0: "1s", 2: "2s", 4: "3s"}  # indices 1,3,5 for 1s,2s,3s
    # Use indices 1, 3, 5 to match 1s, 2s, 3s
    check_indices = [1, 3, 5]
    collisions = {label: 0 for label in ["1s", "2s", "3s"]}

    for idx, label in zip(check_indices, ["1s", "2s", "3s"]):
        ego_box = box_corners(planned[idx], ego_heading[idx], ego_length, ego_width)
        for obj in objects:
            obj_box = box_corners(obj["boxes"][idx], obj["headings"][idx], obj["length"], obj["width"])
            if boxes_collide(ego_box, obj_box):
                collisions[label] = 1
                break

    collisions["avg"] = np.mean([collisions["1s"], collisions["2s"], collisions["3s"]])
    return collisions


def batch_metrics(samples: list[dict]) -> dict:
    """
    Compute average L2 and collision rate over a batch of samples.

    Each sample dict has keys: 'planned', 'ground_truth', 'ego_heading',
    'ego_length', 'ego_width', 'objects'.
    """
    l2s = []
    colls = []
    for s in samples:
        l2s.append(l2_error(s["planned"], s["ground_truth"]))
        colls.append(collision_rate(
            s["planned"], s["ego_heading"], s["ego_length"], s["ego_width"], s["objects"]
        ))

    avg_l2 = {k: np.mean([x[k] for x in l2s]) for k in ["1s", "2s", "3s", "avg"]}
    avg_coll = {k: np.mean([x[k] for x in colls]) * 100 for k in ["1s", "2s", "3s", "avg"]}
    return {"l2_m": avg_l2, "collision_pct": avg_coll}
