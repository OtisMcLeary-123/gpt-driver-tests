"""Tests for L2 error and collision rate metrics."""

import numpy as np
from metrics import l2_error, collision_rate, boxes_collide, box_corners, batch_metrics


def test_l2_perfect():
    """Zero error when planned == ground truth."""
    traj = np.array([[0, 0.5], [0, 1.0], [0, 1.5], [0, 2.0], [0, 2.5], [0, 3.0]])
    result = l2_error(traj, traj)
    assert result["1s"] == 0.0
    assert result["2s"] == 0.0
    assert result["3s"] == 0.0
    assert result["avg"] == 0.0
    print("PASS test_l2_perfect")


def test_l2_known_error():
    """Known displacement at each timestep."""
    planned = np.zeros((6, 2))
    gt = np.zeros((6, 2))
    gt[1] = [1.0, 0.0]   # 1s: error = 1.0m
    gt[3] = [0.0, 2.0]   # 2s: error = 2.0m
    gt[5] = [3.0, 4.0]   # 3s: error = 5.0m
    result = l2_error(planned, gt)
    assert abs(result["1s"] - 1.0) < 1e-9
    assert abs(result["2s"] - 2.0) < 1e-9
    assert abs(result["3s"] - 5.0) < 1e-9
    assert abs(result["avg"] - (1 + 2 + 5) / 3) < 1e-9
    print("PASS test_l2_known_error")


def test_no_collision():
    """No collision when objects are far away."""
    planned = np.array([[0, i] for i in range(6)], dtype=float)
    heading = np.zeros(6)
    obj = {"boxes": np.array([[10, i] for i in range(6)], dtype=float),
           "headings": np.zeros(6), "length": 4.0, "width": 2.0}
    result = collision_rate(planned, heading, 4.5, 2.0, [obj])
    assert result["1s"] == 0
    assert result["2s"] == 0
    assert result["3s"] == 0
    print("PASS test_no_collision")


def test_collision_overlap():
    """Collision when object overlaps ego at a timestep."""
    planned = np.array([[0, i] for i in range(6)], dtype=float)
    heading = np.zeros(6)
    # Place object right on top of ego at timestep index 3 (2s)
    obj_boxes = np.array([[10, i] for i in range(6)], dtype=float)
    obj_boxes[3] = [0.5, 3.0]  # very close to planned[3]=(0,3)
    obj = {"boxes": obj_boxes, "headings": np.zeros(6), "length": 4.0, "width": 2.0}
    result = collision_rate(planned, heading, 4.5, 2.0, [obj])
    assert result["2s"] == 1
    print("PASS test_collision_overlap")


def test_sat_no_overlap():
    """Two boxes clearly separated don't collide."""
    a = box_corners(np.array([0.0, 0.0]), 0.0, 2.0, 2.0)
    b = box_corners(np.array([5.0, 0.0]), 0.0, 2.0, 2.0)
    assert not boxes_collide(a, b)
    print("PASS test_sat_no_overlap")


def test_sat_overlap():
    """Two overlapping boxes do collide."""
    a = box_corners(np.array([0.0, 0.0]), 0.0, 2.0, 2.0)
    b = box_corners(np.array([1.0, 0.0]), 0.0, 2.0, 2.0)
    assert boxes_collide(a, b)
    print("PASS test_sat_overlap")


def test_batch_metrics():
    """Batch averaging works correctly."""
    traj = np.array([[0, i * 0.5] for i in range(6)], dtype=float)
    sample = {
        "planned": traj,
        "ground_truth": traj,
        "ego_heading": np.zeros(6),
        "ego_length": 4.5,
        "ego_width": 2.0,
        "objects": [],
    }
    result = batch_metrics([sample, sample])
    assert result["l2_m"]["avg"] == 0.0
    assert result["collision_pct"]["avg"] == 0.0
    print("PASS test_batch_metrics")


if __name__ == "__main__":
    test_l2_perfect()
    test_l2_known_error()
    test_no_collision()
    test_collision_overlap()
    test_sat_no_overlap()
    test_sat_overlap()
    test_batch_metrics()
    print("\nAll tests passed!")
