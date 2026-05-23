"""
Sample driving scenarios for in-context learning experiments.
Based on the prompt format from GPT-Driver (Figure 2).
"""

EXAMPLES = [
    {
        "scenario": {
            "objects": [
                {"class": "animal", "x": -1.93, "y": 7.00, "pred_x": -2.31, "pred_y": 10.89},
                {"class": "car", "x": -8.67, "y": 0.12, "pred_x": -8.50, "pred_y": -0.08},
                {"class": "adult", "x": -1.21, "y": 6.78, "pred_x": -1.29, "pred_y": 10.48},
            ],
            "ego": {"vx": 0.00, "vy": 1.46, "v_yaw": -0.00, "ax": 0.01, "ay": -0.15},
            "history": [(-0.00, -6.74), (-0.03, -4.73), (-0.03, -3.07), (-0.02, -1.46)],
            "goal": "RIGHT",
        },
        "trajectory": [(0.11, 1.14), (0.45, 2.28), (1.12, 3.47), (2.18, 4.54), (3.65, 5.29), (5.49, 5.58)],
    },
    {
        "scenario": {
            "objects": [
                {"class": "car", "x": 2.50, "y": 15.00, "pred_x": 2.45, "pred_y": 12.00},
                {"class": "truck", "x": -3.00, "y": 20.00, "pred_x": -3.10, "pred_y": 18.50},
            ],
            "ego": {"vx": 0.10, "vy": 5.20, "v_yaw": 0.01, "ax": -0.05, "ay": 0.30},
            "history": [(-0.02, -10.50), (-0.01, -7.80), (0.00, -5.10), (0.02, -2.50)],
            "goal": "FORWARD",
        },
        "trajectory": [(0.03, 2.60), (0.05, 5.20), (0.08, 7.80), (0.10, 10.40), (0.12, 13.00), (0.15, 15.60)],
    },
    {
        "scenario": {
            "objects": [
                {"class": "adult", "x": 1.00, "y": 4.50, "pred_x": 1.50, "pred_y": 5.00},
            ],
            "ego": {"vx": -0.20, "vy": 3.00, "v_yaw": 0.05, "ax": 0.00, "ay": -0.50},
            "history": [(0.10, -6.00), (0.05, -4.50), (0.00, -3.00), (-0.05, -1.50)],
            "goal": "LEFT",
        },
        "trajectory": [(-0.30, 1.20), (-0.80, 2.30), (-1.50, 3.30), (-2.30, 4.10), (-3.20, 4.70), (-4.10, 5.00)],
    },
]

# Test scenario (not in few-shot examples)
TEST_SCENARIO = {
    "objects": [
        {"class": "car", "x": 0.50, "y": 12.00, "pred_x": 0.48, "pred_y": 9.50},
        {"class": "bicycle", "x": -2.00, "y": 5.00, "pred_x": -2.10, "pred_y": 7.00},
    ],
    "ego": {"vx": 0.05, "vy": 4.00, "v_yaw": -0.01, "ax": 0.00, "ay": 0.20},
    "history": [(0.00, -8.00), (0.01, -6.00), (0.02, -4.00), (0.03, -2.00)],
    "goal": "FORWARD",
}
