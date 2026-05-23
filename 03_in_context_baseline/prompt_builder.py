"""
03_in_context_baseline — In-Context Learning Baseline (GPT-Driver Section 4.5)

Reproduces the in-context learning approach: provide few-shot driving examples
in the prompt and ask GPT to plan a trajectory for a new scenario (no fine-tuning).

Usage:
    python run_baseline.py --examples 3 [--api-key KEY | --dry-run]
"""

SYSTEM_PROMPT = """\
Role: You are the brain of an autonomous vehicle. Plan a safe 3-second driving trajectory. Avoid collisions with other objects.
Context
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0).
- Objective: Create a 3-second route using 6 waypoints, one every 0.5 seconds.
Inputs
1. Perception & Prediction: Info about surrounding objects and their predicted movements.
2. Ego-States: Your current state including velocity, heading angular velocity, acceleration.
3. Historical Trajectory: Your past 2-second route, given by 4 waypoints.
4. Mission Goal: High-level goal for the next 3 seconds.
Task
- Thought Process: Note down critical objects and potential effects from your perceptions and predictions.
- Action Plan: Detail your meta-actions based on your analysis.
- Trajectory Planning: Develop a safe and feasible 3-second route using 6 new waypoints.
Output
- Thoughts:
  - Notable Objects
  - Potential Effects
- Meta Action
- Trajectory (MOST IMPORTANT):
  - [(x1,y1), (x2,y2), ... , (x6,y6)]"""


def format_scenario(scenario: dict) -> str:
    """Format a single driving scenario into the prompt input block."""
    lines = ["Perception and Prediction:"]
    for obj in scenario["objects"]:
        lines.append(f"- {obj['class']} at ({obj['x']:.2f},{obj['y']:.2f}), "
                     f"moving to ({obj['pred_x']:.2f},{obj['pred_y']:.2f}).")

    lines.append("Ego-States:")
    ego = scenario["ego"]
    lines.append(f"- Velocity (vx,vy): ({ego['vx']:.2f},{ego['vy']:.2f})")
    lines.append(f"- Heading Angular Velocity (v_yaw): ({ego['v_yaw']:.2f})")
    lines.append(f"- Acceleration (ax,ay): ({ego['ax']:.2f},{ego['ay']:.2f})")

    hist = scenario["history"]
    hist_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in hist)
    lines.append(f"Historical Trajectory (last 2 seconds): [{hist_str}]")
    lines.append(f"Mission Goal: {scenario['goal']}")
    return "\n".join(lines)


def format_response(trajectory: list, thoughts: str = None) -> str:
    """Format a ground-truth response for few-shot examples."""
    parts = []
    if thoughts:
        parts.append(f"Thoughts:\n{thoughts}")
    traj_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in trajectory)
    parts.append(f"Trajectory:\n[{traj_str}]")
    return "\n".join(parts)


def build_messages(examples: list[dict], query: dict) -> list[dict]:
    """
    Build the OpenAI chat messages for in-context learning.

    Args:
        examples: list of few-shot examples, each with 'scenario' and 'trajectory'
        query: the test scenario to plan for

    Returns:
        List of message dicts for the OpenAI API.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in examples:
        messages.append({"role": "user", "content": format_scenario(ex["scenario"])})
        messages.append({"role": "assistant", "content": format_response(ex["trajectory"])})

    messages.append({"role": "user", "content": format_scenario(query)})
    return messages


def parse_trajectory(response_text: str) -> list[tuple]:
    """Extract trajectory coordinates from LLM response text."""
    import re
    match = re.search(r'\[([^\]]+)\]', response_text)
    if not match:
        return []
    coords = re.findall(r'\(([^)]+)\)', match.group(1))
    trajectory = []
    for c in coords:
        parts = c.split(',')
        if len(parts) == 2:
            trajectory.append((float(parts[0].strip()), float(parts[1].strip())))
    return trajectory
