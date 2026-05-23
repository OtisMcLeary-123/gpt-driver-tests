"""
04_cot_ablation — Chain-of-Thought Ablation Study

Compares two prompt strategies from GPT-Driver (Section 3.3):
  A) WITH reasoning: LLM outputs Notable Objects → Potential Effects → Meta Action → Trajectory
  B) TRAJECTORY-ONLY: LLM outputs just the 6 waypoints directly

The paper shows that CoT reasoning improves trajectory precision and interpretability.

Usage:
    python run_ablation.py --dry-run       # Print both prompts
    python run_ablation.py                 # Call API with both variants
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03_in_context_baseline"))
from prompt_builder import format_scenario, parse_trajectory
from sample_data import EXAMPLES, TEST_SCENARIO

# --- Prompt variant A: WITH chain-of-thought reasoning ---
SYSTEM_WITH_COT = """\
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
Output format:
- Notable Objects: <identify critical objects>
- Potential Effects: <how they affect driving>
- Meta Action: <high-level decision>
- Trajectory: [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]"""

# --- Prompt variant B: TRAJECTORY-ONLY (no reasoning) ---
SYSTEM_TRAJ_ONLY = """\
Role: You are the brain of an autonomous vehicle. Plan a safe 3-second driving trajectory. Avoid collisions with other objects.
Context
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0).
- Objective: Create a 3-second route using 6 waypoints, one every 0.5 seconds.
Inputs
1. Perception & Prediction: Info about surrounding objects and their predicted movements.
2. Ego-States: Your current state including velocity, heading angular velocity, acceleration.
3. Historical Trajectory: Your past 2-second route, given by 4 waypoints.
4. Mission Goal: High-level goal for the next 3 seconds.
Output ONLY the trajectory as a list of 6 waypoints, nothing else:
[(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]"""

# Few-shot response formats
COT_RESPONSE_TEMPLATE = """\
Notable Objects:
- {obj_desc}
Potential Effects:
- {effect}
Meta Action: {action}
Trajectory: [{traj}]"""

TRAJ_ONLY_RESPONSE_TEMPLATE = "[{traj}]"


def build_cot_example_response(ex: dict) -> str:
    """Build a CoT-style response for a few-shot example."""
    obj = ex["scenario"]["objects"][0]
    traj_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in ex["trajectory"])
    return COT_RESPONSE_TEMPLATE.format(
        obj_desc=f"{obj['class']} at ({obj['x']:.2f},{obj['y']:.2f}) moving toward ego path",
        effect="May enter ego lane within planning horizon",
        action=f"Proceed {ex['scenario']['goal'].lower()}, maintain safe distance",
        traj=traj_str,
    )


def build_traj_only_response(ex: dict) -> str:
    """Build a trajectory-only response for a few-shot example."""
    traj_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in ex["trajectory"])
    return TRAJ_ONLY_RESPONSE_TEMPLATE.format(traj=traj_str)


def build_variant_messages(variant: str, examples: list, query: dict) -> list[dict]:
    """Build messages for a given variant ('cot' or 'traj_only')."""
    if variant == "cot":
        system = SYSTEM_WITH_COT
        resp_fn = build_cot_example_response
    else:
        system = SYSTEM_TRAJ_ONLY
        resp_fn = build_traj_only_response

    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": format_scenario(ex["scenario"])})
        messages.append({"role": "assistant", "content": resp_fn(ex)})
    messages.append({"role": "user", "content": format_scenario(query)})
    return messages


def call_openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=400,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="CoT ablation: reasoning vs trajectory-only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()

    examples = EXAMPLES[: args.examples]

    for variant in ("cot", "traj_only"):
        messages = build_variant_messages(variant, examples, TEST_SCENARIO)
        label = "WITH CoT reasoning" if variant == "cot" else "TRAJECTORY-ONLY"

        print(f"\n{'='*60}")
        print(f"  Variant: {label}")
        print(f"{'='*60}")

        if args.dry_run:
            for msg in messages:
                print(f"\n[{msg['role'].upper()}]")
                print(msg["content"])
        else:
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(env_path):
                for line in open(env_path):
                    if "=" in line and not line.startswith("#"):
                        k, v = line.strip().split("=", 1)
                        os.environ.setdefault(k, v)

            if not os.environ.get("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY not set. Use --dry-run.", file=sys.stderr)
                sys.exit(1)

            print(f"Calling {args.model}...")
            response = call_openai(messages, args.model)
            print(f"\n--- Response ---\n{response}")
            traj = parse_trajectory(response)
            print(f"\nParsed trajectory: {traj} ({len(traj)} waypoints)")


if __name__ == "__main__":
    main()
