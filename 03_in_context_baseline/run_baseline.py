"""
Run the in-context learning baseline.

Usage:
    python run_baseline.py --dry-run          # Print prompt, no API call
    python run_baseline.py                    # Requires OPENAI_API_KEY in .env or env
    python run_baseline.py --examples 2       # Use 2 few-shot examples
"""

import argparse
import json
import os
import sys

from prompt_builder import build_messages, parse_trajectory
from sample_data import EXAMPLES, TEST_SCENARIO


def call_openai(messages: list[dict], model: str = "gpt-3.5-turbo") -> str:
    """Call OpenAI Chat API and return the assistant response text."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="GPT-Driver in-context learning baseline")
    parser.add_argument("--examples", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    args = parser.parse_args()

    examples = EXAMPLES[: args.examples]
    messages = build_messages(examples, TEST_SCENARIO)

    if args.dry_run:
        print("=== DRY RUN — Messages to be sent ===\n")
        for msg in messages:
            print(f"[{msg['role'].upper()}]")
            print(msg["content"])
            print()
        print(f"Total messages: {len(messages)}")
        print(f"Few-shot examples: {args.examples}")
        return

    # Load .env if present
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Use --dry-run or set the key.", file=sys.stderr)
        sys.exit(1)

    print(f"Calling {args.model} with {args.examples} few-shot examples...")
    response = call_openai(messages, model=args.model)
    print("\n=== LLM Response ===")
    print(response)

    trajectory = parse_trajectory(response)
    if trajectory:
        print(f"\n=== Parsed Trajectory ({len(trajectory)} waypoints) ===")
        print(trajectory)
    else:
        print("\nWarning: Could not parse trajectory from response.")


if __name__ == "__main__":
    main()
