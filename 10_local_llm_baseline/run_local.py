"""
10_local_llm_baseline — Run GPT-Driver prompts on a local open-source LLM

Runs the same in-context learning prompts from test 03/04 on a local model
(default: Qwen2.5-1.5B-Instruct) to compare against GPT-3.5.

Usage:
    python run_local.py                          # Download model & run
    python run_local.py --model Qwen/Qwen2.5-0.5B-Instruct  # Smaller model
"""

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03_in_context_baseline"))
from prompt_builder import SYSTEM_PROMPT, format_scenario, parse_trajectory
from sample_data import EXAMPLES, TEST_SCENARIO

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_chat(examples: list, query: dict) -> list[dict]:
    """Build chat messages in the same format as the OpenAI baseline."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": format_scenario(ex["scenario"])})
        traj_str = ", ".join(f"({x:.2f},{y:.2f})" for x, y in ex["trajectory"])
        messages.append({"role": "assistant", "content": f"Trajectory:\n[{traj_str}]"})
    messages.append({"role": "user", "content": format_scenario(query)})
    return messages


def run_inference(model_name: str, messages: list[dict], max_new_tokens: int = 200) -> str:
    """Load model and generate a response."""
    print(f"Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating...")
    t1 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.time() - t1
    tps = len(new_tokens) / elapsed

    print(f"Generated {len(new_tokens)} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
    return response


def main():
    parser = argparse.ArgumentParser(description="Local LLM baseline for GPT-Driver")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--examples", type=int, default=2, help="Few-shot examples")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    examples = EXAMPLES[: args.examples]
    messages = build_chat(examples, TEST_SCENARIO)

    response = run_inference(args.model, messages, args.max_tokens)

    print(f"\n{'='*60}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    print(response)

    trajectory = parse_trajectory(response)
    if trajectory:
        print(f"\nParsed trajectory ({len(trajectory)} waypoints): {trajectory}")
    else:
        print("\nWarning: Could not parse trajectory from response.")


if __name__ == "__main__":
    main()
