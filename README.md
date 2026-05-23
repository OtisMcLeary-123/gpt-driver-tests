# GPT-Driver Paper — Reproduction & Tests

Experiments based on **GPT-Driver: Learning to Drive with GPT** (Mao et al., NeurIPS 2023 Workshop).

- Paper: https://arxiv.org/abs/2310.01415
- Official code: https://github.com/PointsCoder/GPT-Driver

## What is GPT-Driver?

GPT-Driver reformulates autonomous vehicle **motion planning as a language modeling problem**. Instead of using neural networks to predict trajectories, it converts driving scenarios into text prompts and uses GPT-3.5 to generate 6 waypoints (3 seconds, one every 0.5s). A chain-of-thought reasoning step makes the planner interpretable.

## Tests Overview

| # | Test | Runs locally? | Needs API key? |
|---|------|:---:|:---:|
| 01 | Tokenizer behavior on coordinates | ✅ | ❌ |
| 02 | L2 error & collision rate metrics | ✅ | ❌ |
| 03 | In-context learning baseline | ❌ | ✅ |
| 04 | Chain-of-thought ablation | ❌ | ✅ |
| 10 | Open-source LLM comparison | ✅ | ❌ |

---

## Test 01 — Tokenizer Behavior on Coordinates

**Goal:** Verify how GPT's tokenizer splits trajectory coordinate strings (Section 3.2).

**Key finding:** The tokenizer splits numbers at 2-3 digit boundaries, meaning each `(x, y)` coordinate costs ~10 tokens and a full 6-waypoint trajectory costs 60 tokens.

```
=== Coordinate Tokenization ===
  (0.12, 2.98)         -> ['(', '0', '.', '12', ',', ' ', '2', '.', '98', ')']  (10 tokens)
  (3.45, 18.90)        -> ['(', '3', '.', '45', ',', ' ', '18', '.', '90', ')']  (10 tokens)
  (-2.34, 0.01)        -> ['(-', '2', '.', '34', ',', ' ', '0', '.', '01', ')']  (10 tokens)

=== Precision vs Token Count ===
  2.3      -> ['2', '.', '3']       (3 tokens)
  2.34     -> ['2', '.', '34']      (3 tokens)
  2.345    -> ['2', '.', '345']     (3 tokens)
  2.3456   -> ['2', '.', '345', '6'] (4 tokens)

=== Full Trajectory Token Cost ===
  [(0.12, 2.98), (0.45, 5.67), (1.23, 8.45), (2.10, 11.30), (2.89, 14.56), (3.45, 18.90)]
  Total tokens: 60
```

**Run:** `python 01_tokenizer_test/test_tokenizer.py`

---

## Test 02 — L2 Error & Collision Rate Metrics

**Goal:** Implement the two evaluation metrics from Section 4.1.

- **L2 error (m):** Euclidean distance between planned and ground-truth waypoints at 1s, 2s, 3s.
- **Collision rate (%):** Place ego bounding box on each waypoint, check overlap with objects using SAT (Separating Axis Theorem).

```
PASS test_l2_perfect          — zero error when planned == ground truth
PASS test_l2_known_error      — correct L2 at 1s=1.0m, 2s=2.0m, 3s=5.0m
PASS test_no_collision        — no collision when objects are far
PASS test_collision_overlap   — detects collision when boxes overlap
PASS test_sat_no_overlap      — SAT correctly separates distant boxes
PASS test_sat_overlap         — SAT correctly detects overlapping boxes
PASS test_batch_metrics       — batch averaging works

All tests passed!
```

**Run:** `cd 02_metrics && python test_metrics.py`

---

## Test 03 — In-Context Learning Baseline

**Goal:** Reproduce the few-shot prompting approach (Section 4.5) — feed example driving scenarios as context and ask GPT to plan a trajectory for a new scenario.

**Prompt structure** (from the paper):
```
[SYSTEM] Role: You are the brain of an autonomous vehicle...
[USER]   Perception: car at (0.50,12.00)... Ego: velocity (0.05,4.00)... Goal: FORWARD
[ASSISTANT] Trajectory: [(x1,y1), ..., (x6,y6)]
```

**Run:**
```bash
python 03_in_context_baseline/run_baseline.py --dry-run     # preview prompt
python 03_in_context_baseline/run_baseline.py               # call API (needs OPENAI_API_KEY)
```

---

## Test 04 — Chain-of-Thought Ablation

**Goal:** Compare two prompting strategies (Section 3.3):

| Variant | What the LLM outputs |
|---------|---------------------|
| **With CoT** | Notable Objects → Potential Effects → Meta Action → Trajectory |
| **Trajectory-only** | Just `[(x1,y1), ..., (x6,y6)]` |

The paper shows CoT reasoning improves trajectory precision and adds interpretability.

**Run:**
```bash
python 04_cot_ablation/run_ablation.py --dry-run    # preview both prompts
python 04_cot_ablation/run_ablation.py              # call API (needs OPENAI_API_KEY)
```

---

## Test 10 — Open-Source LLM Comparison

**Goal:** Run the same prompts on a local open-source model (no API key needed) and compare quality against GPT-3.5.

**Model:** Qwen2.5-1.5B-Instruct (runs on CPU, ~3GB RAM)

**Result:**
```
Model: Qwen/Qwen2.5-1.5B-Instruct
Input tokens: 980
Generated 162 tokens in 31.0s (5.2 tok/s)

Thoughts:
Notable Object: Car at (0.50,12.00) moving towards us.
Potential Effect: The car could potentially collide with our vehicle.
Action Plan: Maintain current speed, increase deceleration slightly.

Trajectory:
[(0.05,4.00), (0.07,4.20), (0.09,4.40), (0.11,4.60), (0.13,4.80), (0.15,5.00)]
```

**Comparison:**

| | Qwen2.5-1.5B | GPT-3.5 (paper) |
|---|---|---|
| Follows format | ✅ | ✅ |
| Correct waypoint count | ✅ (6) | ✅ (6) |
| Trajectory quality | Overly conservative, linear | Centimeter-level (fine-tuned) |
| Reasoning quality | Basic | Detailed |
| Cost | Free (local) | API cost |

**Run:** `python 10_local_llm_baseline/run_local.py`

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For API-based tests (03, 04), set your key:
```bash
export OPENAI_API_KEY=sk-...
```

## Project Structure

```
├── 01_tokenizer_test/
│   └── test_tokenizer.py       # GPT tokenizer analysis
├── 02_metrics/
│   ├── metrics.py              # L2 error + collision rate implementation
│   └── test_metrics.py         # Unit tests
├── 03_in_context_baseline/
│   ├── prompt_builder.py       # Prompt construction + response parsing
│   ├── sample_data.py          # Few-shot driving scenarios
│   └── run_baseline.py         # CLI to run the baseline
├── 04_cot_ablation/
│   └── run_ablation.py         # CoT vs trajectory-only comparison
├── 10_local_llm_baseline/
│   └── run_local.py            # Local model inference
├── requirements.txt
└── README.md
```

## Notes

This is an independent study/replication for learning purposes, not the official codebase.
