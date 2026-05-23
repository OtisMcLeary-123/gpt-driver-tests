# GPT-Driver Paper — Reproduction & Tests

Experiments and notes based on the paper:
**GPT-Driver: Learning to Drive with GPT** (Mao et al., NeurIPS 2023 Workshop)
arXiv: https://arxiv.org/abs/2310.01415
Official code: https://github.com/PointsCoder/GPT-Driver

## Planned tests
- `01_tokenizer_test/` — Verify the GPT tokenizer's behavior on coordinate strings (Section 3.2).
- `02_metrics/` — Implement L2 error and collision rate metrics (Section 4.1).
- `03_in_context_baseline/` — Reproduce the in-context-learning baseline (Section 4.5).
- `04_cot_ablation/` — Compare with-reasoning vs. trajectory-only prompts.
- (more to come)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes
This is an independent study/replication for learning purposes, not the official codebase.
