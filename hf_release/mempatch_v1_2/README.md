---
license: cc-by-4.0
language:
- en
pretty_name: MemPatch
tags:
- agent-memory
- llm-agents
- rapid-memory-integration
- memory-revision
- evaluation
configs:
- config_name: default
  data_files:
  - split: train
    path: train/scenarios.jsonl
  - split: main
    path: main/scenarios.jsonl
  - split: hard
    path: hard/scenarios.jsonl
---

# MemPatch v1.2.0

v1.2 fixes the v1.1 label-space gap: **main** previously contained zero `ask_clarification` / `escalate` rows while **hard** did, making train-on-main / eval-on-hard policy learning invalid.

## Public splits

| split | rows | purpose |
|---|---:|---|
| `train` | 2700 | local SFT only (all five decision labels) |
| `main` | 800 | public broad eval / dev |
| `hard` | 500 | held-out adversarial probe |

**Public total: 4000 rows.** See `manifest.json` for per-split `decision_quotas`.

All three splits include `ask_clarification` and `escalate` labels (generated with `unified_renderer_v12`).

## v1.1 → v1.2 breaking changes

- `main` shrinks from 3000 → 800 rows with full five-decision coverage.
- New `train` split (2700 rows) for local SFT; not for leaderboard eval.
- `hard` keeps 500 rows but with relabeled decision mix.
- Renderer: `unified_renderer_v12` (replaces `main_final_renderer` / `hard_final_renderer` bias).

## Format

Each line in `{split}/scenarios.jsonl` is a JSON scenario with `public_input` and `hidden_gold`.
Use the official public view; do not feed `hidden_gold` to models.

## Scoring

Use the official evaluator from the GitHub repository:

```bash
python scripts/evaluate_mempatch_predictions.py \
  --data hard/scenarios.jsonl --predictions <your_predictions>.jsonl
```

## Licensing

- **Dataset:** CC BY 4.0 (see `DATASET_LICENSE.md`).
- **Code:** MIT.

## Provenance

- Release version `1.2.0`.
- Generation seed policy: `split-prefixed:v12` (see `manifest.json`).
