# Open-source budget experiment plan (blind review)

Do not run closed-source flagship models on full splits until open-source Stage 3 shows signal.

## Stages

| Stage | Scope | Cost | Goal |
|-------|-------|------|------|
| **0** | `compileall` + evaluator strict tests + fixtures | $0 | Schema / projection sanity |
| **1** | `main20` + `hard20`, one cheap open model | low | Direct Response smoke |
| **2** | `main80` + `hard20`, up to 3 open models | medium | Small comparison |
| **3** | `main200` + `hard100` | medium-high | Direct vs full module vs w/o DPA |
| **4** | full `main3000` + `hard500` | high | Only if Stage 3 shows clear gain |

Closed-source flagship: run at most `main50` + `hard50` sanity **after** Stage 3, never full 3500 first.

## Primary results table (target)

| Method | Split | Decision Macro-F1 | Memory State Acc | Evidence F1 | Joint Revision Success | Stale Reuse Rate |
|--------|-------|-------------------:|-----------------:|------------:|-----------------------:|-----------------:|

## Ablation table (target)

| Method | Memory State Acc | Evidence F1 | Joint Success | Stale Reuse Rate |
|--------|-----------------:|------------:|--------------:|-----------------:|
| Full MemPatch Revision Module | | | | |
| w/o DPA-Consistent Projection | | | | |
| w/o evidence grounding | | | | |
| w/o explicit memory_state | | | | |
| Direct response only | | | | |

## Commands (Stage 0)

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
PYTHONPATH=.:src .venv/bin/python -m pytest -q

PYTHONPATH=. python scripts/evaluate_mempatch_predictions.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --predictions tests/fixtures/smoke_predictions.jsonl \
  --print-table

python scripts/run_mempatch_revision_module.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --out-predictions local/predictions/mempatch_smoke.jsonl \
  --max-cases 1
```

## Commands (Stage 1 example)

```bash
python scripts/run_mempatch_model.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --provider siliconflow \
  --model <OPEN_MODEL_NAME> \
  --out-predictions local/predictions/direct_main20.jsonl \
  --max-cases 20 \
  --resume

python scripts/run_mempatch_revision_module.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --out-predictions local/predictions/module_main20.jsonl \
  --max-cases 20 \
  --resume

PYTHONPATH=. python scripts/evaluate_mempatch_predictions.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --predictions local/predictions/direct_main20.jsonl \
  --print-table
```

Download public splits from the anonymous Hugging Face artifact referenced in `hf_release/mempatch_v1_1/manifest.json` into `local/MemPatch/`.
