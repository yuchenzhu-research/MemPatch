# Scripts

Thin CLIs that wire inference/training to `benchmark.api`. After `pip install -e ".[dev]"`, no manual `PYTHONPATH` is required.

| Script | Role |
|--------|------|
| `workflows/evaluate_mempatch_predictions.py` | **Score** any `predictions.jsonl` |
| `workflows/run_kfold_train.sh` | **Train** Path B LoRA (unique `RUN_ID` per run) |
| `workflows/run_eval_test.sh` | **Eval** LoRA on test split + print metrics |
| `workflows/audit_decision_boundary.py` | Dataset gate before release |
| `data/generate_mempatch.py` | Regenerate scenario JSONL |
| `data/package_mempatch_release.py` | Manifest + checksums |
| `data/prepare_mempatch_v13_smoke.py` | SFT bundle + MLX LoRA yaml |
| `data/build_paper_eval_bundle.py` | Test-split SFT bundle |
| `eval/run_lora_test_eval.py` | MLX inference + metrics JSON |

```bash
# Train
RUN_ID=full256 KFOLD_FOLD=0 bash scripts/workflows/run_kfold_train.sh qwen3_14b

# Score test split (after generating local/data/mempatch)
ADAPTER=local/adapters/qwen3_14b_pathB_lora/fold0/full256 \
  bash scripts/workflows/run_eval_test.sh

# Score existing predictions
python scripts/workflows/evaluate_mempatch_predictions.py \
  --data local/data/mempatch/test/scenarios.jsonl \
  --predictions path/to/predictions.jsonl --print-table
```
