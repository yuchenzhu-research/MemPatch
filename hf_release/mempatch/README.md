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
  - split: validation
    path: validation/scenarios.jsonl
  - split: test
    path: test/scenarios.jsonl
---

# MemPatch v1.3.0

Decision-boundary-aware benchmark for Rapid Memory Integration (RMI).

## Splits

| Split | Rows | Purpose |
|-------|-----:|---------|
| `train` | 2700 | Fine-tuning / SFT only |
| `validation` | 800 | Development eval |
| `test` | 500 | Held-out final eval (L4-heavy) |

**Total: 4000.** Renderer: `unified_renderer_v13`. All five `expected_decision` labels in every split.

Legacy names: `main` → validation, `hard` → test.

## Files

- `train/scenarios.jsonl`, `validation/scenarios.jsonl`, `test/scenarios.jsonl`
- `manifest.json`, `checksums.json`, `dataset_info.json`
