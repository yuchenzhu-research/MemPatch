# MemPatch v1.3

Decision-boundary-aware benchmark for Rapid Memory Integration (RMI).

## Splits (standard ML naming)

| Split | Rows | Purpose |
|-------|-----:|---------|
| `train` | 2700 | Fine-tuning / SFT only |
| `validation` | 800 | Development eval (tune checkpoints) |
| `test` | 500 | Held-out final eval (L4-heavy) |

**Total: 4000.** Renderer: `unified_renderer_v13`. All five `expected_decision` labels present in every split.

Legacy names: `main` → validation, `hard` → test.

## Files

- `train/scenarios.jsonl`, `validation/scenarios.jsonl`, `test/scenarios.jsonl`
- `manifest.json`, `checksums.json`, `dataset_info.json`

Generate locally:

```bash
PYTHONPATH=.:src python scripts/generate_mempatch.py --full --out-dir hf_release/mempatch
PYTHONPATH=.:src python scripts/package_mempatch_release.py \
  --input-dir hf_release/mempatch --out-dir hf_release/mempatch \
  --release-version 1.3.0 --validate --report
```
