# MemPatch

Benchmark and revision module for post-admission validity revision in persistent agent memory.

| Component | Location |
|-----------|----------|
| **MemPatch-Bench** (scorer + generator) | `benchmark/` |
| **Revision module** (Path A) | `mempatch/revision/` |
| **DPA kernel** (deterministic verifier) | `mempatch/dpa/` |
| **Tests** | `mempatch/tests/` |
| **Linux CUDA pipeline** | `scripts/linux/` |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset (v1.3)

| Split | Rows | Difficulty | Use |
|-------|-----:|:----------:|-----|
| `train` | 3500 | L3 | SFT + fixed 80/20 val partition |
| `test` | 500 | L4 | Held-out final eval |

Regenerate locally:

```bash
python scripts/data/generate_mempatch.py --full --out-dir local/data/mempatch
python scripts/data/package_mempatch_release.py \
  --input-dir local/data/mempatch --out-dir local/data/mempatch --validate
```

Or load from the public host: `Sylvan-Vale-Moon/MemPatch`

## Evaluate

```bash
python scripts/evaluate_mempatch_predictions.py \
  --data local/data/mempatch/test/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.api import load_scenarios, load_predictions, evaluate_predictions
result = evaluate_predictions(
    load_scenarios("local/data/mempatch/test/scenarios.jsonl"),
    load_predictions("predictions.jsonl"),
    strict=True,
)
print(result["headline_metrics"])
```

## Paper reproduction (Linux CUDA)

```bash
export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
bash scripts/linux/run_experiment.sh formal
```

See `scripts/linux/README.md` for the full pipeline.

## Citation

If you use MemPatch-Bench or this code, please cite the accompanying paper (bibtex to be added upon publication).
