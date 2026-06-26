# MemPatch

MemPatch is a compact v1.4 benchmark kernel and reference revision runtime for persistent LLM-agent memory. The benchmark side is now split into three explicit stages: generate raw internal scenarios, export public/label release files, and score predictions.

Top-level layout:

- `mempatch/`: all importable code. `mempatch/benchmark/` is the benchmark kernel; `mempatch/revision/`, `mempatch/dpa/`, and `mempatch/reference_semantics/` are the reference runtime.
- `scripts/data/`: generation, packaging, and validation wrappers.
- `scripts/memory/`: memory-baseline prompt/context builders.
- `scripts/server/`: server/GPU model campaign runner, guard stress replay, validation, and analysis.
- `configs/`: benchmark and run configuration.

Minimal commands:

```bash
pip install -e ".[dev]"

MemPatch generate-synthetic \
  --output local/data/mempatch/v1.4/raw_internal \
  --quota dev_calibration=22 \
  --quota main_test_synthetic=22 \
  --quota challenge_test_hard=22

MemPatch export-release \
  --input dev_calibration=local/data/mempatch/v1.4/raw_internal/dev_calibration.jsonl \
  --input main_test_synthetic=local/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl \
  --input challenge_test_hard=local/data/mempatch/v1.4/raw_internal/challenge_test_hard.jsonl \
  --output local/data/mempatch/v1.4/release

python scripts/evaluate_mempatch_predictions.py --help
```

Blind-review paper material is under `Montreal/`.
