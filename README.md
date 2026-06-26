# MemPatch

MemPatch is a compact v1.4 benchmark kernel and reference revision runtime for persistent LLM-agent memory. The benchmark side is split into three explicit stages: generate raw internal scenarios, export public/label release files, and score predictions. The current synthetic track is paper-scale by default: 500 dev-calibration, 3000 main synthetic, and 500 hard challenge cases with mixed difficulty, variable structure, lifecycle memory operations, and follow-up contamination checks.

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
  --config configs/benchmark/v1.4.yaml \
  --output local/data/mempatch/v1.4/raw_internal \

MemPatch export-release \
  --input dev_calibration=local/data/mempatch/v1.4/raw_internal/dev_calibration.jsonl \
  --input main_test_synthetic=local/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl \
  --input challenge_test_hard=local/data/mempatch/v1.4/raw_internal/challenge_test_hard.jsonl \
  --output local/data/mempatch/v1.4/release

MemPatch score \
  --labels local/data/mempatch/v1.4/release/labels/main_test_synthetic.labels.jsonl \
  --predictions path/to/predictions.jsonl \
  --output local/data/mempatch/v1.4/runs/model.scores.jsonl
```

Blind-review paper material is under `Montreal/`.
