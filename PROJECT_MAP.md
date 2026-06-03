# ReTrace Conceptual Map

```text
ReTrace
├── ReTrace-Bench
│   ├── benchmark/retrace_bench/       # benchmark package: schema/scoring/baselines
│   ├── data/retrace_bench/            # v1.0 splits: main/hard/realistic/calibration
│   ├── docs/retrace_bench/            # benchmark docs/results
│   └── scripts/run_retrace_bench_*.py # benchmark runners
│
└── ReTrace-Learn
    ├── src/retrace_learn/             # learned modules/training/runtime wrappers
    ├── src/retracemem/                # Authorization Court / ReTrace-Engine / authorize(...)
    └── data/retrace_learn/            # training/validation datasets (under v1_0/)
```

## Explanations

- **`src/retrace_learn/`**: The learned method layer. Owns Graph Extractor and Typed Revision Proposer models, datasets, SFT, and training logic.
- **`src/retracemem/`**: The deterministic authorization layer (also referred to as the Authorization Court or ReTrace-Engine). Runs `authorize(...)`, `RevisionGate`, and `DPA`.
- **`benchmark/retrace_bench/`**: Evaluation-only code, metrics, baseline scoring logic.
- **`data/retrace_bench/`**: The four ReTrace-Bench v1.0 evaluation splits — `main_3000_en`, `hard_300_en`, `realistic_100_en`, `calibration_80_en`.
- **`data/retrace_learn/v1_0/`**: Clean SFT training, validation, and DPA preference datasets used for training ReTrace-Learn policies (not benchmark tests).
- **`release/huggingface/`**: Release-package snapshots generated from `data/`. It is not a repository source of truth.
