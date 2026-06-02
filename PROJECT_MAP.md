# ReTrace Conceptual Map

```text
ReTrace
├── ReTrace-Bench
│   ├── benchmark/retrace_bench/       # benchmark package: schema/scoring/baselines
│   ├── data/retrace_bench/            # held-out benchmark/calibration data
│   ├── data/retrace_supervision/      # supervision/dev pools for future ReTrace-Learn
│   ├── docs/retrace_bench/            # benchmark docs/results
│   └── scripts/run_retrace_bench_*.py # benchmark runners
│
└── ReTrace-Learn
    ├── src/retrace_learn/             # learned modules/training/runtime wrappers
    └── src/retracemem/                # Authorization Court / ReTrace-Engine / authorize(...)
```

## Explanations

- **`src/retrace_learn/`**: The learned method layer. Owns Graph Extractor and Typed Revision Proposer models, datasets, SFT, and training logic.
- **`src/retracemem/`**: The deterministic authorization layer (also referred to as the Authorization Court or ReTrace-Engine). Runs `authorize(...)`, `RevisionGate`, and `DPA`.
- **`benchmark/retrace_bench/`**: Evaluation-only code, metrics, baseline scoring logic.
- **`data/retrace_bench/`**: Benchmark held-out tests and calibration datasets.
- **`data/retrace_supervision/`**: Supervision/dev selection pools used for learning policies (not benchmark tests).
- **`release/huggingface/`**: A generated local packaging area. It is not a repository source of truth (canonical data lives under `data/`).
