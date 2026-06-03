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
    ├── src/retrace_learn/             # learned stages (Graph Builder, Proposal Policy) + runtime wrappers
    ├── src/retracemem/                # ReTrace-Engine: authorize(...) / RevisionGate / DPA
    └── data/retrace_learn/            # training/validation datasets (under v1_0/)
```

## ReTrace-Learn v1: three paper-facing stages

1. **Graph Builder** — raw dialogue / memory snapshot → candidate memory graph (learned).
2. **Proposal Policy** — candidate graph + new evidence → typed revision proposal (learned).
3. **DPA-guided RSFT / DPO** — DPA verifies/filters/ranks proposals and creates RSFT/DPO training signals (a training *protocol*, not a trainable module).

The deterministic commit path (ReTrace-Engine = `authorize(...)`, Parser, RevisionGate, DPA, Audit Trace) is an *implementation detail* of stages 2–3, not a separate paper module. DPA is a deterministic verifier and does **not** learn.

## Paper analysis code boundary (future figures/tables — not yet implemented)

Paper analysis code (JSON/JSONL → figures/tables) is a **separate concern** from
the method core, the benchmark evaluator, and the data. It is **not implemented
in this pass**; this note only fixes where it should live so future work stays
outside the core packages.

Recommended locations (pick one convention when the scripts are written):

- thin entrypoints under `scripts/`:
  `make_retrace_bench_tables.py`, `make_retrace_bench_figures.py`,
  `make_retrace_learn_tables.py`, `make_retrace_learn_figures.py`; or
- analysis packages under the paper dirs:
  `papers/retrace_bench/analysis/`, `papers/retrace_learn/analysis/`.

Rules for analysis scripts:

- **consume** JSON/JSONL outputs (e.g. under `outputs/`), never raw benchmark data;
- **emit** CSV/LaTeX/PDF/SVG/PNG under `papers/<track>/figures/` or `papers/<track>/tables/`;
- must **not** modify ReTrace-Bench data, the scorer, or any scoring semantics;
- must **not** live inside `benchmark/retrace_bench/`, `src/retrace_learn/`, or `src/retracemem/`.

## Explanations

- **`src/retrace_learn/`**: The learned method layer. Owns the Graph Builder and Proposal Policy models, datasets, SFT, and DPA-guided training logic.
- **`src/retracemem/`**: The deterministic ReTrace-Engine. Runs `authorize(...)`, `RevisionGate`, and `DPA` (verifier only — not learned).
- **`benchmark/retrace_bench/`**: Evaluation-only code, metrics, baseline scoring logic.
- **`data/retrace_bench/`**: The four ReTrace-Bench v1.0 **evaluation-only** splits — `main_3000_en`, `hard_300_en`, `realistic_100_en`, `calibration_80_en`. Never used as ReTrace-Learn training data.
- **`data/retrace_learn/v1_0/`**: Where future clean SFT/validation/DPA-preference datasets for training ReTrace-Learn policies should live (not benchmark tests). The current `src/retrace_learn/data/build_synthetic_raw_dialogue.py` is a smoke/sanity generator only, not this large-scale corpus.
- **`release/huggingface/`**: Release-package snapshots generated from `data/`. It is not a repository source of truth.
- **Trilogy Plan**: The high-level research plan should be documented in `RETRACE_AGENT_MEMORY_TRILOGY.md` (TODO: pending creation of the trilogy document).
