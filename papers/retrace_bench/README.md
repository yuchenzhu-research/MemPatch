# ReTrace-Bench — Benchmark Paper Workspace

This folder is the **standalone working package for the ReTrace-Bench benchmark
paper**. It is a writing/index workspace, not a second copy of the code or data.

ReTrace-Bench is an independent **benchmark / resource / evaluation** paper on
agent memory revision reliability under evolving evidence. It is method-neutral
and evaluation-only: any memory-enabled agent (LLM-only, RAG, CRUD store,
Mem0-style system, or a trained policy) can be scored on it.

## What is in this folder

- `paper_skeleton.md` — section-by-section skeleton to lift into LaTeX.
- `section_bank.md` — consolidated, paper-ready prose for each section.
- `table_plan.md` — planned tables, their source commands/files, columns, and
  what data already exists vs. is missing.
- `figure_plan.md` — planned figures, inputs, and whether they can be produced
  now.
- `artifact_checklist.md` — artifact-track readiness checklist.

## Where the canonical artifact lives (do not duplicate here)

The authoritative code, data, and docs stay in their existing locations:

- Benchmark code (schema, scoring, baselines, public API):
  `benchmark/retrace_bench/`
- Public scoring API: `benchmark/retrace_bench/api.py`
- Official evaluator CLI: `scripts/evaluate_retrace_bench_predictions.py`
- ReTrace-Bench v1.0 splits (public names `main` / `hard` / `realistic` /
  `calibration`): `data/retrace_bench/{main_3000_en,hard_300_en,realistic_100_en,calibration_80_en}/`
- ReTrace-Learn method-track data (ReTrace-Bench-derived scenarios with declared
  split roles): `data/retrace_learn/` (older pre-v1.0 supervision scaffolding is
  recoverable only from Git history)
- Benchmark docs: `docs/retrace_bench/`
- Hugging Face release package + card: `release/huggingface/ReTrace-Bench/`
- Example predictions + quickstart: `examples/retrace_bench/`
- Tests: `tests/retrace_bench/`

## Key supporting documents

- Benchmark write-up: `docs/retrace_bench/benchmark_paper.md`
- Dataset design: `docs/retrace_bench/dataset_design.md`
- Manual validation protocol:
  `docs/retrace_bench/manual_validation_protocol.md`

Legacy pre-v1 artifacts were removed from the active tree; current results must
be regenerated on v1.0 splits.

## Reproducing the headline numbers

```bash
# Score an external prediction file (no model, no API key):
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/main_3000_en/ \
  --predictions <your_predictions.jsonl> \
  --out-metrics outputs/retrace_bench/your_model.metrics.json \
  --print-table

# Built-in offline baseline / ablation suite on the main split:
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --out-dir outputs/retrace_bench/ablation_main_3000_offline
```

Public artifact links live in the benchmark README and Hugging Face card; paper
drafts should use the link policy required by the submission venue.
