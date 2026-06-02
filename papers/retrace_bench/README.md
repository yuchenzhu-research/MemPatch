# ReTrace-Bench — Benchmark Paper Workspace

This folder is the **standalone working package for the ReTrace-Bench benchmark
paper**. It is a writing/index workspace, not a second copy of the code or data.
It is prepared for **blind review**: it names no target venue, no authors, and no
public usernames. Use the anonymized placeholders `[anonymized repository]` and
`[anonymized dataset link]` in any manuscript text.

ReTrace-Bench is an independent **benchmark / resource / evaluation** paper on
agent memory revision reliability under evolving evidence. It is method-neutral
and evaluation-only: any memory-enabled agent (LLM-only, RAG, CRUD store,
Mem0-style system, or a trained policy) can be scored on it.

## What is in this folder

- `paper_skeleton.md` — section-by-section skeleton to lift into LaTeX.
- `section_bank.md` — consolidated, paper-ready prose for each section, drawn
  from the canonical docs and cleaned of internal TODO / venue language.
- `table_plan.md` — planned tables, their source commands/files, columns, and
  what data already exists vs. is missing.
- `figure_plan.md` — planned figures, inputs, and whether they can be produced
  now.
- `artifact_checklist.md` — artifact-track readiness checklist.
- `blind_review_checklist.md` — anonymization checklist for the manuscript.

## Where the canonical artifact lives (do not duplicate here)

The authoritative code, data, and docs stay in their existing locations:

- Benchmark code (schema, scoring, baselines, public API):
  `benchmark/retrace_bench/`
- Public scoring API: `benchmark/retrace_bench/api.py`
- Official evaluator CLI: `scripts/evaluate_retrace_bench_predictions.py`
- Canonical test split: `data/retrace_bench/test_800_templateheldout_en/`
- Calibration / quickstart split: `data/retrace_bench/sample_80_hard_en/`
- Supervision pools: `data/retrace_supervision/{train_3000_en,dev_400_en}/`
- Benchmark docs: `docs/retrace_bench/`
- Hugging Face release package + card: `release/huggingface/ReTrace-Bench/`
- Example predictions + quickstart: `examples/retrace_bench/`
- Tests: `tests/retrace_bench/`

## Key supporting documents

- Benchmark write-up: `docs/retrace_bench/benchmark_paper.md`
- Dataset design: `docs/retrace_bench/dataset_design.md`
- Canonical baseline results:
  `docs/retrace_bench/baseline_results_test_800_templateheldout_en.md`
- Manual validation protocol: `docs/retrace_bench/manual_validation_protocol.md`
- Leakage / template-heldout reports:
  `docs/retrace_bench/template_lookup_test_800_templateheldout_en.md`,
  `docs/retrace_bench/template_signature_report.md`,
  `docs/retrace_bench/split_leakage_report.md`
- Submission readiness: `docs/retrace_bench/submission_readiness.md`

## Reproducing the headline numbers

```bash
# Score an external prediction file (no model, no API key):
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/test_800_templateheldout_en/ \
  --predictions <your_predictions.jsonl> \
  --out-metrics outputs/retrace_bench/your_model.metrics.json \
  --print-table

# Built-in offline baseline / ablation suite on the canonical split:
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
  --out-dir outputs/retrace_bench/ablation_test_800_templateheldout_offline \
  --max-cases 800
```

For blind review, replace any concrete repository or dataset URL in the
manuscript with `[anonymized repository]` / `[anonymized dataset link]`. The real
links live only in the public artifact docs (`benchmark/README.md`, the HF card)
and can be de-anonymized after review.
