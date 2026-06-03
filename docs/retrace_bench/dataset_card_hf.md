# ReTrace-Bench General English Dataset Card

## Dataset Summary

ReTrace-Bench is a synthetic English benchmark for evaluating persistent-memory
reliability in agentic workflows. It tests whether a system can maintain,
update, suppress, restore, or reject memories using multi-event evidence traces.

## Intended Use

Use this dataset to evaluate memory-enabled agents, LLM-only agents, RAG memory,
CRUD memory, Mem0-style memory systems, and trained memory models. The benchmark
supports black-box answering, memory-state classification, evidence retrieval,
and diagnostic failure identification.

## Data Files (ReTrace-Bench v1.0)

Four paper-facing splits, published under public names `main` / `hard` /
`realistic` / `calibration` (never train / dev / validation / test):

- Hugging Face `main`: viewer-friendly flattened form of
  `data/retrace_bench/main_3000_en/scenarios.jsonl` (3000 scenarios) — controlled
  benchmark main split.
- Hugging Face `hard`: `data/retrace_bench/hard_300_en/scenarios.jsonl` (300
  scenarios) — long-context / multi-evidence stress split.
- Hugging Face `realistic`: `data/retrace_bench/realistic_100_en/scenarios.jsonl`
  (100 scenarios) — realistic-style workflow split, `annotation_status = pending`
  (gold not yet annotated; empty template under `annotations/`).
- Hugging Face `calibration`: `data/retrace_bench/calibration_80_en/scenarios.jsonl`
  (80 scenarios) — smoke / quickstart only, not for model selection or headline
  claims.

Supervision / selection pools (`data/retrace_learn/v1_0/`) are **not**
part of the public benchmark release. (Older pre-v1.0 supervision scaffolding was removed from the active tree due to leakage).

## Fields

Each record contains scenario metadata, visible workflow inputs, four public
tasks, and a hidden evaluation section. Hidden fields should be used only for
scoring and audit, not as model input.

## Domains

Software engineering, enterprise tool workflows, customer support CRM, calendar
coordination, research work, personal assistant preferences, ecommerce
recommendations, and BI analysis.

## Failure Modes

Stale reuse, under-update, over-update, conflict collapse, scope leakage, policy
violation, wrong source attribution, memory hallucination, unnecessary memory
write, failure to forget, and failure to release or restore.

## Privacy

All entities are synthetic. Identifiers use formats such as `C-1842`,
`EMP-093`, and `PROJ-A17`. The generator does not use real personal data.

## Limitations

The public HF release exposes the four v1.0 splits `main` / `hard` / `realistic`
/ `calibration`. `main` carries the primary headline results; `hard` is a
long-context stress split; `calibration` is smoke/quickstart only and must not
be used for model selection or headline claims. The `realistic` split is
**unannotated** in this release (`annotation_status = pending`) — no human
validation or public-source provenance is claimed. Hidden labels on the
synthetic splits are deterministic and auditable, which is intentional for
reproducibility. The legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

## License

The Hugging Face dataset release uses CC-BY-4.0.
