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

## Data Files

- Hugging Face `test`: viewer-friendly flattened form of
  `data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl` (800
  scenarios).
- Hugging Face `validation`: viewer-compatible calibration split from
  `data/retrace_bench/sample_80_hard_en/scenarios.jsonl` (80 scenarios). This
  split is for quickstarts and viewer compatibility, not model/checkpoint
  selection.
- Hugging Face `train`: supervision pool from
  `data/retrace_supervision/train_3000_en/scenarios.jsonl` (3000 scenarios).
- Hugging Face `dev`: selection pool from
  `data/retrace_supervision/dev_400_en/scenarios.jsonl` (400 scenarios).

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

The public HF release exposes the held-out test split, calibration split, and
synthetic supervision / selection pools. Only `test` is the canonical held-out
benchmark split; `validation` is a quickstart/calibration split, and `train` /
`dev` are supervision and selection pools. Hidden labels are deterministic and
auditable, which is intentional for reproducibility.

## License

The Hugging Face dataset release uses CC-BY-4.0.
