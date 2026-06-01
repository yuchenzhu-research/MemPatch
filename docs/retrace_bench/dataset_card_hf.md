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

- `sample_40_en/scenarios.jsonl`
- `dev_800_en/scenarios.jsonl`
- `stress_1760_en/scenarios.jsonl`

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

The initial release uses template rendering, so linguistic style is controlled
but less diverse than fully human-written workflows. Hidden labels are
deterministic and auditable, which is intentional for reproducibility.

## License

Project license applies unless a separate release license is added at packaging
time.

