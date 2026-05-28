# Upstream Integration

This document records the canonical roles of external systems and how ReTrace
may interact with them.

`docs/upstreams/*.md` files are historical reconnaissance notes for local
upstream inspections. They are non-authoritative and are not required reading
for method work.

## Clone and License Policy

Use `reference/` as the ignored local clone location:

```text
reference/
  STALE/
  Memora/
  nemori/
  graphiti/
  ...
```

For every upstream used in a result, record:

- URL;
- branch and commit SHA;
- observed license from the actual license file;
- setup commands attempted;
- entrypoints used;
- raw output checksums;
- local wrapper or adapter commit SHA;
- patch path and rationale if a patch was unavoidable.

Do not vendor upstream code into ReTrace. Do not edit `reference/` as part of
normal ReTrace implementation.

## Primary Executable Upstreams

### STALE / CUPMem

Role:

- primary external stale-memory benchmark and closest executable CUPMem
  comparison.

Use:

- adapter and official-evaluator integration;
- CUPMem wrapper or reproduction where licensed and executable;
- provenance records for upstream commit, inputs, and outputs.

Do not inherit:

- fixed state-slot ontology as ReTrace core;
- CUPMem prompts or adjudications as ReTrace supervision;
- official examples as prompt-development data.

### Memora

Role:

- primary evolving-history evaluation harness with repeated mutation and FAMA
  scoring.

Use:

- read-only adapter;
- official evaluator invocation where possible;
- separate preservation of raw evaluator output and local summaries;
- evaluation annotations may be consumed only in explicitly labelled
  oracle-conditioned authorization diagnostics.

Do not:

- turn Memora into a method template;
- expose official evaluation evidence or labels to training/prompt tuning;
- call oracle-conditioned annotation-driven diagnostics official end-to-end
  Memora evaluation.

## Secondary or Optional Upstreams

### NEMORI

Role:

- optional comparative memory system and episode/provenance engineering
  reference.

Use:

- ingestion and memory-output interface ideas;
- optional baseline after primary paths are stable.

Do not:

- import distillation or self-organization objectives into ReTrace method
  claims.

### Graphiti / Zep

Role:

- temporal graph and provenance reference.

Use:

- source-linked temporal fact concepts;
- valid/invalid time design ideas.

Do not:

- require graph database infrastructure for Paper 1 core;
- convert ReTrace into open-world temporal KG discovery.

### TriMem

Role:

- provenance-friendly raw dialogue plus atomic fact design reference.

Use:

- source dialogue IDs and modular retrieval/store ideas.

Do not:

- make profile summarization or prompt evolution the ReTrace mechanism.

### MemoryAgentBench and LongMemEval

Role:

- optional later evaluation bridges or non-regression checks.

Use only after:

- primary STALE/Memora and Stage A/B controlled attribution paths are stable.

### Mem0

Role:

- optional engineering memory baseline and API-surface reference.

If wrapped:

- attach ReTrace provenance metadata such as evidence ids, source message ids,
  and run step ids where possible.

Do not:

- make ReTrace a Mem0 clone with different prompts.

Actionable architecture lesson:

- reduce expensive model loops by narrowing candidate neighborhoods before
  authorization and retrieval, while preserving ReTrace's evidence-edge and DPA
  method identity rather than adopting Mem0-style overwrite/add-only memory
  semantics.

### A-MEM

Role:

- optional engineering baseline or reference for links, evolution history, and
  enriched retrieval text.

Do not:

- make linked-note evolution the core authorization method.

Actionable architecture lesson:

- keep reusable memory-state construction separate from paper reproduction and
  evaluation sweeps so authorization state can be cached or persisted once per
  persona-period and reused across many questions.

### A-MAC

Role:

- admission-control related work and possible logging reference.

Do not:

- confuse write-admission scoring with reversible belief authorization.

### AgeMem

Role:

- related work for learned memory operations and RL-style memory management.

Do not:

- start RL memory-action training in Paper 1.

### MEM1

Role:

- future latent/compact-state related work.

Do not:

- reshape Paper 1 into shared latent memory-state learning.

## Clean-Room Adapter Rules

Adapters may:

- parse official inputs;
- invoke a separately cloned upstream implementation;
- normalize raw outputs into local records;
- attach provenance and manifests;
- invoke an official evaluator unchanged.

Adapters may not:

- alter official examples to improve a method;
- expose gold facts to Stage A/B during inference;
- convert external adjudications into ReTrace labels without an explicit study
  design;
- silently rewrite evaluator code or output files.

## No-Leakage Rules

Official evaluation assets must not be used for:

- prompt development;
- parser repair based on held-out failures;
- synthetic Stage C training data;
- heuristic fixture construction;
- model-selection decisions.

If Stage C is later approved, generated or human-labelled training data must be
kept separate from official test material, and contamination checks must be run
against official evaluation assets before paper-facing results are reported.

## Architecture Lessons Adopted

The convergence path adopts only architectural separation lessons from
executable upstreams:

- write/read separation: write-time evidence ingestion and authorization update
  must be separate from query-time answer evaluation;
- candidate-neighborhood narrowing before expensive authorization: retrieved
  affected beliefs define bounded local update neighborhoods;
- cached or persistent memory state should be reused across multiple queries
  instead of rebuilding authorization state per evaluation question;
- provenance-bearing evidence must remain separate from benchmark adapters and
  runners;
- CUP-Mem's staged write pipeline motivates modular ReTrace update
  orchestration, but ReTrace keeps typed evidence edges, RevisionGate, and DPA
  as the method core;
- Graphiti/Zep-style provenance separation motivates keeping ledger/store/Gate
  and DPA independent from adapters and runners, without adding graph-database
  infrastructure in this task.
