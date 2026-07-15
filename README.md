# MemPatch-Bench

MemPatch-Bench is a research benchmark for post-admission memory revision in
persistent-memory LLM agents. It evaluates whether a system can update,
restrict, block, delete, restore, or preserve durable memory after later
evidence changes the validity of an already stored record.

The benchmark uses a shared response contract and deterministic scoring. It is
intended to diagnose memory-lifecycle behavior rather than free-form answer
quality or LLM-as-judge preferences.

The synchronized public dataset is available at
[Sylvan-Vale-Moon/MemPatch](https://huggingface.co/datasets/Sylvan-Vale-Moon/MemPatch).
Its default `controlled` configuration contains the three deterministic splits;
the structurally distinct GitHub artifact is exposed separately as
`source_backed` so that the Hugging Face Dataset Viewer can infer each schema
without cross-config casting errors.

## Benchmark Data

| Split | Cases | Role | Headline scoring |
|---|---:|---|---|
| `dev_calibration` | 500 | Prompt and implementation calibration | No |
| `main_test_synthetic` | 3,000 | Complete controlled evaluation | Yes |
| `challenge_test_hard` | 500 | Auxiliary stress diagnostic | No |
| Source-backed artifact | 184 | Human-reviewed qualitative grounding | No |

The controlled splits cover eleven fixed `ScenarioSpec` families. They provide
deterministic coverage under that family design, not template-disjoint
generalization. The source-backed artifact retains 184 of 300 mined public
GitHub candidates after automated structural screening and manual acceptance
review.

## Core Contract

A prediction row contains one response object with:

```json
{
  "answer": "short final answer/action text",
  "decision": "use_current_memory",
  "memory_operation": "REVISE",
  "memory_state": [
    {"memory_id": "mem_1", "status": "current"}
  ],
  "evidence_event_ids": ["ev_1"],
  "failure_diagnosis": "stale_memory_reuse",
  "followup_answer": "short answer after applying the memory operation"
}
```

The scorer checks schema validity, decision correctness, memory operation
correctness, exact state-map match, memory-state accuracy, evidence F1, failure
diagnosis, follow-up correctness, unsafe reuse, downstream contamination, and
strict joint success.

## Main Methods

The main synthetic evaluation compares:

- `direct_json`
- `full_context_json`
- `summary_memory_json`
- `bm25_rag_json`
- `dense_rag_json`
- `time_aware_rag_json`
- `mempatch`

`bm25_rag_json` is the historical machine key; the paper display name is
**Lexical RAG**. `mempatch` is a hybrid response path: across the frozen 15,000
main model-case rows it preserves 11,444 valid Direct JSON responses and uses
typed projection fallback for 3,556 rows.

Some provenance files retain aggregate-only `mempatch_noguard` diagnostics.
They are not one of the seven paper interfaces, have no released bottom-up
prediction set, and are not used for main-paper claims.

## Result Boundaries

Headline quantitative results use the complete `main_test_synthetic` split.
The `challenge_test_hard` split is an auxiliary stress diagnostic and may be
partial in release aggregates.

Source-backed GitHub cases are a qualitative grounding audit. Counts such as
67 / 38 / 29 / 26 / 24 report the primary failure-mode distribution; multi-label
hazard reports use a different counting convention.

## Reproduce the Frozen Main Table

The AAAI code-and-data archive contains the frozen controlled input, labels,
all 5 x 7 main prediction cells, aggregate CSVs, and the no-inference rebuild
script. From the cleanly extracted `artifacts/` directory, run:

```bash
python3 code/tools/reporting/rebuild_frozen_table3.py \
  --output results/frozen_table3_rebuild.check.json
```

The command verifies input hashes, prediction IDs and ordering, the
Direct-versus-typed MemPatch branch, and the five-checkpoint macro displayed in
Table 3. It does not run model inference.

## Quick Checks

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the deterministic scorer and schema tests:

```bash
python3 -m pytest \
  mempatch/tests/test_benchmark_final_kernel.py \
  mempatch/tests/test_benchmark_api_strict_response.py \
  mempatch/tests/test_benchmark_response_projection.py \
  mempatch/tests/test_schema_v1_contracts.py
```

## Repository Layout

- `mempatch/`: the official `MemPatch` generation/export CLI, benchmark
  contracts, response projection, and deterministic scoring.
- `configs/`: benchmark generation and evaluation configuration.
- `tools/data_release/`: public-release validation utilities.
- `tools/evaluation/`: deterministic scoring and optional model-run utilities.
- `tools/reporting/`: aggregate reporting and the frozen Table 3 rebuild.
- `tests/` and `mempatch/tests/`: release-pipeline and benchmark-kernel tests.
- `src/`: lightweight local runner utilities used by smoke scripts.

Local outputs such as `results/`, `runs/`, `scratch/`, datasets, model
artifacts, and private notes are intentionally excluded from this repository.

Dataset artifacts are released under CC BY 4.0. Code and scorer utilities in
this repository are released under the MIT License.

## Notes

Do not commit local absolute paths, private tokens, private notes, local result
archives, or raw model checkpoints.
