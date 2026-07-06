# MemPatch-Bench

MemPatch-Bench is a research benchmark for post-admission memory revision in
persistent-memory LLM agents. It evaluates whether a system can update,
restrict, block, delete, restore, or preserve durable memory after later
evidence changes the validity of an already stored record.

The benchmark uses a shared response contract and deterministic scoring. It is
intended to diagnose memory-lifecycle behavior rather than free-form answer
quality or LLM-as-judge preferences.

## Repository Layout

- `mempatch/`: benchmark contracts, scenario generation, response projection,
  and deterministic scoring.
- `configs/`: benchmark generation and evaluation configuration.
- `tools/`: release, evaluation, reporting, and plotting utilities.
- `tests/`: release-pipeline and real-seeded smoke tests retained from the
  original repository history.
- `src/`: lightweight local runner utilities used by smoke scripts.

Local outputs such as `results/`, `runs/`, `scratch/`, datasets, model
artifacts, and private notes are intentionally excluded from this repository.

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

`mempatch_noguard` is an ablation of the guarded projection path.

## Quick Checks

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the deterministic scorer and schema tests:

```bash
python -m pytest \
  mempatch/tests/test_benchmark_final_kernel.py \
  mempatch/tests/test_benchmark_api_strict_response.py \
  mempatch/tests/test_benchmark_response_projection.py \
  mempatch/tests/test_schema_v1_contracts.py
```

## Notes

Do not commit local absolute paths, private tokens, private notes, local result
archives, or raw model checkpoints.
