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

The frozen controlled public rows retain the historical schema tag
`mempatch_bench_v1.4`, while newly exported rows use
`mempatch_bench_final`. Both tags identify the same core fields and are accepted
by the release validators; v1.4 marks the auxiliary `followup_answer` required,
whereas new exports mark it optional. Frozen rows are not rewritten.

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

The paper-facing scorer reports Decision and Operation Macro-F1, exact
state-map match, per-record state accuracy, Evidence F1, and Transition-Joint.
Transition-Joint requires the decision, operation, complete state map, and
evidence set to agree on the same case. Schema, diagnosis, follow-up, unsafe
reuse, and downstream contamination remain auxiliary diagnostics.

The public API accepts five decisions, ten operations, and eight states. The
frozen controlled gold observes four, nine, and seven respectively; those
observed cardinalities describe that split rather than narrowing the compatible
API taxonomy.

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

All model runners now call the same case pipeline in `mempatch.evaluation`.
That pipeline constructs method views, parses responses, runs typed revision,
and applies the conservative Direct-or-projection branch. HF and Ollama supply
only generation adapters, so a case produces one Direct response that is shared
with MemPatch.

## Result Boundaries

Headline quantitative results use the complete `main_test_synthetic` split.
The `challenge_test_hard` split is an auxiliary stress diagnostic and may be
partial in release aggregates.

Source-backed GitHub cases are a qualitative grounding audit. Counts such as
67 / 38 / 29 / 26 / 24 report the primary failure-mode distribution; multi-label
hazard reports use a different counting convention.

## Reproduce the Frozen Main Table

The AAAI code-and-data archive contains the frozen controlled input, labels,
all 5 x 7 main prediction cells, aggregate CSVs, and the deterministic rebuild
script. From the cleanly extracted `artifacts/` directory, run:

```bash
python3 code/tools/reporting/rebuild_frozen_table3.py \
  --output results/frozen_table3_rebuild.check.json
```

The command verifies input hashes, prediction IDs and ordering, the
Direct-versus-typed MemPatch branch, and the five-checkpoint macro displayed in
Table 3. Its six main columns are Decision Macro-F1, Operation Macro-F1,
State-map Exact Match, per-record State Accuracy, Evidence F1, and
Transition-Joint exact-hit count; Diagnosis is retained only as an auxiliary
diagnostic. The Scope-only row is a no-evidence control. The rebuild does not
run model inference.

The former general reporting script was retired because it maintained a second,
incompatible metric implementation. Use `MemPatch aggregate` for ordinary
score rows and the frozen rebuild above for paper values.

## Minimal Use

```bash
pip install -e .
MemPatch generate-synthetic \
  --config configs/benchmark/synthetic.yaml \
  --output scratch/data/mempatch/synthetic/raw_internal
```

For an optional server-model smoke run:

```bash
pip install -e ".[server]"
bash tools/evaluation/run_paper_campaign.sh smoke qwen3_14b
```

## Repository Layout

- `configs/`: benchmark generation and evaluation configuration.
- `mempatch/`: the only Python package: contracts, shared evaluation pipeline,
  typed revision/DPA kernel, release logic, and deterministic scoring.
- `tools/`: thin data-release, experiment-adapter, reporting, and source-mining
  commands; model-specific code does not reimplement the benchmark pipeline.

Commands may create `scratch/` as a disposable local workspace. It is output,
not a pipeline module, and is excluded from Git together with `results/`,
`runs/`, datasets, model artifacts, and private notes.

Dataset artifacts are released under CC BY 4.0. Code and scorer utilities in
this repository are released under the MIT License.

## Notes

Do not commit local absolute paths, private tokens, private notes, local result
archives, or raw model checkpoints.
