# MemPatch-Bench

MemPatch-Bench v1.4 is a benchmark kernel and reference runtime for testing
post-admission memory-state revision in LLM-agent systems. The benchmark asks a
system to read a visible memory state plus later events, decide whether durable
memory should be revised, blocked, scoped, deleted, restored, preserved, or left
unwritten, and then answer a follow-up query after that memory operation.

The scope is deliberately controlled. MemPatch-Bench does not claim to solve
agent memory and does not use LLM-as-judge for primary scoring. It provides
deterministic contracts, hidden labels, and scorer outputs for diagnosing common
memory revision failures.

## What It Tests

Most long-context QA and RAG benchmarks ask whether a model can find and answer
from relevant context. Many agent benchmarks ask whether a model can complete a
workflow. MemPatch-Bench focuses on a narrower failure point: a fact has already
entered durable memory, then later evidence changes whether that memory should
remain usable.

Post-admission memory-state revision covers cases such as:

- A newer authoritative event supersedes an old memory.
- An update is valid only for one user, workspace, repo, or session.
- Two credible sources conflict and the memory should become unresolved.
- A policy or forget request should prevent future reuse.
- A one-shot answer is allowed but should not become durable memory.
- A later release event restores a previously blocked or held memory.

The benchmark separates answer text from memory lifecycle behavior. A system can
produce a plausible answer and still fail if it cites the wrong evidence, leaves
stale memory active, updates too broadly, or contaminates a follow-up answer.

## Dataset Tracks

The v1.4 synthetic core is generated deterministically from
`configs/benchmark/v1.4.yaml`.

| Split | Count | Purpose |
|---|---:|---|
| `dev_calibration` | 500 | Prompt and method calibration. Do not use for headline test results. |
| `main_test_synthetic` | 3000 | Main synthetic evaluation split. |
| `challenge_test_hard` | 500 | Harder synthetic challenge split with more challenge cases. |

An optional real-seeded challenge pipeline mines public GitHub evidence and
normalizes accepted candidates into the same public/label contract. It is not
the primary v1.4 headline split unless a release explicitly publishes audited
accepted cases. See `docs/release/LIMITATIONS.md` for caveats.

## Contracts

Release bundles are split into model-visible public files and scorer-only label
files:

- `PublicScenario`: the model-facing JSON object. It contains
  `scenario_id`, `split`, `domain`, `workflow_context`, `public_input`,
  `tasks`, and `output_contract`. Public rows must not contain hidden gold,
  failure-mode labels, resolver traces, or expected outputs.
- `PrivateLabel`: the scorer-only JSON object keyed by `scenario_id`. It
  contains expected decisions, memory operations, memory states, evidence IDs,
  failure modes, follow-up facts, stale-answer guards, and audit metadata.
- `Prediction`: one JSONL row per scenario, with `scenario_id`, optional
  `method` and `model`, and a parsed response object. The canonical release
  field is `parsed`; `response` and flat response rows are accepted for
  compatibility.
- `ScoreRecord`: deterministic per-scenario scorer output containing schema
  validity, decision correctness, memory operation correctness, exact state-map
  match, memory-state accuracy, evidence precision/recall/F1, diagnosis
  correctness, follow-up correctness, strict joint success, unsafe reuse, and
  downstream contamination.

See `docs/release/DATA_SCHEMA.md` for full field definitions and enum values.

## Output Schema

Each prediction response must use the exact enum strings from the benchmark
contract:

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

Allowed decisions:
`use_current_memory`, `escalate`, `ask_clarification`,
`refuse_due_to_policy`, `mark_unresolved`.

Allowed memory operations:
`PRESERVE`, `REVISE`, `RESTRICT_SCOPE`, `BLOCK`, `MARK_UNRESOLVED`,
`DELETE_OR_FORGET`, `RESTORE_OR_RELEASE`, `REJECT_NEW_MEMORY`, `NO_WRITE`,
`ESCALATE`.

Allowed memory statuses:
`current`, `blocked`, `unresolved`, `out_of_scope`, `should_not_store`,
`outdated`, `deleted`, `restored`.

Failure modes:
`stale_memory_reuse`, `under_update`, `over_update`, `conflict_collapse`,
`scope_leakage`, `policy_violation`, `wrong_source_attribution`,
`memory_hallucination`, `unnecessary_memory_write`, `failure_to_forget`,
`failure_to_release_or_restore`.

## Metrics

Primary metrics are deterministic and do not use LLM-as-judge:

- `strict_joint` / `joint_revision_success`: all required revision signals are
  correct and no unsafe reuse or follow-up contamination is detected.
- `decision_correct` and paper-facing `decision_macro_f1`: correctness over the
  decision taxonomy.
- `memory_operation_correct`: exact lifecycle operation match.
- `exact_state_map` and `memory_state_accuracy`: exact and per-memory state
  correctness.
- `evidence_f1`: set F1 against the minimal supporting event IDs.
- `diagnosis_correct` / `failure_diagnosis_accuracy`: exact failure-mode match.
- `followup_answer_correct`: deterministic key-fact check for the follow-up.
- `unsafe_reuse` / `stale_reuse_rate`: reuse of stale or forbidden answer
  content.
- `downstream_contamination`: follow-up answer contains unsafe reuse patterns.
- `schema_valid` / `response_schema_compliance_rate`: required fields and enum
  values are valid.

Answer text is checked with deterministic key-fact and forbidden-content rules
where applicable. It is not judged by another model.

## Baselines And Methods

The final v1.4 synthetic matrix uses these canonical method names:

- `direct_json`: answer from the frozen public input without retrieval or
  revision. Legacy alias: `frozen_direct`.
- `full_context_json`: use the complete event trace in timestamp order. Legacy
  alias: `full_context`.
- `summary_memory_json`: deterministic chronological extractive summary instead
  of the raw event trace. Legacy alias: `summary_memory`.
- `bm25_rag_json`: deterministic lexical BM25-style event selection. Legacy
  alias: `lexical_rag`.
- `dense_rag_json`: deterministic local dense hash retrieval over visible
  event/memory text. It requires no proprietary API.
- `time_aware_rag_json`: lexical selection with a recency prior. Legacy alias:
  `time_aware_rag`.
- `mempatch_noguard`: ablation that projects proposed actions without the DPA
  guard. Legacy alias: `mempatch_no_guard`.
- `mempatch`: reference guarded revision-module path.

Exact model matrices and final paper tables should be reported from run
manifests and scorer outputs, not from README prose.

## Install

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[llm]"
pip install -e ".[mem0]"
```

## Generate And Export Data

Generate raw internal synthetic scenarios:

```bash
MemPatch generate-synthetic \
  --config configs/benchmark/v1.4.yaml \
  --output local/data/mempatch/v1.4/raw_internal
```

Export public scenarios, scorer labels, manifests, checksums, and leakage audit:

```bash
MemPatch export-release \
  --input dev_calibration=local/data/mempatch/v1.4/raw_internal/dev_calibration.jsonl \
  --input main_test_synthetic=local/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl \
  --input challenge_test_hard=local/data/mempatch/v1.4/raw_internal/challenge_test_hard.jsonl \
  --output local/data/mempatch/v1.4/release \
  --release-version v1.4.0
```

Model-facing files are written under `public/`; scorer-only files are written
under `labels/`.

## Load Public Data

```python
import json
from pathlib import Path

path = Path("local/data/mempatch/v1.4/release/public/main_test_synthetic.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

first = rows[0]
print(first["scenario_id"])
print(first["tasks"])
```

Do not include label rows, hidden gold, resolver traces, or expected outputs in
model prompts.

## Score Predictions

Create a prediction JSONL with one row per public scenario:

```json
{"scenario_id":"mp_syn_00000","method":"my_method","model":"my_model","parsed":{"answer":"...","decision":"use_current_memory","memory_operation":"REVISE","memory_state":[{"memory_id":"mp_syn_00000_m01","status":"current"}],"evidence_event_ids":["mp_syn_00000_e1"],"failure_diagnosis":"stale_memory_reuse","followup_answer":"..."}}
```

Run the label-based scorer:

```bash
MemPatch score \
  --labels local/data/mempatch/v1.4/release/labels/main_test_synthetic.labels.jsonl \
  --predictions path/to/predictions.jsonl \
  --output local/data/mempatch/v1.4/runs/my_model.scores.jsonl
```

Aggregate scores:

```bash
MemPatch aggregate \
  --scores local/data/mempatch/v1.4/runs/my_model.scores.jsonl \
  --output local/data/mempatch/v1.4/runs/my_model.aggregate.json \
  --group-by failure_mode \
  --group-by domain
```

## Reporting And Plotting

Build canonical aggregate CSVs from completed score and prediction outputs:

```bash
python scripts/reporting/build_final_aggregate.py \
  --scores-root results/v1.4/local_ollama_smoke/scores \
  --predictions-root results/v1.4/local_ollama_smoke/predictions \
  --output-dir results/v1.4/final_synthetic/aggregates \
  --allow-partial
```

Render tables and figures from aggregate CSVs only:

```bash
python scripts/reporting/make_tables.py
python scripts/reporting/make_figures.py
```

The exporters do not load models or run evaluation. If final aggregate data is
absent, they emit pending artifacts unless `--strict` is passed.

## Smoke Evaluation

For scorer-only validation, use `MemPatch score` on an existing prediction file.

For an optional local model smoke run, configure `configs/eval/local_ollama_smoke.yaml`
and run:

```bash
python scripts/local/run_ollama_smoke.py \
  --config configs/eval/local_ollama_smoke.yaml \
  --max-cases 2 \
  --resume
```

That command invokes a local model backend and may generate or export local data.
It is intended for adapter, prompt, parsing, and scorer smoke tests, not final
benchmark reporting.

## Reproduce Paper-Scale Results

To reproduce the v1.4 synthetic benchmark scale:

1. Use the exact release version, commit, config, and dataset checksums from the
   paper run manifest.
2. Generate and export the synthetic release from `configs/benchmark/v1.4.yaml`.
3. Run each model/method on `main_test_synthetic` and `challenge_test_hard`.
4. Keep `dev_calibration` out of headline test tables.
5. Score with the deterministic scorer and aggregate by split, domain,
   difficulty, failure mode, and expected memory operation.
6. Report parse/schema failures and run metadata alongside task metrics.

The current repository contains the benchmark kernel, local smoke path, server
runner components, and analysis utilities. Final paper-result reproduction
should follow the run manifests and published prediction artifacts for the
specific paper release.

## Repository Layout

- `mempatch/benchmark/`: generation, public export, leakage audit, contracts,
  scoring, and API compatibility.
- `mempatch/revision/`, `mempatch/dpa/`, `mempatch/reference_semantics/`:
  reference revision runtime components.
- `scripts/memory/`: context and memory-baseline helpers.
- `scripts/server/`: server model campaign runner, validation, and analysis.
- `scripts/local/`: local Ollama smoke runner.
- `scripts/real_seeded/`: optional public GitHub real-seeded challenge pipeline.
- `configs/`: benchmark, smoke, and real-seeded mining configuration.
- `docs/release/`: release-facing dataset card, quickstart, schema, protocol,
  reproducibility, and limitations docs.
- `Montreal/`: blind-review paper material.

## Privacy, Leakage, And Provenance

Synthetic public rows are generated and then exported through a leakage audit
that strips hidden labels, resolver traces, failure-mode names, pattern names,
and expected outputs from model-facing files. Labels are scorer-only and should
not be used in prompts, retrieval indexes, tool state, or model memory.

The optional real-seeded pipeline is restricted to public GitHub sources and
sanitizes emails, secrets, private URLs, and sensitive security details. Real
seeded cases require public source URLs, provenance/license notes, evidence
span hashes, and audit checks before publication.

## License

Code and documentation are released under the MIT License. Dataset releases
should carry the same license unless a release manifest states otherwise.

## Cite

```bibtex
@misc{mempatchbench2026,
  title = {MemPatch-Bench v1.4: A Benchmark for Post-Admission Memory-State Revision},
  author = {MemPatch-Bench Authors},
  year = {2026},
  note = {Benchmark software and dataset release}
}
```
