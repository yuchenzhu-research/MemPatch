# ReTrace Companion: Codebase Integration and AI-Model Handoff Context

> **How to use this file:** Read this together with `iclr_2027_paper_1_final_blueprint_re_trace.md`. The blueprint is the scientific specification of Paper 1. This companion is an execution contract for code-writing models: it explains why the project settled on its current direction, which external repositories to inspect, what may be reused, what must not be silently altered, and how to build a reproducible implementation without drifting into a different paper.
>
> **Non-duplication rule:** This document intentionally does not restate the paper title, abstract, contribution list, mathematical formulation, benchmark selection rationale, figure/table plan, or manuscript timetable already defined in the blueprint.

---

# 1. Interpretation Contract: what a coding model must understand before writing code

## 1.1 The research history that determines the implementation

This project began from a broad interest in long-term LLM-agent memory: not merely recalling old dialogue, but deciding what earlier information remains safe to use after the user's circumstances change. Reading STALE/CUPMem shifted the project away from claiming that implicit invalidation itself is new. CUPMem already demonstrates that old state can be retired when later evidence renders it invalid, especially under stale premises.

A second line of reading then raised a deeper but much heavier question: whether raw episodes should be transformed into semantic abstractions, when consolidation is useful, and how harmful abstractions should be rolled back. That question remains important, but recent work has occupied much of the immediate space: NEMORI addresses episodic/semantic distillation; recent diagnosis work studies faulty consolidation; Auto-Dreamer learns offline consolidation from task reward; AgeMem learns tool-level memory actions. This makes consolidation or latent memory an unsuitable first implementation target for a time-bounded Paper 1.

The current project therefore takes a narrower engineering commitment: **build an auditable revision layer over evolving memory evidence, while avoiding irreversible memory rewriting and avoiding a predefined life-domain ontology.** The paper is not trying to learn all of memory. It is trying to establish a reliable mechanism for deciding when a previously supported belief may or may not be allowed to influence the present.

## 1.2 Priority rules when the two supplied files seem to disagree

A model implementing ReTrace must follow this order of authority:

1. **Scientific definitions, claims, method boundary, evaluation choices:** follow the blueprint file.
2. **Repository inspection, adapter design, reproducibility rules, task ordering, logging and clean-room integration:** follow this companion file.
3. **External repositories:** treat them as upstream sources to reproduce or wrap, not as authority for changing the ReTrace research question.
4. **Any improvised idea introduced during coding:** place it in `docs/parking_lot.md`; do not implement it in the main experiment pipeline unless explicitly approved.

## 1.3 Drift alarms: stop and ask before making these changes

A coding model must not silently turn Paper 1 into any of the following:

- an episodic-to-semantic consolidation project;
- an RL/GRPO memory-policy training project;
- a full latent-state or memory-token project;
- a new large-scale benchmark construction effort;
- a direct fork or rebranding of CUPMem;
- an LLM-judge-only system in which the judge can rewrite beliefs without preserved evidence and inspectable authorization traces;
- a Graphiti/Zep product integration whose infrastructure dominates the research method.

If a code suggestion requires one of these shifts, write a short proposal in `docs/parking_lot.md` instead of modifying the implementation plan.

---

# 2. Upstream Repository Intake Map: inspect first, wrap second, implement new modules last

All URLs below are canonical upstream locations to be checked again when cloning. At clone time, record the exact commit SHA, branch, license file, environment setup, and any local patch. Never paste external repository code into ReTrace without license inspection and attribution; prefer wrappers and adapters.

## 2.1 Tier A: executable upstreams needed for the first results

### A1. STALE / CUPMem

- **Repository:** `https://github.com/icedreamc/STALE`
- **Role in ReTrace:** nearest executable competitor and evaluation source for implicit-conflict behavior.
- **Inspect before writing any ReTrace module:**
  - root README and environment setup;
  - `STALE/` dataset/evaluation scripts and expected prediction format;
  - `cup_mem/` single-sample runner and output artifacts;
  - any files implementing schema definitions, invalidation candidate expansion, invalidation adjudication, retrieval, and premise verification.
- **Harvest into local notes:**
  - exact command that produces a score on a minimal subset;
  - input JSON fields required by CUPMem;
  - output JSON fields returned by CUPMem;
  - where API/model configuration enters the run;
  - whether a run can be repeated deterministically under fixed model outputs.
- **Integration decision:** Do not edit CUPMem code initially. Create a ReTrace-side wrapper that invokes upstream output and normalizes it. Any unavoidable patch must be stored as a patch file with reason and upstream commit SHA.
- **Do not inherit:** its state ontology, its prompt text, or its adjudication result as ground truth for ReTrace training.

### A2. Memora

- **Repository:** `https://github.com/geniesinc/Memora`
- **Role in ReTrace:** external evolving-history evaluation harness; it is not a module to embed into the method.
- **Inspect first:**
  - released `data/` hierarchy for weekly, monthly, quarterly settings;
  - question/evidence structures, particularly how useful evidence and forgetting/obsolete evidence are represented;
  - `evals/model_eval/` and `evals/agent_eval/` entry points;
  - the exact production of FAMA and any judge/model assumptions.
- **Harvest:**
  - a read-only dataset loader;
  - a scoring call or command that remains identical to upstream;
  - one tiny fixture containing a question, relevant evidence, obsolete evidence, and expected evaluator input.
- **Integration decision:** ReTrace should generate predictions in the format expected by the official evaluator rather than reimplementing FAMA early. Store upstream scoring results and local normalized summaries separately.
- **Do not inherit:** task-specific answer hints or any evaluation evidence into training data.

### A3. NEMORI

- **Repository:** `https://github.com/nemori-ai/nemori`
- **Role in ReTrace:** runnable comparative memory system and source of engineering patterns for conversation ingestion and memory search.
- **Inspect:**
  - installation and minimal end-to-end run;
  - how conversations are segmented into episodes;
  - how semantic knowledge is stored and retrieved;
  - evaluation folders for LoCoMo/LongMemEval or other released tasks;
  - exported memory objects and whether source episode identity survives distillation.
- **Harvest:** only its public run interface, memory-output format, and evaluation compatibility notes.
- **Integration decision:** Treat NEMORI as a separately runnable upstream baseline. Do not transplant its distillation objective into ReTrace; the scientific question differs.

## 2.2 Tier B: structure donors and optional baselines

### B1. Graphiti / Zep temporal graph engine

- **Repository:** `https://github.com/getzep/graphiti`
- **Role:** architectural reference for temporal provenance and evolving facts; not required as a database dependency for the first runnable ReTrace prototype.
- **Inspect:** node/edge representation, episode ingestion, source provenance, valid/invalid time handling, query interface, storage dependencies.
- **Borrow conceptually:** clean representation of source-linked temporal relationships and historical validity.
- **Do not do initially:** replace the ReTrace store with Graphiti or require Neo4j/service deployment before baseline experiments work.

### B2. TriMem

- **Repository:** `https://github.com/tmlr-group/TriMem`
- **Role:** compact reference implementation showing raw-dialogue retention plus extracted atomic facts and persona/profile memory.
- **Inspect:** `core/memory_builder.py`, `core/hybrid_retriever.py`, `core/profile_manager.py`, `database/dialogue_store.py`, `models/memory_entry.py`, and its evaluation runner.
- **Borrow conceptually:** source-dialogue IDs, separation between raw dialogue and derived memory, modular stores, simple configuration.
- **Do not import as method premise:** profile summarization and prompt-evolution are not the ReTrace scientific mechanism.

### B3. MemoryAgentBench

- **Repository:** `https://github.com/HUST-AI-HYZ/MemoryAgentBench`
- **Role:** optional third external evaluation source after the two primary evaluation pipelines work.
- **Inspect:** `FactConsolidation`/conflict-resolution subsets, dataset config format, already-integrated memory systems, output and judge paths.
- **Harvest:** adapter interface and a reproducible subset command.
- **Stop condition:** do not spend time porting every supported baseline until primary results are stable.

### B4. LongMemEval

- **Repository:** `https://github.com/xiaowu0162/LongMemEval`
- **Role:** later non-regression check and accepted-benchmark bridge.
- **Inspect only after primary pipelines:** official evaluation script, session format, categories concerning update/temporal reasoning.
- **Integration decision:** adapter only; no redesign around its lower mutation pressure.

### B5. Mem0 and A-MEM

- **Repositories:**
  - `https://github.com/mem0ai/mem0`
  - `https://github.com/WujiangXu/A-mem-sys`
- **Role:** common engineering-style baseline candidates.
- **Inspect:** whether they can accept a conversation stream and emit retrieved memories without large modifications; record API calls and storage dependencies.
- **Decision gate:** only integrate one of these early unless Memora already supplies comparable released predictions. Time spent wiring many generic frameworks must not displace ReTrace experiments.

### B6. A-MAC

- **Repository:** `https://github.com/GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents`
- **Role:** code reference for interpretable admission signals, not a central competitor for revision behavior.
- **Inspect later:** how feature scores/configs are recorded and how memory admission outputs are audited.
- **Borrow only if useful:** logging/reporting conventions for interpretable decisions.

## 2.3 Tier C: novelty threats to read, not dependencies to integrate first

### C1. Auto-Dreamer

- **Paper:** `https://arxiv.org/abs/2605.20616`
- **Code status at planning time:** no paper-specific official implementation located; the paper references OpenTinker and verl infrastructure.
- **Role:** prevents ReTrace from claiming learned offline consolidation or future-reward consolidation as its Paper 1 contribution.
- **Coding instruction:** do not try to reproduce before ReTrace's primary experiments exist.

### C2. Useful Memories Become Faulty When Continuously Updated by LLMs

- **Paper:** `https://arxiv.org/abs/2605.12978`
- **Code status at planning time:** no official paper-specific repository located.
- **Role:** motivation and failure evidence for preserving source episodes; not a runnable method baseline.

### C3. AgeMem

- **Repository:** `https://github.com/y1y5/AgeMem`
- **Role:** learned memory-operation competitor relevant to later expansion.
- **Coding instruction:** read its operation/action interfaces and experiment setup, but do not allow costly RL reproduction to block Paper 1.

### C4. MEM1 and other latent/compact-state systems

- **Repository:** `https://github.com/MIT-MI/MEM1`
- **Role:** reserve for a future learned-state continuation; do not use it to reshape Paper 1 into a latent-memory project.

---

# 3. Clean-Room Integration Protocol: how ReTrace should interact with other repositories

## 3.1 External repositories are not submodules of the scientific claim

The ReTrace repository should remain small and readable. Clone upstream repositories outside the tracked codebase or under an ignored directory; invoke them through wrappers. Do not vendor entire external repositories into Git history. The paper must be able to say exactly which upstream commit generated each result.

Create an ignored local layout such as:

```text
external/
  STALE/
  Memora/
  nemori/
  graphiti/
  MemoryAgentBench/
```

Create a tracked registry file:

```yaml
# registry/upstreams.lock.yaml
upstreams:
  stale:
    url: https://github.com/icedreamc/STALE
    commit: TO_FILL_AFTER_CLONE
    license: TO_CHECK
    role: executable_competitor_and_eval
    patched: false
  memora:
    url: https://github.com/geniesinc/Memora
    commit: TO_FILL_AFTER_CLONE
    license: TO_CHECK
    role: official_evaluation
    patched: false
```

Every upstream added later must include the same metadata. Never let an AI model write `license: MIT` unless it inspected that repository's actual LICENSE file.

## 3.2 Separate upstream reproduction from ReTrace evaluation

Results must be divided into three layers:

```text
artifacts/
  upstream_raw/       # untouched outputs produced by the cloned external repos
  normalized/         # converted into ReTrace-compatible records
  retrace_runs/       # outputs produced by your method
  reports/            # score summaries and comparisons
```

Rules:

- Never overwrite raw upstream outputs.
- Each conversion writes a manifest identifying source file, source repo SHA, conversion script SHA, and output checksum.
- Evaluation wrappers must retain original evaluator output in addition to local summary metrics.
- A method failure or parse failure must be recorded, not silently skipped.

## 3.3 Minimal normalized contracts

These are not research claims; they are interoperability records so that several models can write compatible code.

### `EvidenceRecord`

```json
{
  "evidence_id": "string",
  "session_id": "string",
  "timestamp": "string|null",
  "text": "string",
  "source_dataset": "stale|memora|manual_audit|other",
  "source_pointer": "string",
  "is_raw_source": true
}
```

### `MethodTraceRecord`

```json
{
  "example_id": "string",
  "method_name": "string",
  "upstream_commit": "string|null",
  "query": "string",
  "candidate_evidence_ids": ["string"],
  "decision_payload": {},
  "answer": "string",
  "model_config_id": "string",
  "token_counts": {},
  "call_counts": {},
  "errors": []
}
```

### `ScoreRecord`

```json
{
  "example_id": "string",
  "benchmark": "string",
  "method_name": "string",
  "official_scores": {},
  "local_diagnostics": {},
  "evaluator_version": "string",
  "run_manifest_id": "string"
}
```

Do not invent benchmark labels or local diagnostics in upstream output files. Put all ReTrace-only analysis in `local_diagnostics`.

## 3.4 Research registry: prevent literature misunderstanding

Maintain `registry/related_work.yaml` with one record per relevant paper:

```yaml
- key: cupmem_stale_2026
  paper_url: https://arxiv.org/abs/2605.06527
  repo_url: https://github.com/icedreamc/STALE
  code_status: verified_public_repo
  relevance: nearest_method_and_eval
  reproduce_for_paper1: true
  prohibited_claims:
    - implicit invalidation is new to ReTrace
    - stale-premise evaluation is introduced by ReTrace
```

This file is important because AI coding/writing models often expand claims beyond what the literature supports. It should later be used to generate a related-work checklist for the manuscript.

---

# 4. Implementation Work Packets for Gemini / Opus: tasks that can be safely delegated

The following packets should be issued one at a time. A model should not jump ahead to a later packet while earlier acceptance criteria fail.

## Packet 0: repository reconnaissance only

**Goal:** understand upstream executable surfaces without writing ReTrace method code.

**Required outputs:**

```text
docs/upstreams/stale_notes.md
docs/upstreams/memora_notes.md
docs/upstreams/nemori_notes.md
docs/upstreams/graphiti_notes.md
registry/upstreams.lock.yaml
```

Each note must contain:

- cloned commit SHA and license observed locally;
- environment/setup command attempted;
- entrypoint commands;
- input/output file samples;
- dependencies requiring API keys or special services;
- minimal run status: not attempted / blocked / success / failed with error;
- functions or files potentially relevant for an adapter;
- explicit statement of what must not be copied into ReTrace.

**Forbidden in Packet 0:** rewriting upstream code, installing several heavyweight systems before STALE/Memora reconnaissance is complete, implementing the ReTrace method.

## Packet 1: reproducibility skeleton and record validators

**Goal:** establish evidence and run trace contracts before model logic exists.

**Required outputs:**

```text
registry/schema/
  evidence_record.schema.json
  method_trace_record.schema.json
  score_record.schema.json
src/retrace/io/
  records.py
  manifests.py
  validation.py
tests/test_record_roundtrip.py
tests/fixtures/minimal_records/
```

**Acceptance criteria:**

- Pydantic or dataclass records serialize deterministically to JSONL;
- invalid missing provenance fields fail fast;
- no API key or provider dependency is needed for unit tests;
- a run manifest can bind method version, model config, upstream SHA, and output path.

## Packet 2: STALE/CUPMem adapter

**Goal:** reproduce a tiny official STALE/CUPMem run and normalize its artifacts.

**Required outputs:**

```text
src/retrace/adapters/stale.py
src/retrace/adapters/cupmem.py
scripts/run_upstream_stale_smoke.py
scripts/normalize_stale_outputs.py
tests/test_stale_normalization.py
```

**Acceptance criteria:**

- accepts a configurable path to a local clone, never assumes it exists in the ReTrace repository;
- runs or parses at least one official sample;
- preserves raw output unchanged;
- normalized trace points back to input/sample IDs;
- official score output is stored without reimplementation.

## Packet 3: Memora adapter

**Goal:** create a read-only pipeline for one persona and one evaluation sample before scaling.

**Required outputs:**

```text
src/retrace/adapters/memora.py
scripts/inspect_memora_sample.py
scripts/run_memora_official_eval.py
tests/test_memora_evidence_mapping.py
```

**Acceptance criteria:**

- distinguishes information meant to support answers from information that must not appear after mutation;
- never puts evaluation evidence into training-data generation;
- keeps weekly/monthly/quarterly split labels in all records;
- official FAMA calculation remains upstream-controlled whenever possible.

## Packet 4: baseline execution layer

**Goal:** compare trivial retrieval and direct adjudication before adding ReTrace-specific logic.

**Required outputs:**

```text
src/retrace/baselines/retrieval_only.py
src/retrace/baselines/direct_adjudication.py
src/retrace/evaluation/cost_accounting.py
configs/baselines/*.yaml
```

**Acceptance criteria:**

- all baselines use the same configurable model client wrapper;
- token and call counts are emitted for every answer;
- runs can be resumed and cached;
- failures are never counted as blank correct answers.

## Packet 5: ReTrace prototype implementation

**Goal:** implement only the components prescribed by the scientific blueprint, using the completed input/output pipeline.

**Required behavior:**

- no permanent deletion of source evidence;
- all changes in current-use authorization are logged with evidence pointers;
- all relation predictions are inspectable records;
- a missing or uncertain justification defaults to a conservative result rather than fabricated state;
- a later contradicting or expiring condition can reopen earlier evidence for use.

**Acceptance criteria:**

- toy examples cover direct override, condition-based blocking, irrelevant-nearby belief, temporal expiry, and rollback;
- each toy result includes a machine-readable trace and a human-readable explanation;
- prompt-only version must work before any SFT data generation starts.

## Packet 6: training data and verifier training, only after prototype success

**Goal:** train only the narrow relation-verification component after prompt experiments show the mechanism is useful.

**Required safeguards:**

- generated training examples must be stored separately from all official test data;
- no STALE or Memora test answer/evidence leakage into synthetic generation prompts;
- prompts, generator model, seeds, filtering rules, and acceptance/rejection counts must be logged;
- human audit samples must be stored in a versioned CSV/JSONL file;
- compare the trained verifier against a same-model direct-prompt version, not only against weak retrieval.

---

# 5. Engineering Principles: keep the repository publishable from day one

## 5.1 Never let generated code hide experimental decisions

The codebase should expose all choices through config and manifests rather than burying them in prompts or hard-coded functions. In particular, store:

- model identifier and provider;
- temperature/seed/max tokens;
- prompt template version hash;
- retrieval budget;
- number of calls for ingestion, updating, answering and judging;
- upstream repo SHA;
- data split identifier;
- result checksum.

A research run without this record is disposable and should not be cited in a paper table.

## 5.2 No silent prompt evolution

TriMem explicitly explores using a stronger model to rewrite prompts based on evaluation outcomes. That is interesting engineering, but it creates serious comparability risk for this project. If any model alters a prompt after looking at evaluation failures, the change must create:

```text
prompts/<component>/<version>.txt
prompts/CHANGELOG.md
runs/<run_id>/prompt_manifest.json
```

Never overwrite the prompt used for earlier results. Never optimize prompts against the final evaluation split and report the result as clean generalization.

## 5.3 Upstream code modification policy

For each upstream dependency:

- prefer command-line invocation or thin wrapper;
- if an upstream bug blocks running, keep a patch under `patches/<repo>/<sha>/` and explain it;
- do not change an evaluator unless the original evaluator is also run and reported;
- do not merge upstream source trees into `src/retrace/`;
- do not publish external dataset content unless its license permits redistribution.

## 5.4 Keep API expenses measurable

Since multiple model providers may be used during experiments, all generation calls must pass through one wrapper that records tokens, retries, errors, latency, and cache status. A model should not write direct provider calls in random modules.

Suggested rule:

```text
src/retrace/providers/ is the only location allowed to import SDK clients.
```

## 5.5 Failure-led development, not feature-led development

Before implementing a new module, attach it to one observed failure trace. A feature is justified only when it remedies a recorded error type or enables an official benchmark run. This prevents the codebase from becoming a collection of attractive but untestable memory concepts.

---

# 6. What remains deliberately parked for later work

This file exists to help complete Paper 1. The following ideas are not abandoned; they are quarantined so they do not destabilize the first submission.

## 6.1 Candidate future benchmark expansion

A future benchmark project may grow from the small manual audit cases only after the current method establishes that over-revision or unsupported suppression occurs reliably in real systems. Until then, manual cases are diagnostic artifacts, not a headline contribution.

Store any promising case in:

```text
docs/future_benchmark/candidate_cases.jsonl
```

with fields for trigger, protected belief, affected belief, expected rationale, and why existing official benchmarks fail to expose the case.

## 6.2 Candidate latent/consolidation continuation

A later method paper may study episodic-to-semantic transformation, delayed utility and latent belief states. For now, store all such ideas in:

```text
docs/future_latent_memory/idea_log.md
```

Each entry must explicitly state which recent paper is closest—NEMORI, Auto-Dreamer, AgeMem, MEM1, or faulty-consolidation diagnosis—and what non-incremental distinction remains.

## 6.3 Final instruction to any coding model

Do not attempt to make ReTrace look more ambitious by quietly adding latent memory, reward learning or a large benchmark. Ambition in this project means producing an auditable, reproducible result that survives comparison with the nearest released systems. Read the scientific blueprint for the method definition; use this companion to build the codebase without losing that definition.

