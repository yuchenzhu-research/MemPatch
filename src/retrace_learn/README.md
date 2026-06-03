# ReTrace-Learn

Engineering scaffolding that upgrades ReTrace from a prompt-engineered baseline proposer to the trainable **ReTrace-Learn** system described in `AGENTS.md`. It turns *raw multi-subagent dialogue* into *typed memory revisions* that are committed by the existing deterministic kernel—the **ReTrace-Engine** (`retracemem.authorize(...)`), which this package never re-implements or overrides.

---

## 1. ReTrace-Learn v1: three paper-facing stages

ReTrace-Learn v1 has exactly **three** paper-facing stages. Only the first two
are learned; the third is a training protocol, not a trainable module:

```
1. Graph Builder      raw dialogue / memory snapshot -> candidate memory graph   (learned)
2. Proposal Policy    candidate graph + new evidence -> typed revision proposal   (learned)
3. DPA-guided RSFT/DPO  DPA verifies/filters/ranks proposals -> RSFT/DPO signals  (protocol)
```

The deterministic commit path — **ReTrace-Engine** (`authorize(...)`), its
**Parser**, **RevisionGate**, **DPA**, and the **Audit Trace** — is an
*implementation detail* of stages 2–3, **not** a separate paper-level module.
DPA is a deterministic verifier / feedback source: it does **not** learn. The
learned components are the Graph Builder and the Proposal Policy; DPA-guided
RSFT/DPO trains the Proposal Policy.

Hard constraints (enforced in code/tests):
- Canonical action vocabulary only: `SUPERSEDES, BLOCKS, RELEASES, UNCERTAIN, REAFFIRMS, NO_REVISION`.
- Final commit is API-free and deterministic: the only authorization entrypoint is `authorize(...)`.
- The learned policy sees **only method-visible inputs** (candidate view, new evidence,
  candidate beliefs/replacements, conditions, pre-existing REQUIRES anchors). It never
  sees gold revision targets or evaluator final statuses.

The first realistic experiment can fix/oracle the candidate graph and train only
the Proposal Policy.

## 2. Implementation map

The table below lists the *files* that implement the three stages plus the
deterministic commit path. "Stage" maps each file to its paper-facing stage;
rows marked *impl detail* / *future* are not separate paper modules.

| File | Stage | Learned? | In → Out |
|------|-------|----------|----------|
| `runtime/graph_extractor.py` | 1 Graph Builder | yes (`LearnedGraphExtractor`) / oracle (`RuleBasedGraphExtractor`) | raw_dialogue, roles → memory graph dict |
| `runtime/learned_proposer.py` | 2 Proposal Policy | yes (`LearnedTypedRevisionProposer`) / replay (`ScriptedProposer`) | `SharedCandidateView` → `ProposalOutput` (typed actions) |
| `runtime/dpa_runtime.py` | impl detail (ReTrace-Engine) | no (kernel) | raw text / actions → `RuntimeResult` (final statuses + audit) |
| `runtime/reward.py` | 3 DPA-guided signal | no (signal) | actions + parser + DPA + gold → `LearnRewardBreakdown` |
| `runtime/path_ranker.py` | future/optional | yes (`LearnedPathRanker`) / baseline (`HeuristicPathRanker`) | legal candidate paths → selected path (replayable status) |

Bridge: `runtime/views.py` converts graph/candidate dicts into the kernel's
`SharedCandidateView` and converts `RevisionAction`s into the kernel's
`EvidenceProposalBatch`es. This is the only seam between learned dict-IO and the
typed kernel contracts.

## 3. Data schema (`schemas.py`)

Three JSONL row types, all with `to_dict`/`from_dict`/`validate`:

**`graph_extraction_sft.jsonl`** — `GraphExtractionExample`
`example_id, raw_dialogue, subagent_roles, output_graph{evidence_nodes, belief_nodes,
condition_nodes, candidate_replacement_beliefs, dependency_edges}, metadata`.

**`typed_revision_sft.jsonl`** — `TypedRevisionExample`
`example_id, current_graph, new_evidence, candidate_beliefs,
candidate_replacement_beliefs, candidate_conditions_by_belief,
dependency_edges_by_belief, gold_actions, gold_final_statuses, metadata`.

**`dpa_rl_rollouts.jsonl`** — `RLRolloutExample`
`example_id, prompt_input, sampled_actions, parser_result, gate_decisions,
dpa_final_statuses, gold_final_statuses, reward_breakdown, total_reward,
failure_category, audit_trace, metadata`.

The candidate graph keeps `belief_nodes` (existing memory beliefs / old
candidates that can be *targeted*) and `candidate_replacement_beliefs` (new
belief-like candidates selectable as `replacement_belief_id` in `SUPERSEDES`) as
**separate** lists — they share a belief-like schema but must not be merged.
`condition_nodes` are prerequisites/constraints (not replacement beliefs).
`dependency_edges` use `belief_id` + `condition_id` (edge_type `REQUIRES`), never
`source_id`/`target_id`; replacement candidates are keyed by `belief_id`, never
`replacement_id`.

The v1 action object is **closed-world** (no open `scope` field):
`action_type` from the canonical enum, `target_belief_id` from candidate beliefs,
`replacement_belief_id` from candidate replacement beliefs, `target_condition_id`
from candidate conditions, `evidence_ids` from visible evidence.

`RevisionAction` field constraints (`RevisionAction.validate`):
- **every** action carries ≥1 grounding `evidence_id`, **including `NO_REVISION`**
  (which must cite the new evidence; this matches the runtime parser);
- `SUPERSEDES` → `target_belief_id` + `replacement_belief_id` (no condition; replacement ≠ target);
- `BLOCKS`/`RELEASES` → `target_condition_id` only;
- `UNCERTAIN`/`REAFFIRMS` → `target_belief_id` only;
- `NO_REVISION` → no belief/condition/replacement target.

Compatibility is asserted by tests: `CANONICAL_ACTIONS` equals the runtime
`typed_revision_policy.CANONICAL_ACTIONS`, and `FINAL_STATUSES` equals the runtime
`AuthorizationStatus` set.

## 4. Five minimal tasks

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| T1 dialogue → evidence/belief | `raw_dialogue`, roles | `evidence_nodes`, `belief_nodes` | `evidence_node_f1`, `belief_node_f1` |
| T2 belief → condition anchors | belief + dialogue context | `condition_nodes`, `dependency_edges` | `condition_node_f1`, `dependency_edge_f1` |
| T3 graph+evidence → typed actions | `current_graph` + `new_evidence` | `gold_actions` | `action_type_accuracy`, grounding, `exact_action_match` |
| T4 actions → DPA final status | typed actions | `final_belief_statuses` + audit (program runtime, not trained) | replayability / status correctness |
| T5 DPA-in-the-loop reward | model output | `reward_breakdown` + `failure_category` | mean reward, failure distribution |

T1/T2 are evaluated by `eval/eval_graph_extraction.py`; T3 by
`eval/eval_revision_policy.py`; T4 is the deterministic `dpa_runtime`; T5 by
`eval/eval_dpa_reward.py`. The synthetic episodes in
`data/build_synthetic_raw_dialogue.py` provide one worked example per task and
score 1.0 under the oracle extractor/proposer (harness sanity check).

**Generator boundary.** `data/build_synthetic_raw_dialogue.py` is a
*smoke/sanity* generator (all six actions, DPA-computed gold statuses) — it is
**not** the large-scale ReTrace-Learn training generator, and its examples are
not the final method training corpus. Future Learn training data (graph SFT,
proposal SFT, DPA RSFT/DPO preference rows) should live under
`data/retrace_learn/v1_0/`; that large generator is not implemented in v1.
ReTrace-Bench splits under `data/retrace_bench/` are **evaluation-only** and must
never be used as ReTrace-Learn training data.

## 5. SFT training plan

**Stage SFT-1 — graph extractor.** Target = JSON memory graph. Loss = completion
cross-entropy. Builder: `training/train_lora_sft.py::build_graph_sft_pairs`.
Metrics: `valid_json`, `{evidence,belief,condition}_node_f1`, `dependency_edge_f1`.

**Stage SFT-2 — typed revision proposer.** Target = JSON action list. Builder:
`build_revision_sft_pairs`. Metrics: `valid_json`, `action_type_accuracy`,
`target_grounding`, `evidence_grounding`, `exact_action_match`, and the
kernel-grounded `final_status_accuracy_after_DPA`.

Configs: `training/configs/qwen_3b_lora.yaml`, `qwen_4b_lora.yaml`. Heavy deps
(torch/transformers/peft/trl) are imported lazily inside `train()`; `--dry-run`
builds and inspects datasets with stdlib only.

## 6. DPA-in-the-loop reward (`reward.py`)

```
R = + w_final  * final_status_reward      # fraction of gold final statuses DPA reproduces
    + w_json   * valid_json_reward         # parser + schema valid
    + w_tgrd   * target_grounding_reward   # fraction of revision actions with grounded targets
    + w_egrd   * evidence_grounding_reward # fraction with grounded evidence
    + w_nostale* no_stale_propagation_reward
    - w_parse  * parser_error_penalty      # fail-closed parse/schema failure
    - w_inv    * invalid_target_penalty    # 1 - target_grounding
    - w_miss   * missing_evidence_penalty  # 1 - evidence_grounding
    - w_over   * over_update_penalty        # gold AUTHORIZED but DPA defeated it
    - w_under  * under_update_penalty       # gold defeated/unresolved but DPA left AUTHORIZED
    - w_spur   * spurious_uncertain_penalty # UNCERTAIN not present in gold
    - w_stale  * stale_propagation_penalty  # gold SUPERSEDED/BLOCKED but DPA AUTHORIZED
```

Each term is computed from parser/gate/DPA outputs vs gold:
`final_status_*` from `dpa_final_statuses` vs `gold_final_statuses`; grounding from
candidate id sets in the view; over/under/stale from gold-vs-pred status deltas;
`spurious_uncertain` from `gold_actions`. `classify_failure` picks the dominant
failure for curriculum/analysis. This reward is the **DPA-guided training
signal** for stage 3: consumed offline to build DPO/RSFT preference pairs
(`data/export_rl_rollouts.py::build_preference_pairs`). An online GRPO path
(`training/train_grpo.py::score_completion`) is a future/optional extension.

## 7. Learned defeat-path ranker (future/optional)

> Not part of the v1 three-stage method. `path_ranker.py` is a safe, advisory
> future extension kept for experiments; it is never on the deterministic commit
> path and emits no final status. Treat it as optional.

`path_ranker.py` ranks **DPA-legal** candidate paths
(`DIRECT_SUPERSEDE, PREREQUISITE_BLOCK, UNRESOLVED_UNCERTAIN, AUTHORIZED_DEFAULT`)
and selects one. Safety invariants (tested):
- selection is restricted to `legal_paths_from_runtime(...)`, so an illegal path can
  never be chosen — even an adversarial scorer keeps `legal_selection_rate == 1.0`;
- each path maps deterministically to a final status (`PATH_TYPE_TO_STATUS`), so the
  runtime can replay/audit the decision (`replay_status_from_ranking`);
- `HeuristicPathRanker` mirrors canonical precedence and replays DPA exactly
  (`dpa_replay_consistency == 1.0`);
- `rationale` is advisory and never decides the outcome.

Training modes (data shapes provided): supervised path classification, pairwise
ranking, DPO preference data, RL-from-final-status-correctness. The ranker never
touches `RevisionGate` and never emits a final status directly.

## 8. Code layout

```
src/retrace_learn/
  schemas.py                 # data contracts + RevisionAction validation
  runtime/
    views.py                 # dict ↔ SharedCandidateView / EvidenceProposalBatch bridge
    graph_extractor.py       # Stage 1 Graph Builder (rule-based oracle + learned wrapper)
    learned_proposer.py      # Stage 2 Proposal Policy (learned proposer + scripted replay)
    dpa_runtime.py           # impl detail: ReTrace-Engine (parser + gate + authorize() wrapper)
    reward.py                # Stage 3 DPA-guided training signal
    path_ranker.py           # future/optional: safe defeat-path ranker
  data/
    build_synthetic_raw_dialogue.py  # synthetic episodes (all 6 actions)
    jsonl_io.py
    export_graph_sft.py / export_revision_sft.py / export_rl_rollouts.py
  training/
    train_lora_sft.py / train_dpo.py / train_grpo.py
    configs/qwen_3b_lora.yaml / qwen_4b_lora.yaml
  eval/
    metrics.py
    eval_graph_extraction.py / eval_revision_policy.py / eval_dpa_reward.py / eval_ablation.py
tests/retrace_learn/
    test_schema_validation.py / test_reward.py / test_path_ranker.py / test_end_to_end_smoke.py
```

Implemented now: schemas, the runtime modules, the bridge, synthetic smoke data,
three exporters, four eval scripts, dataset-building halves of all trainers, tests.
Stubbed (lazy, GPU-gated): the actual `train()` loops in the trainers. The path
ranker and online GRPO trainer are future/optional, not part of the v1 method.

## 9. Minimal smoke test (`tests/retrace_learn/test_end_to_end_smoke.py`)

Dialogue: A asserts an old belief; B supersedes it with new evidence; C blocks a
condition another belief requires; D releases that condition. Verified end-to-end:
- extractor recovers `ev1..ev4`, beliefs `b_old/b_dep`, replacement `b_new`, condition `c1`, dep `(b_dep, c1)`;
- proposer emits `SUPERSEDES`, `BLOCKS`, `RELEASES`; parser + gate admit all three;
- DPA: `b_old → SUPERSEDED`, `b_new → AUTHORIZED`, `b_dep → AUTHORIZED`;
- audit trace carries gate decisions and a `DIRECT_SUPERSEDE` defeat path
  (`replacement_belief_id = b_new`, non-empty `evidence_edge_ids`).

Run: `python -m pytest tests/retrace_learn -q`.

## 10. Two-week plan

- **Days 1–2:** freeze schemas + runtime bridge against the kernel; expand synthetic
  generator coverage and adversarial perturbations. *(scaffolding landed)*
- **Days 3–5:** SFT-1 graph extractor LoRA; wire `train()`; report node F1s on a held-out split.
- **Days 6–8:** SFT-2 revision proposer LoRA; track `final_status_accuracy_after_DPA`,
  not just action match.
- **Days 9–11:** rollout generation at scale; build DPO pairs; run DPO; compare reward distribution.
- **Days 12–13:** GRPO with the DPA-in-the-loop reward; ablate path ranker (heuristic vs learned).
- **Day 14:** consolidate eval (`eval/*`), write report.

## 11. Risk register

| Risk | Mitigation / downgrade |
|------|------------------------|
| Replacement beliefs aren't scored by the kernel (disjoint set) | Runtime surfaces them as `AUTHORIZED` only via *admitted* SUPERSEDES gate edges (replayable). Full DPA over replacements is a future extension; downgrade = report replacement status from gate audit only. |
| Small model emits invalid JSON | Fail-closed parser → no revisions admitted; `parser_error_penalty`. Downgrade = constrained decoding / JSON grammar. |
| Reward hacking via spurious actions | `over_update`/`spurious_uncertain` penalties; grounding required. Downgrade = SFT-only (skip RL). |
| Learned path ranker picks wrong path | Constrained to legal candidates + deterministic replay; ranker is advisory. Downgrade = `HeuristicPathRanker` (provably DPA-consistent). |
| Synthetic-only data | Builders are the seed; real reviewed examples plug into the same schemas. Downgrade = keep synthetic for smoke/regression only. |
| Heavy ML deps unavailable | Trainers import torch/peft/trl lazily; `--dry-run` + eval run on stdlib. |

### Quick start

```bash
python -m pytest tests/retrace_learn -q                 # smoke + unit
python -m retrace_learn.eval.eval_revision_policy        # oracle policy metrics
python -m retrace_learn.eval.eval_ablation               # path-ranker safety
python -m retrace_learn.data.export_rl_rollouts --out outputs/retrace_learn/dpa_rl_rollouts.jsonl
python -m retrace_learn.training.train_lora_sft --task revision --dry-run
```
