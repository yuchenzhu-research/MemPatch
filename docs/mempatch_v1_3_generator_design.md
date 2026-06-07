# MemPatch v1.3 Generator Design

Status: **scaffold** — blueprint registry and resolver in MemPatch; upstream renderer required for JSONL export.

## Problem (v1.1 / v1.2)

- v1.1 `main` lacks `ask_clarification` / `escalate`.
- v1.2 remaps v1.1 pools but `ask_clarification`, `escalate`, and ci-derived `mark_unresolved` share the same ready+CI-failed public skeleton → **public decision boundary is not learnable**.
- LoRA v3/v4 improved memory/evidence/diagnosis but **ask=0, mark=0** persist → P0 is generator redesign, not rank/iters.

## v1.3 principles

1. **Decision from resolver, not post-hoc labels** — `hidden_gold.expected_decision` = `decision_resolver(public_input, blueprint_params)`.
2. **Visible, mutually exclusive triggers** — each non-answer decision embeds distinct marker phrases in `public_input.event_trace`.
3. **No shared core skeleton** across ask / escalate / mark for the same pattern family instance.
4. **Split-disjoint seeds** — `split_seed_namespace = mempatch_v13`.

## Architecture

```text
blueprints.py          pattern families + decision_variant + decision_triggers registry
split_sampler.py       stratified quotas → V13BlueprintInstance list
scenario_builder.py    blueprint + UnifiedRendererV13 → scenario dict
decision_resolver.py   rendered public_input → expected_decision (gold)
export_jsonl.py        train/main/hard/scenarios.jsonl + manifest
```

Renderer name: `unified_renderer_v13` (upstream, not in MemPatch).

## Required scenario metadata

Every row must include in `metadata`:

| Field | Example |
|-------|---------|
| `renderer` | `unified_renderer_v13` |
| `pattern` | `authority_conflict` |
| `pattern_trap_type` | `verified_sources_disagree` |
| `decision_variant` | `dual_verified_no_policy_gate` |
| `decision_triggers` | `["mark_verified_conflict"]` |
| `core_event_signature` | readable preview (audit also hashes) |
| `split_seed_namespace` | `mempatch_v13` |

## Decision semantics (five-way)

### use_current_memory

Verified maintainer / CI / release / auditor confirms current state; stable scope matches target; beta/out-of-scope distractors present but rejected.

Triggers: `verified_maintainer_confirms`, `verified_ci_release_confirms`, `verified_auditor_confirms`, `stable_scope_matches_target`, `distractor_beta_out_of_scope`.

### refuse_due_to_policy

Explicit prohibition on storage/modification.

Triggers: `credential_secret_forbidden`, `security_policy_forbids_storage`, `compliance_do_not_store`.

### ask_clarification

Missing user intent / scope / target — **not** policy/human-review gates.

Triggers: `missing_target_scope`, `ambiguous_user_intent`, `ambiguous_workspace`, `multiple_candidate_memories`, `explicit_hold_for_user`.

**Forbidden:** policy gate phrases, ready+CI-failed-only skeletons.

### escalate

Policy/authority/human-review blocks automatic update despite sufficient evidence.

Triggers: `policy_human_review_required`, `verified_reviewer_hold`, `security_compliance_block`, `protected_production_memory`, `evidence_sufficient_but_policy_blocks`.

**Forbidden:** ambiguous user intent as sole trigger.

### mark_unresolved

Cannot safely determine state; not user clarification; not escalation process.

Non-CI: `assume_duplicate_means_fixed`, `treat_open_pr_as_merged`, `trust_unverified_over_verified`, `mark_verified_conflict`, `mark_insufficient_passive`, `mark_stalemate_no_authority`.

CI-derived (independent skeletons): `ci_second_verified_contradiction`, `ci_passive_monitor_gap`, `ci_no_authority_path`.

## Pilot vs full

| Split | Pilot | Full |
|-------|------:|-----:|
| train | 500 | 2700 |
| main | 100 | 800 |
| hard | 100 | 500 |

Pilot gates:

- All 5 decisions covered per split
- ≥2 pattern families per decision (registry-level)
- ask / escalate / mark: ≥3 `decision_variant` each
- Audit release gate passes (see below)

## Upstream files required (not in MemPatch)

| Upstream module | Role |
|-----------------|------|
| `render/unified_renderer_v13.py` | Emit `public_input` with trigger phrases |
| `blueprints/registry.yaml` | Pattern templates parameterized by `decision_variant` |
| Per-pattern blueprint modules | Event/memory/task templates |
| Gold field generators | memory_state, evidence ids, failure diagnosis (decision via resolver only) |

MemPatch provides the **contract** (`UnifiedRendererV13` protocol in `scenario_builder.py`) and **validation** (`audit_decision_boundary.py`).

## Commands

Registry + sampling dry-run (no render):

```bash
PYTHONPATH=.:src .venv/bin/python scripts/generate_mempatch_v13_pilot.py --dry-run
```

Pilot export (blocked until upstream renderer wired):

```bash
PYTHONPATH=.:src .venv/bin/python scripts/generate_mempatch_v13_pilot.py \
  --out-dir local/mempatch_v13_pilot \
  --manifest-out local/mempatch_v13_pilot/manifest.json
```

Distribution report:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/report_split_decision_distribution.py \
  --split train local/mempatch_v13_pilot/train/scenarios.jsonl \
  --split main local/mempatch_v13_pilot/main/scenarios.jsonl \
  --split hard local/mempatch_v13_pilot/hard/scenarios.jsonl
```

Release audit:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/audit_decision_boundary.py \
  --data local/mempatch_v13_pilot/train \
  --data local/mempatch_v13_pilot/main \
  --data local/mempatch_v13_pilot/hard \
  --out-json local/results/decision_boundary_audit_v13.json \
  --out-md local/results/decision_boundary_audit_v13.md
```

## Training gate

Do **not** train until pilot audit shows:

- Five-decision distribution matches quotas
- ask/escalate/mark trigger coverage = 100%
- ask↔escalate shared `core_event_signature` = 0
- ask/esc/mark full public-view collision = 0
- `pattern × decision × split` matrix reasonable
- `validate_mempatch_bench_dataset.py --packaging-final` passes

## Relation to v1.2

- Do **not** extend `scripts/generate_mempatch_v12.py` remap logic.
- v1.2 export in `local/MemPatch/` remains for regression comparison until v1.3 pilot passes audit.
