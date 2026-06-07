---
license: cc-by-4.0
language:
- en
pretty_name: MemPatch
tags:
- agent-memory
- llm-agents
- rapid-memory-integration
- memory-revision
- evaluation
configs:
- config_name: default
  data_files:
  - split: train
    path: train/scenarios.jsonl
  - split: validation
    path: validation/scenarios.jsonl
  - split: test
    path: test/scenarios.jsonl
---

# MemPatch v1.3.0

Decision-boundary-aware benchmark for Rapid Memory Integration (RMI).

MemPatch evaluates whether an agent can update the usable state of memory when
new evidence arrives. The public input is a scenario event trace and initial
memory set; the target output is strict JSON with:

- `decision`
- `memory_state`
- `evidence_event_ids`
- `failure_diagnosis`
- `answer`

Gold labels are in `hidden_gold`. Public evaluation should use
`benchmark.api.evaluate_predictions`.

## Version Notes

Earlier internal dataset drafts had decision-boundary issues:

- v1.1 did not cover `ask_clarification` and `escalate` in the main split.
- v1.2 filled the five labels, but several `ask_clarification`, `escalate`,
  and `mark_unresolved` rows shared the same visible CI-failure skeleton.
  This produced ask/escalate public-signature collisions.

v1.3 fixes the label coverage and public decision boundary:

- all five decision labels appear in every split;
- `ask_clarification`, `escalate`, and `mark_unresolved` have mutually visible
  natural-language triggers;
- literal public trigger markers such as `[trigger:...]` are not present;
- each `decision_variant` has 5 normalized public-view surface templates;
- normalized surface templates are split-disjoint across train/validation/test;
- the decision-boundary audit reports 0 ask/escalate shared signatures.

## Splits

| Split | Rows | Purpose |
|-------|-----:|---------|
| `train` | 2700 | Fine-tuning / SFT only |
| `validation` | 800 | Development eval |
| `test` | 500 | Held-out final eval (L4-heavy) |

**Total: 4000.** Renderer: `unified_renderer_v13`. All five `expected_decision` labels in every split.

Use the current split names only: `train`, `validation`, `test`.
Do not use the old `main` / `hard` naming in new experiments.

## v1.3 Taxonomy Coverage

The v1.3 release uses primary labels from the broader taxonomy:

- 7 primary `failure_diagnosis` labels:
  `conflict_collapse`, `under_update`, `wrong_source_attribution`,
  `scope_leakage`, `policy_violation`, `stale_memory_reuse`,
  `memory_hallucination`
- 8 primary pattern families
- 6 primary domains
- difficulty is represented with short labels: `train` and `validation` are
  `L3`; `test` is `L4`

Additional taxonomy labels in the code are reserved for future releases and are
not present as v1.3 gold labels.

## Decision Distribution

| split | use_current_memory | mark_unresolved | ask_clarification | escalate | refuse_due_to_policy | total |
|-------|-------------------:|----------------:|------------------:|---------:|---------------------:|------:|
| train | 600 | 600 | 600 | 600 | 300 | 2700 |
| validation | 400 | 150 | 100 | 75 | 75 | 800 |
| test | 150 | 100 | 100 | 75 | 75 | 500 |

## Pattern Distribution

Aggregate pattern x decision counts across all splits:

| pattern | use_current_memory | mark_unresolved | ask_clarification | escalate | refuse_due_to_policy | total |
|---------|-------------------:|----------------:|------------------:|---------:|---------------------:|------:|
| authority_conflict | 0 | 142 | 200 | 250 | 0 | 592 |
| ci_failed_after_claim | 0 | 282 | 0 | 250 | 0 | 532 |
| closed_as_duplicate_not_fixed | 0 | 142 | 200 | 0 | 0 | 342 |
| label_state_mismatch | 383 | 0 | 0 | 0 | 224 | 607 |
| maintainer_correction_over_user_claim | 384 | 0 | 200 | 0 | 0 | 584 |
| negative_evidence_required | 0 | 284 | 0 | 0 | 0 | 284 |
| security_policy_override | 0 | 0 | 0 | 250 | 226 | 476 |
| version_scope_leakage | 383 | 0 | 200 | 0 | 0 | 583 |

## Variant Distribution

| decision_variant | decision | rows |
|------------------|----------|-----:|
| verified_maintainer_overrides_user | use_current_memory | 384 |
| verified_release_confirms_stable | use_current_memory | 383 |
| verified_auditor_signoff | use_current_memory | 383 |
| missing_target_on_update_request | ask_clarification | 200 |
| ambiguous_scope_no_verified_ruling | ask_clarification | 200 |
| multiple_matching_memories | ask_clarification | 200 |
| user_intent_ambiguous_action | ask_clarification | 200 |
| human_review_gate_active | escalate | 250 |
| compliance_block_with_sufficient_evidence | escalate | 250 |
| protected_prod_memory | escalate | 250 |
| credential_forbidden_write | refuse_due_to_policy | 226 |
| compliance_do_not_store | refuse_due_to_policy | 224 |
| dual_verified_no_policy_gate | mark_unresolved | 142 |
| duplicate_ticket_assumed_fixed | mark_unresolved | 142 |
| passive_monitor_gap | mark_unresolved | 142 |
| trust_chain_broken | mark_unresolved | 142 |
| ci_contradiction_independent | mark_unresolved | 141 |
| ci_monitor_gap | mark_unresolved | 141 |

## Boundary Audit

Run the release-blocking audit:

```bash
PYTHONPATH=.:src python scripts/audit_decision_boundary.py \
  --data hf_release/mempatch/train/scenarios.jsonl \
  --data hf_release/mempatch/validation/scenarios.jsonl \
  --data hf_release/mempatch/test/scenarios.jsonl \
  --out-json local/results/post_marker_fix_audit.json \
  --out-md local/results/post_marker_fix_audit.md
```

Expected v1.3 audit properties:

- public marker leakage: 0 rows;
- normalized public-view templates: 90;
- train/validation/test normalized template overlap: 0;
- each `decision_variant`: 5 normalized public-view templates;
- ask/escalate/mark cross-decision collisions: 0.

## Checksums

Verify `checksums.json`:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path("hf_release/mempatch")
checksums = json.loads((root / "checksums.json").read_text())
for rel, expected in checksums.items():
    actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    print(rel, "OK" if actual == expected else f"FAIL {actual}")
PY
```

## Loading

```python
from benchmark.api import evaluate_predictions, load_predictions, load_scenarios

scenarios = load_scenarios("hf_release/mempatch/test/scenarios.jsonl")
predictions = load_predictions("predictions.jsonl")
result = evaluate_predictions(scenarios, predictions, strict=True)
print(result["headline_metrics"])
```

## Reproducibility

Regenerate the release:

```bash
PYTHONPATH=.:src python scripts/generate_mempatch.py \
  --full --out-dir hf_release/mempatch \
  --manifest-out hf_release/mempatch/manifest.json

PYTHONPATH=.:src python scripts/package_mempatch_release.py \
  --input-dir hf_release/mempatch \
  --out-dir hf_release/mempatch \
  --validate --report
```

## Limitations

MemPatch v1.3 is controlled synthetic data. The release now has 90 normalized
public-view templates with split-disjoint surfaces, but templates are still
generated from a finite registry of pattern families and variants. Report
per-decision metrics, and treat small SFT smoke results as diagnostic rather
than as final benchmark evidence.

## Files

- `train/scenarios.jsonl`, `validation/scenarios.jsonl`, `test/scenarios.jsonl`
- `manifest.json`, `checksums.json`, `dataset_info.json`
