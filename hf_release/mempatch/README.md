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

- metadata schema version: `mempatch_bench_general_1`;
- all five decision labels appear in every split;
- `ask_clarification`, `escalate`, and `mark_unresolved` have mutually visible
  natural-language triggers;
- literal public trigger markers such as `[trigger:...]` are not present;
- each `decision_variant` has 5 normalized public-view surface templates;
- normalized surface templates are split-disjoint across train/validation/test;
- the decision-boundary audit reports 0 ask/escalate shared signatures.

## Splits

| Split | Purpose |
|-------|---------|
| `train` | SFT + stratified k-fold held-out |
| `test` | Held-out final eval (L4) |

Renderer: `unified_renderer_v13`. Scenario IDs are contiguous: `case-1` … `case-N`
(train first, then test). Use split names `train` and `test` only — not the old
`main` / `hard` naming.

## v1.3 Taxonomy Coverage

The v1.3 release uses primary labels from the broader taxonomy:

- 7 primary `failure_diagnosis` labels:
  `conflict_collapse`, `under_update`, `wrong_source_attribution`,
  `scope_leakage`, `policy_violation`, `stale_memory_reuse`,
  `memory_hallucination`
- 8 primary pattern families
- 6 primary domains
- difficulty: `train` is `L3`; `test` is `L4`

Additional taxonomy labels in the code are reserved for future releases and are
not present as v1.3 gold labels.

## Boundary Audit

Run the release-blocking audit:

```bash
PYTHONPATH=.:src python scripts/workflows/audit_decision_boundary.py \
  --data hf_release/mempatch/train \
  --data hf_release/mempatch/test \
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
PYTHONPATH=.:src python scripts/data/generate_mempatch.py \
  --full --out-dir hf_release/mempatch \
  --manifest-out hf_release/mempatch/manifest.json

PYTHONPATH=.:src python scripts/data/package_mempatch_release.py \
  --input-dir hf_release/mempatch \
  --out-dir hf_release/mempatch \
  --validate
```

## Limitations

MemPatch v1.3 is controlled synthetic data. The release now has 90 normalized
public-view templates with split-disjoint surfaces, but templates are still
generated from a finite registry of pattern families and variants. Report
per-decision metrics, and treat small SFT smoke results as diagnostic rather
than as final benchmark evidence.

## Files

- `train/scenarios.jsonl`, `test/scenarios.jsonl`
- `manifest.json`, `checksums.json`, `dataset_info.json`
