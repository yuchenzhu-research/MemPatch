# MemPatch v1.2 Dataset Redesign Plan

Status: **plan only** — do not train on v1.1 main until v1.2 is released.  
Scope: fix **decision label space mismatch** between train/eval splits without forging labels or copying hard gold into train.

---

## 1. Problem statement

### v1.1 observed distributions

| Split | Renderer | use_current_memory | mark_unresolved | ask_clarification | escalate | refuse_due_to_policy |
|-------|----------|-------------------:|----------------:|------------------:|---------:|---------------------:|
| **main** (3000) | `main_final_renderer` | 2200 | 600 | **0** | **0** | 200 |
| **hard** (500) | `hard_final_renderer` | 250 | 100 | 60 | 40 | 50 |

v2 smoke train (first 1024 main rows) inherited the same 3-label skew.  
Eval on hard50 requires `ask_clarification` / `escalate`, which **cannot be learned** from main-only SFT.

### Root cause (not “too little training”)

1. **Decision labels are assigned at render time**, not by `scripts/prepare_mempatch_sft.py`.
2. **`main_final_renderer` and `hard_final_renderer` are different labeling pipelines**, not just difficulty filters.
3. The **same blueprint pattern** can map to different gold decisions per renderer.

Example — `ci_failed_after_claim`:

| Split | n | expected_decision distribution |
|-------|--:|------------------------------|
| main | 200 | mark_unresolved **200** |
| hard | 110 | ask_clarification 60, escalate 40, mark_unresolved 10 |

So hard is not “main + harder”; it includes **label-space expansion** that main never emits.

### What this repo does / does not contain

| In MemPatch repo | Outside (upstream) |
|------------------|-------------------|
| `benchmark/mempatch_bench/general_taxonomy.py` (`DECISIONS`) | Blueprint registry + scenario templates |
| `scripts/validate_mempatch_bench_dataset.py` | `main_final_renderer` / `hard_final_renderer` |
| `scripts/prepare_mempatch_sft.py` (reads gold, quota guardrail) | Gold resolver: pattern + params → `expected_decision` |
| `hf_release/mempatch_v1_1/{manifest,checksums}.json` | Bulk JSONL generation + split assignment |
| Evaluator / runner | Per-blueprint decision variant tables |

**Do not** fix v1.1 by copying hard rows into train or relabeling in MemPatch.

---

## 2. Design goals (v1.2)

1. **Five-decision coverage** in **train** and **main** (all labels in `DECISIONS`).
2. **train ≠ main ≠ hard** — disjoint `scenario_id` ranges and non-overlapping blueprint instances.
3. **hard** remains adversarial held-out eval; **not** used as SFT source.
4. **No forged labels** — gold only from upstream renderer.
5. **Split-level label quotas** enforced at generation time + validated before release.
6. **Manifest + checksums** per split JSONL (extend v1.1 packaging).
7. Keep `prepare_mempatch_sft.py --target-style decision_balanced` **quota guardrail** (fail if train pool lacks labels).

---

## 3. Proposed v1.2 splits

Target row counts (4000 public total):

### train (2700) — SFT / local training only

| expected_decision | count |
|-------------------|------:|
| use_current_memory | 600 |
| mark_unresolved | 600 |
| ask_clarification | 600 |
| escalate | 600 |
| refuse_due_to_policy | 300 |

### main (800) — public broad-coverage eval / dev

| expected_decision | count |
|-------------------|------:|
| use_current_memory | 400 |
| mark_unresolved | 150 |
| ask_clarification | 100 |
| escalate | 75 |
| refuse_due_to_policy | 75 |

### hard (500) — held-out adversarial probe (unchanged size, relabeled mix)

| expected_decision | count |
|-------------------|------:|
| use_current_memory | 150 |
| mark_unresolved | 100 |
| ask_clarification | 100 |
| escalate | 75 |
| refuse_due_to_policy | 75 |

**Policy**

- `train` ships for **local / method development** (gitignored JSONL or separate HF config); not mixed into v1.1 `main` eval split.
- `main` + `hard` replace v1.1 public release under `hf_release/mempatch_v1_2/`.
- Scenario IDs (suggested disjoint ranges):

| Split | scenario_id range | metadata.split |
|-------|-------------------|----------------|
| train | `case-100001` … `case-102700` | `train` |
| main | `case-000001` … `case-000800` | `main` |
| hard | `case-003001` … `case-003500` | `hard` |

Use **non-overlapping blueprint instance seeds** per split (see §5).

---

## 4. Upstream work required (external engineering)

Bring these from the **blueprint / renderer** repository (historically referenced as ReTrace-style blueprints in `source_pointers`):

### 4.1 Files / modules to add or refactor

| Item | Purpose |
|------|---------|
| `blueprints/registry.yaml` (or equivalent) | Pattern catalog with **decision variant** support per pattern |
| `blueprints/ci_failed_after_claim.py` (example) | Template already used in v1.1; extend with `decision_mode` ∈ {ask, escalate, unresolved, …} |
| `render/decision_resolver.py` | Single function: `(pattern, blueprint_params, difficulty, split_role) → expected_decision` |
| `render/unified_renderer_v12.py` | Replace split-specific label bias; **one** gold logic, split only controls quotas / difficulty / seeds |
| `render/split_sampler.py` | Stratified sampling to hit §3 quotas per split |
| `render/scenario_builder.py` | Builds `public_input`, tasks, `hidden_gold`, `metadata` |
| `packaging/export_jsonl.py` | Writes `train/main/hard/scenarios.jsonl` |
| `packaging/emit_manifest.py` | `manifest.json`, `checksums.json`, `dataset_info.json`, decision histograms |

### 4.2 Deprecate or merge

- **`main_final_renderer`** — remove hard-coded 3-decision mapping; or make thin wrapper over `unified_renderer_v12` with split=`main` quotas.
- **`hard_final_renderer`** — same; split=`hard` quotas.  
  Migrate **ask_clarification / escalate** logic from hard-only branches into shared `decision_resolver` usable at L1–L4 where appropriate.

### 4.3 Blueprint migration checklist (`ci_failed_after_claim`)

Hard v1.1 already proves the pattern supports ask/escalate. For v1.2:

1. Extract hard renderer branches that set `ask_clarification` vs `escalate` vs `mark_unresolved`.
2. Parameterize as blueprint fields (e.g. `authority_gap`, `ci_ambiguity`, `policy_gate`) — **no hand-edited JSONL**.
3. Register L1–L2 variants for **train/main** (not only L3–L4).
4. Add **new blueprint instances** (new seeds / `blueprint-*` ids) for train; do not reuse v1.1 `case-00300x` rows.
5. Validate with MemPatch `validate_mempatch_bench_dataset.py --packaging-final`.

### 4.4 Other patterns

Audit v1.1 main for patterns that could support non-answer decisions but currently do not:

| Pattern (main) | Current main decision | Candidate for v1.2 |
|----------------|----------------------|-------------------|
| `authority_conflict` | mark_unresolved (100) | escalate variants |
| `security_policy_override` | refuse (subset) | already refuse |
| `negative_evidence_required` | mark_unresolved (100) | ask_clarification variants |

Full pattern→decision matrix should be produced by upstream `split_sampler` + documented in manifest `notes`.

---

## 5. Seeds, disjointness, and leakage

### Current v1.1

- main: `case-000001`–`case-003000`, seeds from 2027+
- hard: `case-003001`–`case-003500`, seeds also from 2027+  
- IDs are disjoint; **seed sequences overlap** — acceptable if blueprint instances differ, but v1.2 should use **split-prefixed seed namespaces**:

```
seed_train = hash("mempatch_v12_train", blueprint_id, index)
seed_main  = hash("mempatch_v12_main", blueprint_id, index)
seed_hard  = hash("mempatch_v12_hard", blueprint_id, index)
```

### Leakage checks (automate in packaging)

- No `scenario_id` overlap across splits.
- No identical `(pattern, blueprint_id, seed)` triple across splits.
- `public_input` text hash optional secondary check.
- `hidden_gold` never appears in `public_input` (existing validator).

---

## 6. MemPatch repo integration (after upstream JSONL exists)

### 6.1 New release bundle

```
hf_release/mempatch_v1_2/
  manifest.json          # version 1.2.0, split counts, decision quotas, generation_seed policy
  checksums.json         # sha256 per scenarios.jsonl
  dataset_info.json
  README.md
  train/scenarios.jsonl  # optional in public HF; at minimum documented for local SFT
  main/scenarios.jsonl
  hard/scenarios.jsonl
```

### 6.2 Scripts to add (small, in MemPatch)

| Script | Role |
|--------|------|
| `scripts/report_split_decision_distribution.py` | Print per-split `expected_decision` histogram (CI gate) |
| `scripts/package_mempatch_release.py` | Compute checksums, write manifest from upstream export dir |

`validate_mempatch_bench_dataset.py` — extend `--packaging-final` with:

- Required decision labels present per split (train/main/hard rules differ).
- Observed counts within tolerance of manifest quotas (±0).

### 6.3 Training path (unchanged guardrail)

```bash
# After v1.2 train JSONL exists locally:
python scripts/prepare_mempatch_sft.py \
  --main local/MemPatch/train/scenarios.jsonl \
  --hard local/MemPatch/hard/scenarios.jsonl \
  --out-dir local/train_data/mempatch_qwen14b_v3_decision_balanced \
  --target-style decision_balanced \
  --decision-quotas "..." \
  --valid-size 128
```

- SFT reads **train split only** (`--main` path points to train JSONL).
- `decision_balanced` **fails closed** if quotas exceed pool (v1.1 main correctly fails today).
- Eval remains `hard/scenarios.jsonl` + `run_mempatch_mlx.py` + `evaluate_mempatch_predictions.py`.

---

## 7. Acceptance criteria (v1.2 release)

- [ ] train JSONL: all 5 decisions, quotas ≥ targets in §3  
- [ ] main JSONL: all 5 decisions, quotas ≥ targets in §3  
- [ ] hard JSONL: all 5 decisions, quotas ≥ targets in §3  
- [ ] No scenario_id / blueprint-instance leakage across splits  
- [ ] `validate_mempatch_bench_dataset.py --packaging-final` passes on each split  
- [ ] `report_split_decision_distribution.py` matches manifest  
- [ ] `checksums.json` matches files  
- [ ] `prepare_mempatch_sft.py decision_balanced` succeeds against **train** with v1.2 quotas  
- [ ] v1.1 → v1.2 breaking change documented (main no longer 3000 rows; label space complete)

---

## 8. Explicit non-goals

- Do **not** train on v1.1 hard or copy hard rows into train.
- Do **not** relabel v1.1 JSONL inside MemPatch.
- Do **not** map `blocked` → `escalate` in runner/canonicalizer to fake training labels.
- Do **not** run full 3000-main LoRA until v1.2 train exists.

---

## 9. Immediate next steps

1. **Upstream**: implement `unified_renderer_v12` + train split export (2700 rows).  
2. **MemPatch**: add `report_split_decision_distribution.py` + manifest packaging helper.  
3. **Validate**: run distribution + validator on upstream output dropped into `local/MemPatch/{train,main,hard}/`.  
4. **Then** regenerate SFT with `decision_balanced` / `evidence_compact` for v3 training — **not before**.

---

## Appendix A — v1.1 main inventory (reference)

```
use_current_memory:    2200
mark_unresolved:        600
refuse_due_to_policy:   200
ask_clarification:        0
escalate:                 0
renderer: main_final_renderer (100%)
```

## Appendix B — v1.1 hard ask/escalate patterns

```
ask_clarification: 60 × ci_failed_after_claim (L3/L4)
escalate:          40 × ci_failed_after_claim (L3/L4)
```
