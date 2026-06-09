# Scripts map

One evaluator (`benchmark.api.evaluate_predictions`) scores every run. Outputs go under `local/` (gitignored).

Path helpers live in `scripts/_root.py` (`REPO_ROOT`, `SCRIPTS_DIR`, `bootstrap_from`).

## Layout

```
scripts/
  _root.py              REPO_ROOT / SCRIPTS_DIR helpers
  workflows/            audit, validate, evaluate, paper pipeline
  eval/                 model runners (API, MLX, memory baselines)
  data/                 dataset generation, SFT prep, release packaging
  mlx/                  MLX download + chat utilities
  memory/               external memory baseline context builders
  analysis/             post-hoc error analysis
```

## Experiment lines (what to compare)

| Line | Script | What it does | Trained? |
|------|--------|--------------|----------|
| **External memory baselines** | `eval/run_mempatch_memory_baselines.py` | `build_prompt` → MLX → five-field JSON. Backends: `base`, `full`, `rag`, `mem0`. | No |
| **DirectJudge** | `eval/run_mempatch_model.py` | API/MLX provider → five-field JSON (same schema as baselines). | Optional |
| **Path A** (`\baseproj{}`) | `eval/run_mlx_revision_module_eval.py` | Revision view → **typed actions JSON** → DPA → projection → five fields. | No (base MLX) |
| **Path B** (`\adapted{}`) | `eval/run_mlx_lora_smoke_eval.py` | SFT prompt → MLX+LoRA → **direct five-field JSON**. | Yes (LoRA) |

**Fair comparisons**

- vs external memory: **RAG / Full / mem0** vs **Path B** (same JSON protocol; align backbone + split).
- vs untrained module: **Path A** vs **Path B** (same stack; only training differs).
- Path A joint≈0 means *untrained proposer fails*; Path B joint≈0.54 on test500 means *LoRA is what works*.

### Memory baselines and `build_prompt`

RAG, Full, Base, and Mem0 backends all funnel through `benchmark.model_runner.build_prompt` on a sanitized `public_scenario_view` (see `memory/mempatch_memory_context.py`). The MLX answer step in `eval/run_mempatch_memory_baselines.py` uses the same five-field JSON schema as DirectJudge and Path B.

## Metrics (from `benchmark/scorers_general.py`)

### Headline (paper tables)

| Metric | Meaning |
|--------|---------|
| **joint_revision_success** | Strict all-or-nothing: decision + memory_state + evidence F1=1 + answer consistency + no stale reuse |
| **structural_revision_success** | decision + memory + evidence F1=1 + diagnosis (no answer requirement) |
| decision_macro_f1 | Primary decision metric (macro over classes) |
| memory_state_accuracy | Per-slot memory status match |
| evidence_f1 | Evidence set F1 vs gold |
| failure_diagnosis_accuracy | Diagnosis label match |
| minimal_evidence_exact_match | Predicted evidence set equals gold exactly |
| stale_reuse_rate | Stale memory reuse (lower better) |
| non_answer_decision_accuracy | escalate / refuse / clarify / mark_unresolved |
| scope_authority_accuracy | Authority/scope trap cases |
| answer_state_consistency | decision + answer rubric + memory aligned |

### Auxiliary (diagnostics only)

`black_box_decision_accuracy`, `decision_balanced_accuracy`, `use_current_memory_accuracy`, `answer_key_fact_accuracy`, `answer_exact_match`, `format_failure_rate`

## Script categories

### Workflows
- `workflows/audit_decision_boundary.py` — decision-boundary audit gate
- `workflows/validate_mempatch_bench_dataset.py` — schema validation
- `workflows/evaluate_mempatch_predictions.py` — score existing JSONL
- `workflows/run_paper_pipeline.sh` — audit → download → train Path B → Path A+B eval on test500

### Eval runners
- `eval/run_mempatch_memory_baselines.py` — RAG / full / mem0 / base
- `eval/run_mlx_revision_module_eval.py` — Path A
- `eval/run_mlx_lora_smoke_eval.py` — Path B
- `eval/run_mempatch_model.py` — DirectJudge (API)
- `eval/run_mempatch_revision_module.py` — revision module (scripted/prompt policy)

### Memory baseline helpers
- `memory/mempatch_memory_context.py` — context builders
- `memory/mempatch_mem0_local.py` — local Mem0 config (no OpenAI)

### MLX utilities
- `mlx/mlx_chat_utils.py`, `mlx/download_mlx_model.py`, `mlx/check_mlx_lora_model.py`

### Data / release
- `data/generate_mempatch.py`, `data/prepare_mempatch_v13_smoke.py`
- `data/build_paper_eval_bundle.py`, `data/package_mempatch_release.py`

### Analysis
- `analysis/analyze_mlx_lora_errors.py` — per-case error breakdown (optional)
