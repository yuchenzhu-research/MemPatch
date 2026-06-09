# Scripts map

One evaluator (`benchmark.api.evaluate_predictions`) scores every run. Outputs go under `local/` (gitignored).

## Experiment lines (what to compare)

| Line | Script | What it does | Trained? |
|------|--------|--------------|----------|
| **External memory baselines** | `run_mempatch_memory_baselines.py` | `build_prompt` → MLX → five-field JSON. Backends: `base`, `full`, `rag`, `mem0`. | No |
| **DirectJudge** | `run_mempatch_model.py` | API/MLX provider → five-field JSON (same schema as baselines). | Optional |
| **Path A** (`\baseproj{}`) | `run_mlx_revision_module_eval.py` | Revision view → **typed actions JSON** → DPA → projection → five fields. | No (base MLX) |
| **Path B** (`\adapted{}`) | `run_mlx_lora_smoke_eval.py` | SFT prompt → MLX+LoRA → **direct five-field JSON**. | Yes (LoRA) |

**Fair comparisons**

- vs external memory: **RAG / Full / mem0** vs **Path B** (same JSON protocol; align backbone + split).
- vs untrained module: **Path A** vs **Path B** (same stack; only training differs).
- Path A joint≈0 means *untrained proposer fails*; Path B joint≈0.54 on test500 means *LoRA is what works*.

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

### Eval runners
- `run_mempatch_memory_baselines.py` — RAG / full / mem0 / base
- `run_mlx_revision_module_eval.py` — Path A
- `run_mlx_lora_smoke_eval.py` — Path B
- `run_mempatch_model.py` — DirectJudge (API)
- `run_mempatch_revision_module.py` — revision module (scripted/prompt policy)
- `evaluate_mempatch_predictions.py` — score existing JSONL

### Memory baseline helpers
- `mempatch_memory_context.py` — context builders
- `mempatch_mem0_local.py` — local Mem0 config (no OpenAI)

### MLX utilities
- `mlx_chat_utils.py`, `download_mlx_model.py`, `check_mlx_lora_model.py`

### Data / release
- `generate_mempatch.py`, `audit_decision_boundary.py`, `validate_mempatch_bench_dataset.py`
- `prepare_mempatch_v13_smoke.py`, `build_paper_eval_bundle.py`, `package_mempatch_release.py`

### Pipeline
- `run_paper_pipeline.sh` — audit → download → train Path B → Path A+B eval on test500

### Analysis
- `analyze_mlx_lora_errors.py` — per-case error breakdown (optional)
