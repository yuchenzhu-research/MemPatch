# STALE Upstream Reconnaissance Notes

## Cloned Commit SHA and Observed License
- **Commit SHA**: `ea7d391103a151927cd29d2f01d87597a782bdcb`
- **Observed License**: MIT License (found in `reference/STALE/LICENSE`)

## Entrypoint Commands and Entrypoint Source Files
- **Dataset Generation Entrypoint**:
  - Command:
    ```bash
    python Generation/StepALL_IC_gen.py \
      --seed-file data/ontology_seeds_demo5.json \
      --output-name demo_T1 \
      --conflict-type T1 \
      --output-dir outputs \
      --num-workers 1
    ```
  - File: [StepALL_IC_gen.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/STALE/STALE/Generation/StepALL_IC_gen.py)
- **Target Model Evaluator**:
  - Command:
    ```bash
    python Evaluation/run_target_model.py \
      --icds-path outputs/demo_T1_MAIN.json \
      --output-path outputs/demo_T1_answers.json
    ```
  - File: [run_target_model.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/STALE/STALE/Evaluation/run_target_model.py)
- **Judge / Performance Summarizer**:
  - Command:
    ```bash
    python Evaluation/full_eval_performance.py \
      --answers-path outputs/demo_T1_answers.json \
      --dataset-path outputs/demo_T1_MAIN.json \
      --output-path outputs/demo_T1_eval.json \
      --conflict-type T1 \
      --model-method target_model
    ```
  - File: [full_eval_performance.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/STALE/STALE/Evaluation/full_eval_performance.py)
- **CUP-Mem Sample Runner**:
  - Command:
    ```bash
    python -m cup_mem.run_cup_mem \
      --data-path STALE/outputs/demo_T1_MAIN.json \
      --sample-index 0 \
      --session-mode relevant_only
    ```
  - File: [run_cup_mem.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/STALE/cup_mem/run_cup_mem.py)

## Input/Output Formats and Dataset Structure
- **Dataset File Formats**:
  - `<output-name>_MAIN.json`: Complete dataset entries with timestamps, old states, new states, and haystack sessions.
  - `<output-name>_EVID.json`: Contains evidence metadata (conflict pairs).
- **Dataset Structure (MAIN.json entry)**:
  - `uid` / `sample_id`: Unique identifier.
  - `M_old`: Old user state statement.
  - `M_new`: Updated user state statement (implicitly invalidates `M_old`).
  - `explanation`: Reason why `M_new` invalidates `M_old`.
  - `probing_queries`: Dict with three query dimensions:
    - `dim1_query` (State Resolution, SR): Queries the current state (must reflect `M_new`).
    - `dim2_query` (Premise Resistance, PR): Queries using `M_old` as a premise (must resist the invalid premise).
    - `dim3_query` (Implicit Policy Adaptation, IPA): Queries recommendations based on current state (must adapt to `M_new`).
  - `haystack_session`: List of noise conversations injected before, between, and after `M_old` and `M_new`.
  - `timestamps`: Timestamps corresponding to each session.
  - `query_time`: Final query timestamp (defines the cutoff/evaluation moment).
- **Target Model Answer Format**:
  - Dict or list containing `uid` and `target_model_responses` (with `dim1_response`, `dim2_response`, `dim3_response`).

## Dependencies
- **API Keys / Services**:
  - OpenAI-compatible endpoint or Responses API (configured via `TARGET_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` in `.env`).
- **Third-Party Libraries**:
  - Python 3.10+
  - `openai`, `httpx`, `torch`, `transformers`
  - Local sentence embedding model: `all-MiniLM-L6-v2` (for CUP-Mem vector retrieval).

## Files/Logic to ONLY Wrap and NOT Copy
The core implementation of ReTrace must not inherit CUP-Mem's fixed-slot ontology or duplicate its logic directly. Instead, they must be wrapped or adapted under `retracemem/` adapters:
- `reference/STALE/STALE/Evaluation/run_target_model.py` (Prompt construction & trimming)
- `reference/STALE/STALE/Evaluation/full_eval_performance.py` (Evaluation judging and aggregation)
- `reference/STALE/STALE/Evaluation/judge_prompts.py` (Evaluation rubrics)
- `reference/STALE/cup_mem/pipeline.py` / `cup_mem/core/sample_runner.py` (CUP-Mem execution flows)
- `reference/STALE/cup_mem/memory/models.py` (Profile items and invalidation proposals)
