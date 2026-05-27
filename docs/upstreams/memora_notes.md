# Memora Upstream Reconnaissance Notes

## Cloned Commit SHA and Observed License
- **Commit SHA**: `a454af42b7dffc21c9106d1020530599fd3d6558`
- **Observed License**: Apache License 2.0 (found in `reference/Memora/LICENSE`)

## Entrypoint Commands and Entrypoint Source Files
- **Track 1 (Direct LLM Evaluation) Entrypoint**:
  - Command:
    ```bash
    uv run python evals/model_eval/model_based_evaluator.py \
      --sessions-dir data/weekly/academic_researcher \
      --model anthropic/claude-sonnet-4.5 \
      --limit 5
    ```
  - File: [model_based_evaluator.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/Memora/evals/model_eval/model_based_evaluator.py)
- **Track 2 (Memory Agent Ingestion) Entrypoint**:
  - Command:
    ```bash
    python conversation_to_memory.py \
      --system mem_0 \
      --user-id academic_researcher_weekly \
      --conversation-directory data/weekly/academic_researcher/conversations
    ```
  - File: [conversation_to_memory.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/Memora/evals/agent_eval/conversation_to_memory.py)
- **Track 2 (Memory Agent Answering) Entrypoint**:
  - Command:
    ```bash
    python memory_to_answer.py \
      data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \
      --system mem_0 \
      --user-id academic_researcher_weekly
    ```
  - File: [memory_to_answer.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/Memora/evals/agent_eval/memory_to_answer.py)
- **Results Aggregator**:
  - Command:
    ```bash
    uv run python evals/model_eval/aggregate_results.py data/ --print
    ```
  - File: [aggregate_results.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/Memora/evals/model_eval/aggregate_results.py)

## Input/Output Formats and Dataset Structure
- **Dataset Structure (Conversations)**:
  - Located under `data/<period>/<persona>/conversations/session_NNNN.json`
  - Fields:
    - `session_id`: Integer, in chronological order.
    - `session_type`: e.g. `"no_memory"`, `"memory_introduction"`, `"memory_update"`, `"memory_deletion"`.
    - `operation` & `operation_details`: Describes what changes or inserts occurred in this session.
    - `date`: Session date (`YYYY-MM-DD`).
    - `persona`: Name of the persona.
    - `conversation`: List of dialogue turns. Each turn has `turn` index, `speaker` (`"user"` or `"ai_agent"`), `message` string, and `share_memory` boolean indicating if new memory-governing facts are introduced.
- **Dataset Structure (Questions)**:
  - Located at `data/<period>/<persona>/evaluation_questions_<persona>.json`
  - Divided into three task categories under `questions`:
    - `remembering`: Retrieves raw facts.
    - `reasoning`: Connects multiple facts.
    - `recommending`: Direct recommendations.
  - Each question contains:
    - `question_id`: Unique identifier.
    - `question`: Probing query text.
    - `question_date`: Active date of the question.
    - `evaluation`: Scoring rubric, containing `evaluation_questions` (list of sub-questions with `expected_answer` "yes" / "no" and `evaluation_type` `"memory_presence"` or `"forgetting_absence"`).
- **Scoring Output Format (FAMA)**:
  - Writes detailed results to `eval_results_<TIMESTAMP>.json` and reports to `eval_report_<TIMESTAMP>.json` inside `eval_results/<run_id>/`.
  - Calculates FAMA (Forgetting-Aware Memory Accuracy):
    $$\text{FAMA} = \max(0, \text{MPA} - \lambda \cdot (1 - \text{FAA}))$$
    where $\text{MPA}$ is the memory presence accuracy, $\text{FAA}$ is the forgetting absence accuracy, and $\lambda = \frac{N_{\text{forget}}}{N_{\text{presence}} + N_{\text{forget}}}$.

## Dependencies
- **API Keys / Services**:
  - OpenRouter API (configured via `OPENROUTER_API_KEY` / `OPEN_ROUTER_API_KEY` for Track 1 models and judges).
  - Target model keys (e.g. `OPENAI_API_KEY` for Track 2 `gpt-4o-mini`).
- **Third-Party Libraries**:
  - Python 3.11+, `uv` package manager.
  - `openai`, `httpx`, openrouter SDK.

## Files/Logic to ONLY Wrap and NOT Copy
Memora's evaluator loop and FAMA calculator should be adapted or wrapped, avoiding copying their framework codebase directly:
- `reference/Memora/evals/agent_eval/base_evaluator.py` (Memory agent abstract base class `BaseMemorySystem` and evaluator)
- `reference/Memora/evals/agent_eval/conversation_to_memory.py` (Ingestion pipeline)
- `reference/Memora/evals/agent_eval/memory_to_answer.py` (Evaluation answering loop)
- `reference/Memora/evals/model_eval/model_based_evaluator.py` (Model evaluator)
- `reference/Memora/evals/model_eval/aggregate_results.py` (FAMA scoring and result aggregation)
