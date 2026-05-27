# nemori Upstream Reconnaissance Notes

## Cloned Commit SHA and Observed License
- **Commit SHA**: `d2a6dff6e5481214a0be6a2d10147feccfc16244`
- **Observed License**: MIT License (found in `reference/nemori/LICENSE`)

## Entrypoint Commands and Entrypoint Source Files
- **Quickstart Entrypoint**:
  - Command:
    ```bash
    python examples/quickstart.py
    ```
  - File: [quickstart.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/examples/quickstart.py)
- **LoCoMo Ingestion Step**:
  - Command:
    ```bash
    PYTHONPATH=. python evaluation/locomo/add.py
    ```
  - File: [add.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/evaluation/locomo/add.py)
- **LoCoMo Search Step**:
  - Command:
    ```bash
    PYTHONPATH=. python evaluation/locomo/search.py
    ```
  - File: [search.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/evaluation/locomo/search.py)
- **LoCoMo Evaluation Step**:
  - Command:
    ```bash
    PYTHONPATH=. python evaluation/locomo/evals.py
    ```
  - File: [evals.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/evaluation/locomo/evals.py)
- **LoCoMo Score Aggregation**:
  - Command:
    ```bash
    PYTHONPATH=. python evaluation/locomo/generate_scores.py
    ```
  - File: [generate_scores.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/evaluation/locomo/generate_scores.py)
- **Unified Library Interface**:
  - File: [facade.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/nemori/nemori/api/facade.py) (Exposes `NemoriMemory` & `MemoryConfig`)

## Input/Output Formats and Dataset Structure
- **Input (facade.add_messages)**:
  - Takes a list of messages. Each message is a dictionary containing:
    - `role`: String (e.g. `"user"`, `"assistant"`, or custom speaker name).
    - `content`: String message content.
    - `timestamp`: ISO-8601 formatted datetime string (optional).
    - `metadata`: Dictionary of custom metadata.
- **LoCoMo Dataset Structure**:
  - Located in `evaluation/locomo/`.
  - JSON containing conversations under `conversation` key:
    - `speaker_a` and `speaker_b` definitions.
    - Conversation items labeled with turns and timestamps (e.g., `turn_1`, `turn_1_date_time`).
    - Turns contain `speaker`, `text`, and optional `blip_caption` or `query` metadata.
- **Output Formats**:
  - Search results returned as lists of matched memory documents or segments.
  - In LoCoMo, evaluation results are written to JSON files summarizing correctness against ground-truth templates.

## Dependencies
- **API Keys / Services**:
  - LLM and Embedding models (configured via `LLM_API_KEY`, `LLM_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` in `.env`).
- **External Services**:
  - PostgreSQL (used for metadata and text GIN index search).
  - Qdrant (used for vector storage and retrieval).
- **Third-Party Libraries**:
  - `asyncpg`, `qdrant-client`, `openai`, `pydantic`, `pillow`, `tqdm`.

## Files/Logic to ONLY Wrap and NOT Copy
The PostgreSQL database store, Qdrant vectors, and Event Segmentation boundary logic must not be copied. If used as a baseline, wrap the facade or borrow the message/episode mapping:
- `reference/nemori/nemori/domain/models.py` (`Message -> Episode -> SemanticMemory` models)
- `reference/nemori/nemori/core/memory_system.py` (Boundary alignment write pipeline)
- `reference/nemori/nemori/search/unified.py` (Episodic + semantic hybrid search)
- `reference/nemori/nemori/api/facade.py` (Unified facade wrapper)
