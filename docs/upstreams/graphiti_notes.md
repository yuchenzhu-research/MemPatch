# graphiti Upstream Reconnaissance Notes

## Cloned Commit SHA and Observed License
- **Commit SHA**: `34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6`
- **Observed License**: Apache License 2.0 (found in `reference/graphiti/LICENSE`)

## Entrypoint Commands and Entrypoint Source Files
- **Azure OpenAI Quickstart**:
  - Command:
    ```bash
    python examples/azure-openai/azure_openai_neo4j.py
    ```
  - File: [azure_openai_neo4j.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/examples/azure-openai/azure_openai_neo4j.py)
- **Neo4j Quickstart**:
  - Command:
    ```bash
    python examples/quickstart/quickstart_neo4j.py
    ```
  - File: [quickstart_neo4j.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/examples/quickstart/quickstart_neo4j.py)
- **FalkorDB Quickstart**:
  - Command:
    ```bash
    python examples/quickstart/quickstart_falkordb.py
    ```
  - File: [quickstart_falkordb.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/examples/quickstart/quickstart_falkordb.py)
- **Kuzu Quickstart**:
  - Command:
    ```bash
    python examples/quickstart/quickstart_kuzu.py
    ```
  - File: [quickstart_kuzu.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/examples/quickstart/quickstart_kuzu.py)
- **MCP Server Entrypoint**:
  - File: [mcp_server/](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/mcp_server)
- **REST Service Entrypoint**:
  - File: [server/main.py](file:///Users/yuchenzhu/Desktop/ReTrace/reference/graphiti/server/main.py)

## Input/Output Formats and Dataset Structure
- **Input (add_episode)**:
  - Supports unstructured text or structured JSON payloads representing episodes.
  - Custom entities and edges can be defined via Pydantic models (ontology-based).
  - Facts are represented as triplets (Entity $\rightarrow$ Relation $\rightarrow$ Entity) with valid time and invalid time.
- **Output Formats**:
  - Evolving temporal graphs containing EpisodicNodes, EntityNodes, and EntityEdges with bi-temporal validity windows.
  - Search queries yield retrieved relationships, nodes, or path lists.

## Dependencies
- **API Keys / Services**:
  - OpenAI / Azure OpenAI / Gemini / Anthropic API Key (defaults to `OPENAI_API_KEY` for inference and embeddings).
- **External Services**:
  - Neo4j 5.26+, FalkorDB, Kuzu, or Amazon Neptune / OpenSearch Serverless.
- **Third-Party Libraries**:
  - Python 3.10+
  - `neo4j` (or other graph drivers), `openai`, `pydantic`, `fastapi`, `posthog` (telemetry).

## Files/Logic to ONLY Wrap and NOT Copy
Graphiti's temporal graph database drivers and ontology generation engines are complex and must not be copied into ReTrace's dependency-free core. Only wrappers or design patterns should be adapted:
- `reference/graphiti/graphiti_core/nodes.py` (`EpisodicNode` and entity schema definition)
- `reference/graphiti/graphiti_core/edges.py` (`EntityEdge` schema, temporal validity windows: `valid_at`, `invalid_at`)
- `reference/graphiti/graphiti_core/graphiti.py` (`add_episode` pipeline: extraction, deduplication, invalidation, hydration)
- `reference/graphiti/graphiti_core/search/search_config_recipes.py` (Hybrid BM25 + cosine + BFS search fusion)
