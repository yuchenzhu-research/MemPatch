# Reference Strategy

This document outlines the design reference storage policy for the ReTrace repository.

## Reference Organization

1. **Domain Memory Benchmarks**:
   - Stored in [references/agent_memory/](file:///Users/yuchenzhu/Desktop/ReTrace/references/agent_memory/)
   - Focuses on related academic benchmarks like STALE, Memora, and MemConflict.

2. **Top Benchmark Engineering**:
   - Stored in [references/top_benchmarks/](file:///Users/yuchenzhu/Desktop/ReTrace/references/top_benchmarks/)
   - Reviews mature industrial frameworks like SWE-bench, WebArena, and OSWorld.

## Guardrails Against Repository Bloat

- **Registry Over Cloning**: Always prefer YAML pointers and Markdown analysis logs over cloning large repository history.
- **Gitignore Exclusions**: If a local clone of an external repository is required for diagnostic purposes, it must be located within the ignored `.external_repos/` or `.reference_cache/` folder.
- **No External Commits**: Never vendor or commit external codebases directly to preserve clean branch history.
