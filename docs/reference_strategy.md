# Reference Strategy

This document outlines the design reference storage policy for the ReTrace repository.

## Reference Organization

1. **Domain Memory Benchmarks**:
   - Stored in [references/agent_memory/](../references/agent_memory/)
   - Focuses on related academic benchmarks like STALE, Memora, and MemConflict.

2. **Top Benchmark Engineering**:
   - Stored in [references/top_benchmarks/](../references/top_benchmarks/)
   - Reviews mature industrial frameworks like SWE-bench, WebArena, and OSWorld.

## Tracked References Vs. Local Source Inspection

- `references/` is tracked. It contains only lightweight registries, URLs, and
  short Markdown notes.
- `.external_repos/` and `.reference_cache/` are ignored. They are only for
  local cloned repositories, downloaded papers, or temporary source inspection.
  Do not treat them as project source and never commit them.

## Guardrails Against Repository Bloat

- **Registry Over Cloning**: Always prefer YAML pointers and Markdown analysis logs over cloning large repository history.
- **Gitignore Exclusions**: If a local clone of an external repository is
  required for diagnostic purposes, it must be located within ignored
  `.external_repos/` or `.reference_cache/`.
- **No External Commits**: Never vendor or commit external codebases directly to preserve clean branch history.
