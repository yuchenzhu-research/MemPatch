# ReTrace-Bench Design

ReTrace-Bench evaluates the ability of LLM agents to perform revision authorization over a shared memory space. It focuses on the transition of belief nodes in response to subagent dialogue and evidence logs.

## Critical Design Pillars

1. **Topology Awareness**: Verifies that status updates correctly propagate across `DependencyEdge(REQUIRES)` topology paths.
2. **Path Precedence**: Asserts that `SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED` precedence rules are deterministically applied.
3. **Auditability**: Requires tracking back target beliefs to their grounding evidence source.
