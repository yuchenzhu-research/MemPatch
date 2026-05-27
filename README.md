# ReTrace

ReTrace is the working codebase for an evidence-preserving reversible belief
revision method for dynamic agent memory.

Current scope:

- Preserve episodic evidence immutably.
- Maintain open-text belief propositions.
- Verify local evidence-to-belief relations.
- Authorize current beliefs through a conservative TMS-inspired gate.
- Evaluate first on STALE and Memora.

External repositories are kept under `reference/` and should not be edited as
part of the local implementation.

## Current Documents

- `implementation_plan.md`: execution plan and milestone ownership.
- `docs/project_logic.md`: research alignment and method boundaries.
- `docs/coding_contract.md`: coding rules for all future agents.
- `docs/agent_handoff.md`: short handoff instructions for Gemini/Opus/Codex.
- `docs/today_execution_plan.md`: concrete first-version completion plan.
- `docs/code_direction.md`: code direction from the paper blueprint.
- `docs/reference_integration_map.md`: how cloned references map into local
  modules.

## Verification

No dependency check:

```bash
python3 -m compileall -q retracemem tests
```

After a local test environment includes pytest:

```bash
python3 -m pytest -q
```
