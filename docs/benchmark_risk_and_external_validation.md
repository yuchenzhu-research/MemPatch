# Benchmark Risk & External Validation

## Internal Synthetic vs External Validation

ReTrace-Learn training relies on two distinct data sources with different
contamination risk profiles:

### Internal Synthetic Data (E1–E2)

- Hand-authored multi-subagent dialogues → graph extraction → gold typed
  revision actions → gold DPA final statuses.
- **Risk**: Author bias in action construction. Mitigated by:
  - Requiring explicit evidence grounding for every non-`NO_REVISION` action.
  - Running all gold actions through the deterministic backend to verify that
    the claimed gold final statuses are actually reproduced.
  - Separating authoring (teacher) from evaluation (DPA kernel).

### External Validation (E4 — STALE/CUPMem)

- STALE/CUPMem benchmarks use externally authored episodes with pre-defined
  gold memory states.
- **Risk**: Distribution shift — our typed action vocabulary may not map cleanly
  to external benchmarks designed for different memory models.
- **Mitigation**: The STALE adapter (`experiments/stale_adapter.py`) translates
  between external gold fields and our typed vocabulary. The adapter never feeds
  gold fields into method inputs; it only consumes them at scoring boundaries.

## Teacher Annotation Filtering

Only human-approved reviewed examples may be used for live smoke or training
export. The pipeline enforces:

1. **Review-pack manifest hash** — each approved batch is tied to an immutable
   hash of the reviewed examples.
2. **No auto-promotion** — scripts must not automatically promote pending
   examples to approved status.
3. **Approval record** — every example used in training or smoke testing must
   trace back to an explicit human decision record.

## Gold Leakage Prevention

The following boundaries prevent test-set leakage:

| Boundary | Enforcement |
| --- | --- |
| STALE adapters in `experiments/` only | `test_active_package_boundary.py` |
| `StaleGoldRecord` restricted to eval | Adapter interface strips gold fields |
| Three probing queries share frozen snapshot | Snapshot immutability in runner |
| Stage C policy never sees gold targets | Schema validation in `TypedRevisionExample.validate()` |

## DPA-in-the-Loop Training Reward

The reward signal flows from the deterministic backend, not from surface action
matching:

```text
Model proposes actions
    → Parser validates JSON structure (parser_error_penalty)
    → RevisionGate admits/rejects edges (gate_rejection_penalty)
    → DPA computes final statuses (final_status_reward)
    → Reward = f(final_status_agreement, grounding, memory_safety, error_penalties)
```

Structured `EngineError` objects from each stage feed directly into reward
computation, enabling fine-grained curriculum analysis (see
`retrace_learn.runtime.engine_errors`).
