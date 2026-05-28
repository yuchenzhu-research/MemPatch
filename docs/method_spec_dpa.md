# Method Spec: Typed DPA Runtime

This is the technical authority for canonical ReTrace runtime semantics.

## Method Core

ReTrace separates historical evidence from current-use authorization:

```text
immutable evidence ledger
→ typed belief / condition / edge graph
→ RevisionGate structural admission
→ deterministic Defeat-Path Authorization Algorithm
→ query-conditioned authorized basis
```

Semantic models may propose local objects or edges. They do not decide final
belief authorization. DPA never calls semantic models.

## Canonical Objects

### `EvidenceNode`

An append-only evidence record with source provenance:

- `evidence_id`
- `session_id`
- `timestamp`
- `text`
- `source_dataset`
- `source_pointer`
- `is_raw_source`
- `metadata`

Evidence is not deleted or rewritten to perform revision.

### `BeliefNode`

An open-text proposition derived from evidence:

- `belief_id`
- `proposition`
- `source_evidence_ids`
- `source_span`
- extractor provenance and confidence fields.

`AUTHORIZED` means a belief is eligible to participate in the current answer
basis. It does not mean ReTrace has newly proved that the proposition is true.

### `ConditionNode`

A scoped prerequisite for current use of a belief:

- `condition_id`
- `scope_id`
- `text`

The same condition text in different scopes must not be merged accidentally.

## Canonical Edges

### Dependency Edge

Only one dependency edge type is canonical:

```text
DependencyEdge(REQUIRES): belief -> condition
```

It records the prerequisite conditions for using a belief.

### Evidence Edges

Only these evidence edges are canonical:

- `BLOCKS`: evidence -> condition.
- `RELEASES`: evidence -> condition.
- `SUPERSEDES`: evidence -> prior belief, with `replacement_belief_id`.
- `REAFFIRMS`: evidence -> belief.
- `UNCERTAIN`: evidence -> belief.

`SUPERSEDES` must name a real replacement belief. In Stage A prompt parsing,
the replacement must be among the supplied replacement candidates and grounded
in the current `new_evidence`.

## RevisionGate

`RevisionGate` is a structural admission layer. It checks whether a proposed
typed edge is well-formed against the current `BeliefStore`.

As currently implemented:

- `DependencyEdge` admission requires non-empty ids, `edge_type == "REQUIRES"`,
  existing belief and condition targets, and inducer provenance.
- evidence-edge admission requires non-empty `edge_id`, non-empty
  `evidence_id`, verifier provenance, valid edge type, valid target kind, and
  existing target membership.
- `BLOCKS` and `RELEASES` admission additionally requires target kind
  `condition` and an existing target condition.
- `SUPERSEDES` admission requires target kind `belief`, an existing target
  belief, a non-null replacement id, a replacement different from the target,
  and an existing replacement belief.
- `REAFFIRMS` and `UNCERTAIN` admission require target kind `belief` and an
  existing target belief.
- non-`SUPERSEDES` edges must not carry `replacement_belief_id`.

The gate does not decide belief authorization. A structurally valid `BLOCKS` or
`RELEASES` edge may enter the graph when its target condition exists. It affects
a particular belief only later, when DPA finds an admitted
`REQUIRES(belief, condition)` path.

## DPA

`DefeatPathAuthorizationAlgorithm` consumes the admitted typed graph and
ledger ordering. It computes final authorization for one belief at a time.

Outputs:

- `AUTHORIZED`
- `BLOCKED`
- `SUPERSEDED`
- `UNRESOLVED`

Precedence:

```text
SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED
```

DPA is deterministic. Temporal tie-breaking is based on evidence time, ledger
position, and edge id through the temporal-validity layer. Missing evidence is
not fabricated.

## Supersession

A valid, temporally active `SUPERSEDES(evidence, prior_belief)` edge defeats the
prior belief and returns `SUPERSEDED`.

The accepted defeat path records:

- the superseding evidence edge id;
- the target prior belief id;
- the grounded `replacement_belief_id`.

The old belief remains in the graph for audit and historical queries.

## Prerequisite Blocking

DPA checks each admitted `REQUIRES(belief, condition)` edge for the target
belief. For each required condition, it considers valid `BLOCKS` and `RELEASES`
updates targeting that condition.

If the latest valid update for a required condition is `BLOCKS`, DPA returns
`BLOCKED` and records a prerequisite-block defeat path containing:

- the dependency edge id;
- the blocking evidence edge id.

If a `BLOCKS` edge targets a condition not required by the belief being
authorized, it does not affect that belief.

## Release

`RELEASES` clears a condition blocker for DPA purposes when it is the latest
valid update for that condition.

`RELEASES` does not assert that the belief is currently true. It only removes
the active prerequisite-block path if no higher-precedence defeat exists.

`SUPERSEDES` still wins over any release.

## Uncertainty and Reaffirmation

`UNCERTAIN(evidence, belief)` produces `UNRESOLVED` when it is the latest valid
belief-level status edge after supersession and prerequisite blocking have been
checked.

`REAFFIRMS(evidence, belief)` clears earlier uncertainty when it is the latest
valid belief-level status edge. It does not override a later supersession or an
active prerequisite block.

## Reversibility and Provenance

Revision changes authorization, not historical evidence. A blocked,
superseded, or unresolved belief remains auditable through:

- original source evidence ids;
- admitted dependency edges;
- admitted evidence edges;
- accepted and considered defeat paths;
- model call trace ids where semantic components produced proposals.

An authorized belief has `accepted_defeat_path = None` and records supporting
evidence ids.

## Non-Canonical Runtime Semantics

Do not use these as current method semantics:

- flat `RelationPrediction` method design;
- `SUPPORT` as a Stage A evidence edge;
- `CONDITION` as a Stage A relation label;
- `REQUIRED_BY` as the current dependency edge name;
- heuristic relation verification as the paper method.
