# ReTrace-Bench — Human Annotation Codebook

Definitions annotators must follow. Read this fully before annotating. When in
doubt, prefer the more conservative non-answer decision and explain in `notes`.

## Task framing

Each scenario shows an immutable **event trace** (ordered events with actor role,
trust level, visibility scope) and an **initial memory** (belief entries with
scope). Newer events may confirm, contradict, scope-limit, block, release, or
supersede earlier beliefs. Your job is to decide what a careful agent should do
**now**, citing only the minimal evidence needed.

## Decision labels

- **use_current_memory** — the visible evidence supports answering now from a
  currently-valid belief. Use only when the supporting belief is not
  contradicted, blocked, out-of-scope, or superseded by later evidence.
- **mark_unresolved** — the evidence is internally conflicting or insufficient to
  commit to an answer, and no clarification from the user would resolve it (the
  conflict is in the record itself). The correct state is "unknown / unresolved".
- **ask_clarification** — the scenario is under-specified in a way a **user**
  could resolve (e.g. ambiguous target/scope), and answering would require
  guessing user intent. Use when a single clarifying question would unblock you.
- **refuse_due_to_policy** — answering would violate a stated policy/security
  constraint (e.g. acting on an unauthorized override, exposing restricted-scope
  info). Refusal is required regardless of how "answerable" the content seems.
- **escalate** — resolution requires an authority/owner action beyond the agent
  and beyond a simple user clarification (e.g. a maintainer must adjudicate a
  conflict between a trusted and untrusted actor, or a release decision is owned
  by someone else).

### Choosing among the non-answer decisions
- Conflict lives **in the record** and no one can clarify it away → `mark_unresolved`.
- Missing info a **user** can supply → `ask_clarification`.
- A **policy/security rule** forbids answering → `refuse_due_to_policy`.
- A **different authority/owner** must act → `escalate`.

## Memory status labels (`memory_state_json`)

For each relevant `memory_id`, assign one:
- **current** — belief is valid and usable now.
- **outdated** — superseded by newer evidence; no longer usable as the answer.
- **blocked** — a prerequisite condition is currently blocked, so the belief may
  not be used even though it was never directly contradicted.
- **unresolved** — evidence is conflicting/insufficient to decide validity.
- **out_of_scope** — belief belongs to a different version/branch/scope than the
  current question; not applicable here.
- **deleted** — belief was explicitly removed/retracted.
- **should_not_store** — content that should never have been written to memory
  (e.g. unverified, policy-restricted) and must not be relied upon.
- **restored** — a previously outdated/blocked belief that later evidence has
  re-validated.

Annotate the memory entries the decision actually depends on; you need not label
obviously irrelevant filler memories.

## Failure-diagnosis labels

Pick the single best description of the *trap* the scenario sets for a naive
agent:
- **stale_memory_reuse** — reusing an old belief after newer evidence invalidated it.
- **under_update** — failing to update memory when evidence clearly required it.
- **over_update** — updating/overwriting memory when evidence did not justify it.
- **conflict_collapse** — collapsing a genuine conflict into one side instead of
  marking it unresolved.
- **scope_leakage** — applying a belief from one version/branch/scope to another.
- **policy_violation** — acting against a stated policy/security constraint.
- **wrong_source_attribution** — trusting the wrong actor/source (e.g. untrusted
  over verified, or misattributing which event established a fact).
- **memory_hallucination** — relying on a belief not supported by any event.
- **unnecessary_memory_write** — writing something to memory that should not be
  stored.
- **failure_to_forget** — keeping a belief that should have been deleted/retracted.
- **failure_to_release_or_restore** — not re-validating a belief once a blocking
  condition was released.

## Minimal evidence

`evidence_event_ids` should be the **smallest** set of events that justifies your
decision and answer. Do **not** list every related event. If two events are
jointly required (e.g. the claim *and* the later contradiction), list both; if
one event alone is decisive, list only that one. Over-citing is penalized.

## Cross-scope evidence
Only count evidence as supporting a belief if it is in the **same scope**
(version/branch/component) as the question. Evidence from another scope can make
a belief `out_of_scope` but does not make it `current` for this question.

## Trust / authority
- Prefer **verified/trusted** actors over **untrusted** ones; an untrusted claim
  does not establish a fact.
- A **maintainer/owner correction** overrides a non-owner user claim.
- When a trusted and an authoritative source genuinely conflict and neither can
  be dismissed from the record, use `mark_unresolved` or `escalate` per the rules
  above — do not silently pick one (that is `conflict_collapse`).

## Specific recurring situations
- **PR opened but not merged** → the change is **not** in effect; a belief that
  assumes it merged is `outdated`/`unresolved`, not `current`.
- **Merged but unreleased** → code is merged but not yet in a release; answers
  about released behaviour must not assume it.
- **Release then revert** → the feature/fix is **not** active after the revert;
  treat post-revert state as authoritative (`restored`/`outdated` as appropriate).
- **Backport-only fix** → the fix exists only on the backport branch/version;
  other versions remain affected (scope matters).
- **CI failed after a success claim** → a later failing CI defeats an earlier
  "it works" claim; do not `use_current_memory` on the stale success.
- **Stale comment after a new release** → an old comment does not describe the
  new release; treat as `outdated`/`out_of_scope`.
- **Policy/security override by an unauthorized actor** → `refuse_due_to_policy`
  and mark the relevant belief `should_not_store` if it was injected.

## Flagging issues
- If a scenario is **not solvable** from the visible evidence, set
  `solvable_from_visible_evidence = no` and explain.
- If the topic/domain feels inconsistent, set `topic_domain_consistent = no`.
- If there are **multiple equally valid** answers, set
  `ambiguous_or_multiple_valid_answers = yes`.
- If the trace is padded with irrelevant filler, set `filler_heavy = yes`.

These flags drive the dataset-quality rates; they do not need to match gold.
