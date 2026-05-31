"""Stage C (ReTrace-AdaptiveProposer) API-ICL exemplar contract — fail-closed.

Stage C still proposes *typed actions only*; RevisionGate and the deterministic
DPA / ``authorize(...)`` path are unchanged. API-ICL improves only the proposer
by conditioning it on a few in-context exemplars of well-formed typed revisions.

Per ``AGENTS.md`` (Paper 1 Stage C Training Boundary), exemplars used for any
live smoke or training export must be **human-approved**:

* tied to an immutable review-pack manifest hash (``source_manifest_sha256``);
* carry an explicit human approval decision (``approval.decision == "approved"``);
* contain only *method-visible* fields and must **never** include evaluator
  final statuses or gold typed revision targets;
* every proposed action must cite the visible new evidence that grounds it
  (non-empty ``evidence_ids``).

This module loads/validates exemplar packs and **fails closed** when these
conditions are not met. It deliberately does not auto-promote pending packs and
does not call any model — wiring an approved pack into a live API-ICL proposer
run is the documented next step (see ``docs/stage_c_api_icl.md``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retracemem.proposers.replay import CANONICAL_ACTIONS

# Fields that would leak evaluation gold / evaluator decisions into a method
# input. Their presence anywhere in an exemplar is a hard, fail-closed error.
FORBIDDEN_EXEMPLAR_FIELDS: frozenset[str] = frozenset({
    "gold_final_status",
    "gold_final_statuses",
    "final_belief_statuses",
    "gold_typed_targets",
    "gold_revision_targets",
    "evaluator_status",
    "gold_snapshot",
    "belief_statuses",
    "relevant_session_index",
    "m_old",
    "m_new",
})

REQUIRED_EXEMPLAR_FIELDS: tuple[str, ...] = (
    "exemplar_id",
    "candidate_view_summary",
    "submission_evidence",
    "proposed_actions",
)


class ExemplarPackError(ValueError):
    """Base error for malformed or unsafe Stage C exemplar packs."""


class ExemplarApprovalError(ExemplarPackError):
    """Raised when a pack lacks a valid recorded human approval / manifest hash."""


class ExemplarLeakageError(ExemplarPackError):
    """Raised when an exemplar contains forbidden gold / evaluator fields."""


class ExemplarSchemaError(ExemplarPackError):
    """Raised when an exemplar is missing required fields or has invalid actions."""


def _scan_forbidden(obj: Any, path: str = "") -> None:
    """Recursively reject any forbidden (gold/evaluator) key, case-insensitively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in FORBIDDEN_EXEMPLAR_FIELDS:
                raise ExemplarLeakageError(
                    f"Forbidden gold/evaluator field '{k}' at '{path or '<root>'}'. "
                    "Stage C exemplars must contain method-visible fields only."
                )
            _scan_forbidden(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_forbidden(item, f"{path}[{i}]")


def validate_exemplar(exemplar: dict[str, Any]) -> None:
    """Validate one exemplar: no leakage, required fields present, actions grounded.

    Raises an :class:`ExemplarPackError` subclass on the first violation.
    """
    _scan_forbidden(exemplar)

    missing = [f for f in REQUIRED_EXEMPLAR_FIELDS if f not in exemplar]
    if missing:
        raise ExemplarSchemaError(
            f"Exemplar '{exemplar.get('exemplar_id', '<unknown>')}' missing required "
            f"fields: {', '.join(missing)}."
        )

    actions = exemplar.get("proposed_actions")
    if not isinstance(actions, list) or not actions:
        raise ExemplarSchemaError(
            f"Exemplar '{exemplar['exemplar_id']}' must list at least one proposed action."
        )
    for act in actions:
        atype = act.get("action_type")
        if atype not in CANONICAL_ACTIONS:
            raise ExemplarSchemaError(
                f"Exemplar '{exemplar['exemplar_id']}' has non-canonical action_type "
                f"'{atype}'. Allowed: {', '.join(CANONICAL_ACTIONS)}."
            )
        # NO_REVISION is the only action that need not cite new evidence; every
        # real revision must explicitly ground itself in visible new evidence.
        if atype != "NO_REVISION" and not act.get("evidence_ids"):
            raise ExemplarSchemaError(
                f"Exemplar '{exemplar['exemplar_id']}' action '{atype}' must cite "
                "visible new evidence via non-empty 'evidence_ids'."
            )


def validate_pack_approval(pack: dict[str, Any]) -> None:
    """Fail closed unless the pack carries a recorded human approval + manifest hash."""
    manifest_hash = pack.get("source_manifest_sha256")
    if not manifest_hash:
        raise ExemplarApprovalError(
            "Exemplar pack is missing 'source_manifest_sha256' (the immutable "
            "review-pack manifest hash). Cannot use for live API-ICL."
        )
    approval = pack.get("approval") or {}
    if approval.get("decision") != "approved":
        raise ExemplarApprovalError(
            "Exemplar pack is not human-approved "
            f"(approval.decision={approval.get('decision')!r}). Pending or rejected "
            "packs must not be used for live smoke or training. No auto-promotion."
        )
    if not approval.get("reviewer") or not approval.get("reviewed_at"):
        raise ExemplarApprovalError(
            "Approved pack must record 'approval.reviewer' and 'approval.reviewed_at'."
        )


def load_approved_exemplars(path: str | Path) -> list[dict[str, Any]]:
    """Load and fully validate an approved Stage C exemplar pack (fail-closed).

    Returns the list of exemplars only when the pack is human-approved, hash-tied,
    leakage-free, and schema-valid. Otherwise raises an :class:`ExemplarPackError`.
    """
    path = Path(path)
    if not path.exists():
        raise ExemplarApprovalError(
            f"No exemplar pack at {path}. Live API-ICL requires a human-approved "
            "pack; see docs/stage_c_api_icl.md for the required format."
        )
    pack = json.loads(path.read_text())
    validate_pack_approval(pack)
    exemplars = pack.get("exemplars")
    if not isinstance(exemplars, list) or not exemplars:
        raise ExemplarSchemaError("Approved pack contains no exemplars.")
    for ex in exemplars:
        validate_exemplar(ex)
    return exemplars


def select_icl_exemplars(
    exemplars: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    """Deterministically select up to ``k`` exemplars.

    Prefers diversity across ``failure_category`` (one per category first, in
    sorted order) then fills remaining slots in stable order. Deterministic so
    runs are reproducible.
    """
    if k <= 0:
        return []
    by_category: dict[str, list[dict[str, Any]]] = {}
    for ex in exemplars:
        cat = str(ex.get("failure_category", ""))
        by_category.setdefault(cat, []).append(ex)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for cat in sorted(by_category):
        ex = by_category[cat][0]
        selected.append(ex)
        seen_ids.add(ex["exemplar_id"])
        if len(selected) >= k:
            return selected[:k]
    for ex in exemplars:
        if ex["exemplar_id"] not in seen_ids:
            selected.append(ex)
            seen_ids.add(ex["exemplar_id"])
            if len(selected) >= k:
                break
    return selected[:k]


def format_exemplars_for_prompt(exemplars: list[dict[str, Any]]) -> str:
    """Render exemplars as a typed-action ICL block for the proposer prompt.

    Output is illustrative typed-action format only; it never includes any
    final-status / gold information (those fields are rejected at load time).
    """
    blocks: list[str] = []
    for i, ex in enumerate(exemplars, 1):
        ev = "; ".join(
            f"{e.get('evidence_id', '?')}: {e.get('content', '')}"
            for e in ex.get("submission_evidence", [])
        )
        actions = json.dumps(ex.get("proposed_actions", []), indent=2)
        blocks.append(
            f"### Exemplar {i} ({ex.get('exemplar_id')})\n"
            f"Candidate view: {ex.get('candidate_view_summary')}\n"
            f"New evidence: {ev}\n"
            f"Proposed typed actions:\n{actions}"
        )
    return "\n\n".join(blocks)
