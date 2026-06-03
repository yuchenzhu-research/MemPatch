"""ReTrace-Engine commit path: Parser + RevisionGate + DPA runtime.

(Implementation detail of stages 2-3, not a separate paper-level module.)

This wraps the deterministic kernel (``retracemem.authorize``) with the
ReTrace-Learn JSON parsing front-end. It is the program-only runtime that turns
a (possibly model-generated) action payload into final belief statuses plus a
fully auditable trace. It never learns and never overrides DPA: it parses,
validates structure, then defers all admission/authorization to ``authorize``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retracemem.authorization import authorize
from retracemem.methods.contracts import SharedCandidateView
from retracemem.multiagent.parser import (
    ParseErrorCode,
    StructuredParseError,
    extract_json_array,
)

from retrace_learn.schemas import RevisionAction, SchemaValidationError
from retrace_learn.runtime.engine_errors import (
    EngineError,
    EngineStage,
    ErrorSeverity,
    PARSER_INVALID_JSON,
    PARSER_ITEM_NOT_OBJECT,
    PARSER_SCHEMA_VIOLATION,
)
from retrace_learn.runtime.views import actions_to_proposal_batches


@dataclass(frozen=True)
class ParseResult:
    valid_json: bool
    schema_valid: bool
    actions: tuple[RevisionAction, ...]
    error_code: str | None = None
    error_message: str | None = None
    errors: tuple[EngineError, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_json": self.valid_json,
            "schema_valid": self.schema_valid,
            "n_actions": len(self.actions),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "errors": [e.to_dict() for e in self.errors],
        }


@dataclass(frozen=True)
class RuntimeResult:
    final_belief_statuses: dict[str, str]
    authorized_belief_ids: tuple[str, ...]
    excluded_belief_ids: tuple[str, ...]
    gate_decisions: list[dict[str, Any]]
    defeat_paths: list[dict[str, Any]]
    audit_trace: dict[str, Any]
    parse_result: ParseResult
    engine_errors: tuple[EngineError, ...] = ()

    @property
    def parser_errors(self) -> tuple[EngineError, ...]:
        return tuple(e for e in self.engine_errors if e.stage == EngineStage.PARSER)

    @property
    def gate_errors(self) -> tuple[EngineError, ...]:
        return tuple(e for e in self.engine_errors if e.stage == EngineStage.REVISION_GATE)

    @property
    def dpa_errors(self) -> tuple[EngineError, ...]:
        return tuple(e for e in self.engine_errors if e.stage == EngineStage.DPA)

    @property
    def warnings(self) -> tuple[EngineError, ...]:
        return tuple(e for e in self.engine_errors if e.severity == ErrorSeverity.WARNING)

    @property
    def admitted_actions(self) -> tuple[RevisionAction, ...]:
        admitted_ids = {
            d["edge_id"]
            for d in self.gate_decisions
            if d.get("admitted")
        }
        return tuple(a for a in self.parse_result.actions if a.edge_id in admitted_ids)

    @property
    def rejected_actions(self) -> tuple[RevisionAction, ...]:
        if not self.parse_result.schema_valid:
            return self.parse_result.actions
        admitted_ids = {
            d["edge_id"]
            for d in self.gate_decisions
            if d.get("admitted")
        }
        return tuple(a for a in self.parse_result.actions if a.edge_id not in admitted_ids)

    @property
    def final_statuses(self) -> dict[str, str]:
        return self.final_belief_statuses

    @property
    def failure_categories(self) -> list[str]:
        return sorted(list(set(e.code for e in self.engine_errors)))

    @property
    def reward_breakdown(self) -> dict[str, float]:
        parser_penalty = sum(-10.0 for e in self.parser_errors)
        gate_penalty = sum(-5.0 for e in self.gate_errors)
        dpa_penalty = sum(-2.0 for e in self.dpa_errors)
        return {
            "parser_penalty": parser_penalty,
            "gate_penalty": gate_penalty,
            "dpa_penalty": dpa_penalty,
            "total_penalty": parser_penalty + gate_penalty + dpa_penalty,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_belief_statuses": self.final_belief_statuses,
            "authorized_belief_ids": list(self.authorized_belief_ids),
            "excluded_belief_ids": list(self.excluded_belief_ids),
            "gate_decisions": self.gate_decisions,
            "defeat_paths": self.defeat_paths,
            "audit_trace": self.audit_trace,
            "parse_result": self.parse_result.to_dict(),
            "engine_errors": [e.to_dict() for e in self.engine_errors],
            "parser_errors": [e.to_dict() for e in self.parser_errors],
            "gate_errors": [e.to_dict() for e in self.gate_errors],
            "dpa_errors": [e.to_dict() for e in self.dpa_errors],
            "warnings": [e.to_dict() for e in self.warnings],
            "admitted_actions": [a.to_dict() for a in self.admitted_actions],
            "rejected_actions": [a.to_dict() for a in self.rejected_actions],
            "final_statuses": self.final_statuses,
            "failure_categories": self.failure_categories,
            "reward_breakdown": self.reward_breakdown,
        }


def parse_actions(raw_text: str) -> ParseResult:
    """Parse a model completion into validated typed revision actions.

    Fail-closed: JSON decode failure or any schema constraint violation yields a
    ``ParseResult`` carrying the canonical ``ParseErrorCode`` and an empty action
    list, so the runtime proposes no revisions.
    """
    try:
        raw_list = extract_json_array(raw_text)
    except StructuredParseError as exc:
        return ParseResult(
            valid_json=False,
            schema_valid=False,
            actions=(),
            error_code=exc.code.value,
            error_message=str(exc),
            errors=(
                EngineError(
                    stage=EngineStage.PARSER,
                    code=PARSER_INVALID_JSON,
                    message=str(exc),
                ),
            ),
        )

    actions: list[RevisionAction] = []
    current_idx = 0
    try:
        for current_idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise SchemaValidationError("action items must be JSON objects")
            action = RevisionAction.from_dict(item)
            action.validate()
            actions.append(action)
    except SchemaValidationError as exc:
        msg = str(exc)
        if "must be JSON objects" in msg:
            err = EngineError(
                stage=EngineStage.PARSER,
                code=PARSER_ITEM_NOT_OBJECT,
                message=f"item {current_idx} is not a JSON object",
                action_index=current_idx,
            )
        else:
            err = EngineError(
                stage=EngineStage.PARSER,
                code=PARSER_SCHEMA_VIOLATION,
                message=msg,
                action_index=current_idx,
            )
        return ParseResult(
            valid_json=True,
            schema_valid=False,
            actions=(),
            error_code=ParseErrorCode.SCHEMA_CONSTRAINTS_VIOLATED.value,
            error_message=msg,
            errors=(err,),
        )

    return ParseResult(
        valid_json=True, schema_valid=True, actions=tuple(actions)
    )


def _extract_runtime_fields(trace: dict[str, Any]) -> tuple[dict[str, str], list, list]:
    final_statuses = dict(trace.get("fine_grained_statuses", {}))
    gate_decisions = list(trace.get("edge_proposals", []))
    defeat_paths = list(trace.get("defeat_paths", []))
    return final_statuses, gate_decisions, defeat_paths


def _augment_replacement_statuses(
    final_statuses: dict[str, str],
    actions: list[RevisionAction],
    gate_decisions: list[dict[str, Any]],
    replacement_ids: set[str],
) -> dict[str, str]:
    """Report replacement beliefs entering the authorized basis.

    The kernel only assigns a DPA status to ``candidate_beliefs``; replacement
    beliefs live in their own (disjoint) list so the proposer can cite them.
    When a ``SUPERSEDES`` edge is *admitted by the gate*, its replacement becomes
    the new usable basis, so we surface it as ``AUTHORIZED`` here. This reads only
    admitted edges from the audit trace (fully replayable); it never overrides a
    status the kernel already computed. Running full DPA over replacement beliefs
    is a documented future extension (see design doc, Risks).
    """
    admitted_supersede_targets = {
        d["target_id"]
        for d in gate_decisions
        if d.get("edge_type") == "SUPERSEDES" and d.get("admitted")
    }
    augmented = dict(final_statuses)
    for a in actions:
        if (
            a.action_type == "SUPERSEDES"
            and a.target_belief_id in admitted_supersede_targets
            and a.replacement_belief_id in replacement_ids
            and a.replacement_belief_id not in augmented
        ):
            augmented[a.replacement_belief_id] = "AUTHORIZED"
    return augmented


def run_actions(
    view: SharedCandidateView,
    actions: list[RevisionAction],
    *,
    parse_result: ParseResult | None = None,
    audit_metadata: dict[str, Any] | None = None,
) -> RuntimeResult:
    """Run already-parsed actions through RevisionGate + DPA via ``authorize``."""
    batches = actions_to_proposal_batches(actions)
    auth = authorize(view, batches, audit_metadata=audit_metadata)
    final_statuses, gate_decisions, defeat_paths = _extract_runtime_fields(auth.trace)
    replacement_ids = {b.belief_id for b in view.candidate_replacement_beliefs}
    final_statuses = _augment_replacement_statuses(
        final_statuses, actions, gate_decisions, replacement_ids
    )
    if parse_result is None:
        parse_result = ParseResult(
            valid_json=True, schema_valid=True, actions=tuple(actions)
        )

    # Aggregate errors: start with parser errors, add gate rejections.
    all_errors: list[EngineError] = list(parse_result.errors)
    for idx, gd in enumerate(gate_decisions):
        if not gd.get("admitted", True):
            reason = gd.get("rejection_reason", gd.get("reason", "rejected by gate"))
            all_errors.append(
                EngineError(
                    stage=EngineStage.REVISION_GATE,
                    code=f"GATE_{gd.get('edge_type', 'UNKNOWN')}_REJECTED",
                    message=str(reason),
                    severity=ErrorSeverity.ERROR,
                    action_index=idx,
                    belief_id=gd.get("target_id"),
                )
            )

    return RuntimeResult(
        final_belief_statuses=final_statuses,
        authorized_belief_ids=auth.authorized_belief_ids,
        excluded_belief_ids=auth.excluded_belief_ids,
        gate_decisions=gate_decisions,
        defeat_paths=defeat_paths,
        audit_trace=auth.trace,
        parse_result=parse_result,
        engine_errors=tuple(all_errors),
    )


def run_from_text(
    view: SharedCandidateView,
    raw_text: str,
    *,
    audit_metadata: dict[str, Any] | None = None,
) -> RuntimeResult:
    """Parse a model completion and run it end-to-end through the kernel.

    On parse/schema failure the runtime still calls ``authorize`` with no
    proposals (fail-closed), so every candidate belief receives its default DPA
    status and the failure is recorded in ``parse_result``.
    """
    parse_result = parse_actions(raw_text)
    return run_actions(
        view,
        list(parse_result.actions),
        parse_result=parse_result,
        audit_metadata=audit_metadata,
    )
