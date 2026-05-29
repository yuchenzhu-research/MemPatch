from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from retracemem.methods.contracts import SharedCandidateView
from retracemem.methods.authorization_executor import (
    ProposedEvidenceEdges,
    execute_authorization,
)


@dataclass(frozen=True)
class AuthorizationRequest:
    """Pluggable request wrapper containing the controlled semantic input

    and optional integration metadata.
    """

    view: SharedCandidateView
    provenance: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AuthorizationResult:
    """Unified result containing authorization verdicts and traces."""

    fine_grained_statuses: dict[str, str]
    authorized_belief_ids: tuple[str, ...]
    excluded_belief_ids: tuple[str, ...]
    trace: dict[str, Any]
    provenance: Optional[dict[str, Any]] = None


class AuthorizationFacade:
    """Facade exposing the ReTrace deterministic DPA authorization core."""

    @staticmethod
    def authorize(
        request: AuthorizationRequest,
        proposal_batches: tuple[ProposedEvidenceEdges, ...],
    ) -> AuthorizationResult:
        """Runs the ReTrace authorization flow given proposed local edges."""
        exec_res = execute_authorization(
            view=request.view,
            proposal_batches=proposal_batches,
            base_provenance=request.provenance,
        )
        return AuthorizationResult(
            fine_grained_statuses=exec_res.provenance.get("fine_grained_statuses", {}),
            authorized_belief_ids=exec_res.authorized_belief_ids,
            excluded_belief_ids=exec_res.excluded_belief_ids,
            trace=exec_res.provenance,
            provenance=request.provenance,
        )

    @staticmethod
    def replay_authorize(
        request: AuthorizationRequest,
        proposals: tuple[ProposedEvidenceEdges, ...],
    ) -> AuthorizationResult:
        """Replays authorization offline using recorded edge proposals."""
        return AuthorizationFacade.authorize(request, proposals)
