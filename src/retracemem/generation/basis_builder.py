from __future__ import annotations

from typing import Any
from retracemem.schemas import BeliefNode, DefeatPath
from retracemem.retrieval.typed_retrievers import QueryBeliefRetriever
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm


def defeat_path_to_dict(path: DefeatPath | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path_id": path.path_id,
        "path_type": path.path_type.value if hasattr(path.path_type, "value") else str(path.path_type),
        "target_belief_id": path.target_belief_id,
        "supporting_dependency_edge_ids": list(path.supporting_dependency_edge_ids),
        "supporting_evidence_edge_ids": list(path.supporting_evidence_edge_ids),
        "replacement_belief_id": path.replacement_belief_id,
        "as_of_time": path.as_of_time,
        "as_of_evidence_id": path.as_of_evidence_id,
        "metadata": dict(path.metadata),
    }


class BasisBuilder:
    """Build a query-time authorized current basis using canonical typed components."""

    def __init__(
        self,
        retriever: QueryBeliefRetriever,
        engine: DefeatPathAuthorizationAlgorithm,
    ) -> None:
        self.retriever = retriever
        self.engine = engine

    def build(
        self,
        query: str,
        beliefs: tuple[BeliefNode, ...],
        limit: int = 10,
        query_id: str | None = None,
    ) -> dict[str, Any]:
        if limit <= 0:
            return {
                "query_id": query_id or "",
                "retrieved_belief_ids": [],
                "authorized_basis": [],
                "excluded": [],
            }

        candidates = self.retriever.retrieve_for_query(query, beliefs, limit=limit)
        retrieved_belief_ids = [c.belief_id for c in candidates]

        authorized_basis: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        added_belief_ids: set[str] = set()

        for candidate in candidates:
            trace = self.engine.authorize(candidate.belief_id, query_id=query_id)
            status_str = trace.status.value if hasattr(trace.status, "value") else str(trace.status)

            if status_str == "AUTHORIZED":
                if candidate.belief_id not in added_belief_ids:
                    if len(authorized_basis) < limit:
                        authorized_basis.append({
                            "belief_id": candidate.belief_id,
                            "proposition": candidate.proposition,
                            "source_evidence_ids": list(candidate.source_evidence_ids),
                            "authorization_status": "AUTHORIZED",
                        })
                        added_belief_ids.add(candidate.belief_id)
            elif status_str == "SUPERSEDED":
                excluded.append({
                    "belief_id": candidate.belief_id,
                    "status": "SUPERSEDED",
                    "accepted_defeat_path": defeat_path_to_dict(trace.accepted_defeat_path),
                })
                # check replacement
                if trace.accepted_defeat_path and trace.accepted_defeat_path.replacement_belief_id:
                    rep_id = trace.accepted_defeat_path.replacement_belief_id
                    if self.engine.store.has_belief(rep_id):
                        rep_belief = self.engine.store.get_belief(rep_id)
                        rep_trace = self.engine.authorize(rep_id, query_id=query_id)
                        rep_status = rep_trace.status.value if hasattr(rep_trace.status, "value") else str(rep_trace.status)
                        if rep_status == "AUTHORIZED":
                            if rep_belief.belief_id not in added_belief_ids:
                                if len(authorized_basis) < limit:
                                    authorized_basis.append({
                                        "belief_id": rep_belief.belief_id,
                                        "proposition": rep_belief.proposition,
                                        "source_evidence_ids": list(rep_belief.source_evidence_ids),
                                        "authorization_status": "AUTHORIZED",
                                    })
                                    added_belief_ids.add(rep_belief.belief_id)
            else:  # BLOCKED or UNRESOLVED
                excluded.append({
                    "belief_id": candidate.belief_id,
                    "status": status_str,
                    "accepted_defeat_path": defeat_path_to_dict(trace.accepted_defeat_path),
                })

        return {
            "query_id": query_id or "",
            "retrieved_belief_ids": retrieved_belief_ids,
            "authorized_basis": authorized_basis,
            "excluded": excluded,
        }
