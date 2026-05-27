from __future__ import annotations

from retracemem.schemas import RelationPrediction, RelationType


class RevisionGate:
    """Conservative gate for accepted belief revision paths.

    Paper 1 only accepts direct supersession, explicit blocker paths, and recovery paths.
    """

    ACCEPTED_DIRECT = {RelationType.SUPPORT, RelationType.CONDITION, RelationType.REQUIRED_BY}

    def accept_local_relation(self, prediction: RelationPrediction) -> bool:
        if prediction.relation in self.ACCEPTED_DIRECT:
            return True
        if prediction.relation == RelationType.SUPERSEDE:
            return bool(prediction.target_belief_id)
        if prediction.relation == RelationType.BLOCK:
            return bool(prediction.condition or prediction.target_belief_id)
        if prediction.relation == RelationType.UNCERTAIN:
            return True
        return False
