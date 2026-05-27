from __future__ import annotations

from typing import Any


def calculate_unsupported_revision_rate(records: list[dict[str, Any]]) -> float:
    """UnsupportedRevisionRate (URR)

    Measures the ratio of suppressed beliefs without a valid audit path
    over all suppressed beliefs.
    """
    suppressed_without_audit = 0
    total_suppressed = 0

    for r in records:
        # Check if the record contains decision details in metadata
        meta = r.get("metadata", {})
        decisions = meta.get("decisions", [])
        for dec in decisions:
            if not dec.get("authorized", True):
                total_suppressed += 1
                # If reason is blocked/superseded but justification path is empty or has none
                just_path = dec.get("justification_path", [])
                if not just_path or all(not str(item).strip() for item in just_path):
                    suppressed_without_audit += 1

    if total_suppressed == 0:
        return 0.0
    return suppressed_without_audit / total_suppressed


def calculate_protected_belief_preservation(records: list[dict[str, Any]]) -> float:
    """ProtectedBeliefPreservation (PBP)

    Measures the ratio of protected beliefs left authorized over all protected beliefs.
    """
    protected_authorized = 0
    total_protected = 0

    for r in records:
        meta = r.get("metadata", {})
        decisions = meta.get("decisions", [])
        for dec in decisions:
            # Check if this belief is marked as 'protected' in metadata
            is_protected = dec.get("is_protected", False) or "protected" in str(dec.get("belief_id", ""))
            if is_protected:
                total_protected += 1
                if dec.get("authorized", True):
                    protected_authorized += 1

    if total_protected == 0:
        return 1.0  # Default to 100% preservation if no protected beliefs are marked
    return protected_authorized / total_protected
