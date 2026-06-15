"""Reference Transition Semantics: Rule definitions."""
from __future__ import annotations

import re
from typing import Any
from benchmark.generation.decision_resolver import TRIGGER_PHRASES

# Precedence levels (lower value means higher priority for resolution)
PRIORITY_REFUSE = 1
PRIORITY_ESCALATE = 2
PRIORITY_ASK = 3
PRIORITY_MARK = 4
PRIORITY_USE = 5

class TransitionRule:
    def __init__(self, rule_id: str, trigger_phrases: list[str], target_status: str, priority: int, reason_code: str) -> None:
        self.rule_id = rule_id
        self.trigger_phrases = trigger_phrases
        self.target_status = target_status
        self.priority = priority
        self.reason_code = reason_code

    def matches(self, event_text: str) -> bool:
        # Match using TRIGGER_PHRASES from decision_resolver.py
        pattern = TRIGGER_PHRASES.get(self.rule_id)
        if pattern:
            return bool(pattern.search(event_text))
        # Fallback to simple regex
        for phrase in self.trigger_phrases:
            if re.search(re.escape(phrase), event_text, re.IGNORECASE):
                return True
        return False

# Statically defined rules mapping to the official triggers
RULES: list[TransitionRule] = [
    # Refuse due to policy
    TransitionRule(
        rule_id="credential_secret_forbidden",
        trigger_phrases=["secret", "forbidden"],
        target_status="should_not_store",
        priority=PRIORITY_REFUSE,
        reason_code="refuse_credential",
    ),
    TransitionRule(
        rule_id="security_policy_forbids_storage",
        trigger_phrases=["security policy forbids storage"],
        target_status="should_not_store",
        priority=PRIORITY_REFUSE,
        reason_code="refuse_security_policy",
    ),
    TransitionRule(
        rule_id="compliance_do_not_store",
        trigger_phrases=["compliance", "do not store"],
        target_status="should_not_store",
        priority=PRIORITY_REFUSE,
        reason_code="refuse_compliance",
    ),

    # Escalate
    TransitionRule(
        rule_id="explicit_hold_for_user",
        trigger_phrases=["ask user to confirm before updating"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_user_hold",
    ),
    TransitionRule(
        rule_id="policy_human_review_required",
        trigger_phrases=["human review required before"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_human_review",
    ),
    TransitionRule(
        rule_id="verified_reviewer_hold",
        trigger_phrases=["verified reviewer hold"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_reviewer_hold",
    ),
    TransitionRule(
        rule_id="security_compliance_block",
        trigger_phrases=["compliance block"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_compliance_block",
    ),
    TransitionRule(
        rule_id="protected_production_memory",
        trigger_phrases=["protected production memory"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_protected_prod",
    ),
    TransitionRule(
        rule_id="evidence_sufficient_but_policy_blocks",
        trigger_phrases=["policy blocks automatic"],
        target_status="blocked",
        priority=PRIORITY_ESCALATE,
        reason_code="escalate_policy_blocks",
    ),

    # Ask Clarification
    TransitionRule(
        rule_id="missing_target_scope",
        trigger_phrases=["without specifying", "no target memory specified"],
        target_status="blocked",
        priority=PRIORITY_ASK,
        reason_code="ask_missing_scope",
    ),
    TransitionRule(
        rule_id="ambiguous_user_intent",
        trigger_phrases=["could mean"],
        target_status="blocked",
        priority=PRIORITY_ASK,
        reason_code="ask_ambiguous_intent",
    ),
    TransitionRule(
        rule_id="ambiguous_workspace",
        trigger_phrases=["stable and beta both"],
        target_status="blocked",
        priority=PRIORITY_ASK,
        reason_code="ask_ambiguous_workspace",
    ),
    TransitionRule(
        rule_id="multiple_candidate_memories",
        trigger_phrases=["multiple candidate memories"],
        target_status="blocked",
        priority=PRIORITY_ASK,
        reason_code="ask_multiple_candidates",
    ),

    # Mark Unresolved
    TransitionRule(
        rule_id="mark_verified_conflict",
        trigger_phrases=["verified sources directly conflict"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_conflict",
    ),
    TransitionRule(
        rule_id="mark_insufficient_passive",
        trigger_phrases=["passive monitor gap"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_monitor_gap",
    ),
    TransitionRule(
        rule_id="mark_stalemate_no_authority",
        trigger_phrases=["no authority path"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_no_authority",
    ),
    TransitionRule(
        rule_id="ci_second_verified_contradiction",
        trigger_phrases=["second verified contradiction"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_ci_contradiction",
    ),
    TransitionRule(
        rule_id="ci_passive_monitor_gap",
        trigger_phrases=["ci passive monitor gap"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_ci_monitor_gap",
    ),
    TransitionRule(
        rule_id="ci_no_authority_path",
        trigger_phrases=["ci no authority path"],
        target_status="unresolved",
        priority=PRIORITY_MARK,
        reason_code="mark_ci_no_authority",
    ),

    # Use Current
    TransitionRule(
        rule_id="verified_maintainer_confirms",
        trigger_phrases=["verified maintainer confirms"],
        target_status="current",
        priority=PRIORITY_USE,
        reason_code="use_maintainer_confirm",
    ),
    TransitionRule(
        rule_id="verified_ci_release_confirms",
        trigger_phrases=["verified confirms", "verified ci confirms", "verified release confirms"],
        target_status="current",
        priority=PRIORITY_USE,
        reason_code="use_release_confirm",
    ),
    TransitionRule(
        rule_id="verified_auditor_confirms",
        trigger_phrases=["verified auditor confirms"],
        target_status="current",
        priority=PRIORITY_USE,
        reason_code="use_auditor_confirm",
    ),
    TransitionRule(
        rule_id="stable_scope_matches_target",
        trigger_phrases=["stable scope matches target"],
        target_status="current",
        priority=PRIORITY_USE,
        reason_code="use_scope_matches",
    ),
]
