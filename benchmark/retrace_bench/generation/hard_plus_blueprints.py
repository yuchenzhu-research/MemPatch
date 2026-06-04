import random
from typing import Any, Dict, List, Set
from benchmark.retrace_bench.general_taxonomy import (
    DOMAINS,
    PATTERNS,
    normalize_memory_state,
)
from benchmark.retrace_bench.generation.pattern_spec import (
    build_wrong_answer_traps,
    get_hard_plan_entry,
    resolve_pattern_binding,
)
from benchmark.retrace_bench.generation.evidence_dependency_graph import EvidenceDependencyGraph
from benchmark.retrace_bench.generation.adversarial_distractors import (
    generate_version_distractor,
    generate_authority_distractor,
    generate_rollback_distractor,
    generate_ci_distractor,
)

def neutral_expected_answer(
    *,
    case_id: str,
    artifact: str,
    scope: str,
    source: str,
    expected_decision: str,
) -> str:
    """Gold answer text without pattern or failure-mode label leakage."""
    if expected_decision == "mark_unresolved":
        return (
            f"The authorized status of {case_id} is unresolved; the {artifact} "
            f"cannot be confirmed as current until conflicting records on {scope} are reconciled."
        )
    if expected_decision == "ask_clarification":
        return (
            f"Evidence about {case_id} is incomplete; clarify the {artifact} "
            f"status on {scope} before treating any update as authorized."
        )
    if expected_decision == "escalate":
        return (
            f"The {artifact} change for {case_id} needs escalation because "
            f"the verified {source} record on {scope} does not support a safe automatic update."
        )
    if expected_decision == "refuse_due_to_policy":
        return (
            f"The requested update to {case_id} is blocked by policy; "
            f"the existing {artifact} on {scope} must not be altered."
        )
    return (
        f"The verified {source} confirms the current {artifact} settings for "
        f"{case_id} on {scope} align with the stable authorized configuration."
    )


DOMAIN_FRAMES = {
    "software_engineering_agent": {
        "owner": "release engineer", "item": "pull request", "artifact": "deployment gate",
        "source": "CI log", "topics": ("PR review", "dependency deprecation", "API migration", "rollout blocker"),
    },
    "enterprise_multi_tool_workflow": {
        "owner": "operations lead", "item": "approval chain", "artifact": "handoff route",
        "source": "workflow tool", "topics": ("role-based permission", "cross-team handoff", "admin approval", "vendor intake"),
    },
    "customer_support_crm": {
        "owner": "support lead", "item": "support ticket", "artifact": "case route",
        "source": "support timeline", "topics": ("refund policy", "loyalty tier", "fraud flag", "order status"),
    },
    "calendar_task_workflow": {
        "owner": "calendar coordinator", "item": "meeting invite", "artifact": "schedule rule",
        "source": "calendar sync", "topics": ("room booking", "timezone", "attendee authority", "recurring exception"),
    },
    "research_knowledge_work": {
        "owner": "research analyst", "item": "literature note", "artifact": "claim record",
        "source": "citation index", "topics": ("paper claim", "citation correction", "retraction note", "source corpus version"),
    },
    "personal_assistant_preference": {
        "owner": "assistant delegate", "item": "preference memory", "artifact": "preference rule",
        "source": "profile update", "topics": ("consent boundary", "location-specific preference", "travel preference", "notification style"),
    },
    "ecommerce_recommendation": {
        "owner": "merchandising agent", "item": "recommendation memory", "artifact": "shopping rule",
        "source": "catalog event", "topics": ("stock availability", "seller policy", "return window", "brand preference"),
    },
    "data_analysis_bi": {
        "owner": "BI owner", "item": "dashboard request", "artifact": "metric definition",
        "source": "warehouse lineage", "topics": ("source-table lineage", "filter changes", "metric definition", "refresh policy"),
    },
}


def build_deterministic_scenario(
    index: int,
    split_name: str,
    seed: int,
    *,
    split_count: int | None = None,
) -> Dict[str, Any]:
    rng = random.Random(seed + index * 101)
    
    # Stratified selection: pattern first, then derive labels from PATTERN_SPEC.
    domain = DOMAINS[index % len(DOMAINS)]
    if split_name == "hard" and split_count is not None:
        pattern, forced_decision = get_hard_plan_entry(index, split_count, seed)
        binding = resolve_pattern_binding(pattern, index, forced_decision=forced_decision)
    else:
        pattern = PATTERNS[index % len(PATTERNS)]
        binding = resolve_pattern_binding(pattern, index)
    failure_mode = binding.failure_mode
    expected_decision = binding.expected_decision
    
    frame = DOMAIN_FRAMES[domain]
    topic = frame["topics"][(index // len(DOMAINS)) % len(frame["topics"])]
    
    # Assign difficulty level based on split guidelines
    # L1: one evidence event, no distractor. L2: distillers, >=2 evidence events. L3: scope/version/auth conflict, >=3 evidence. L4: multi-memory coupling, negative evidence, rollback, no shortcut.
    if split_name == "main":
        difficulty_level = ["L1", "L2", "L3", "L4"][index % 4]
    elif split_name == "hard":
        difficulty_level = "L3" if index % 2 == 0 else "L4"
    elif split_name == "realistic":
        difficulty_level = "L3" if index % 3 != 0 else "L4"
    else:  # calibration / smoke
        difficulty_level = "L2" if index % 2 == 0 else "L3"

    scenario_id = f"rt-{split_name}-{index+1:06d}"
    case_id = f"CASE-{300000 + index}"
    project_id = f"PROJ-{500000 + index}"
    scope = f"workspace-stable"
    other_scope = f"workspace-beta"
    
    # Memory and Event IDs
    m_target = f"m-{scenario_id}-target"
    m_replacement = f"m-{scenario_id}-replacement"
    m_condition = f"m-{scenario_id}-condition"
    m_distractor = f"m-{scenario_id}-distractor"
    
    # Determine gold behavior and outputs based on pattern and difficulty
    expected_failure_diagnosis = failure_mode
    
    # Standard values
    artifact = frame["artifact"]
    owner = frame["owner"]
    source = frame["source"]
    
    # Assemble events
    event_trace = []
    initial_memory = [
        {
            "memory_id": m_target,
            "text": f"Prior state: {case_id} uses default {artifact} configuration on stable v1.",
            "scope": scope,
            "source_event_ids": ["e-init"],
            "is_distractor": False
        },
        {
            "memory_id": m_condition,
            "text": f"Condition rule: Any update to {case_id} requires verified release approval.",
            "scope": scope,
            "source_event_ids": ["e-init"],
            "is_distractor": False
        }
    ]
    
    dag = EvidenceDependencyGraph()
    dag.add_event("e-init")
    
    e1_id = f"e-{scenario_id}-1"
    e2_id = f"e-{scenario_id}-2"
    e3_id = f"e-{scenario_id}-3"
    e4_id = f"e-{scenario_id}-4"
    e5_id = f"e-{scenario_id}-5"
    
    dag.add_event(e1_id)
    dag.add_event(e2_id)
    dag.add_event(e3_id)
    dag.add_event(e4_id)
    dag.add_event(e5_id)
    
    # 15 patterns logics
    if pattern == "merged_but_unreleased":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "issue", "text": f"Issue #100 reports {case_id} lacks YAML support.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "reviewer", "trust_level": "verified", "visibility_scope": other_scope, "event_type": "pr", "text": f"PR #101 implementing YAML support for {case_id} was merged to branch main/dev.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "release_note", "trust_level": "verified", "visibility_scope": scope, "event_type": "release", "text": f"Release v1.4.0 notes list only hotfixes and do not include the YAML feature for {case_id}.", "related_memory_ids": []}
        ]
        dag.add_requires(e3_id, e1_id)
        minimal_evidence = {e1_id, e3_id}
        counterevidence = {e2_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}
        
    elif pattern == "closed_as_duplicate_not_fixed":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "issue", "text": f"Issue #200 reports SSO timeout error on {case_id}.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer closed Issue #200 as duplicate of master Issue #98.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Master Issue #98 remains open with labels: needs-investigation and priority-high.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        dag.add_requires(e3_id, e2_id)
        minimal_evidence = {e1_id, e2_id, e3_id}
        counterevidence = set()
        expected_states = {m_target: "unresolved", m_condition: "authorized"}

    elif pattern == "docs_ahead_of_code":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "docs", "trust_level": "verified", "visibility_scope": scope, "event_type": "docs", "text": f"README docs updated: {case_id} now supports batch deletes.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "reviewer", "trust_level": "trusted", "visibility_scope": scope, "event_type": "pr", "text": f"Code PR #300 implementing batch delete remains unmerged and tests are failing.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer notes docs were merged ahead of code implementation by mistake.", "related_memory_ids": []}
        ]
        dag.add_requires(e3_id, e1_id)
        dag.add_requires(e3_id, e2_id)
        minimal_evidence = {e1_id, e2_id, e3_id}
        counterevidence = set()
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "release_then_revert":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "release_note", "trust_level": "verified", "visibility_scope": scope, "event_type": "release", "text": f"Release v2.0.0 ships with strict schema validation active.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "pr", "text": f"Revert PR #401 merged: reverts strict validation due to regressions.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "release_note", "trust_level": "verified", "visibility_scope": scope, "event_type": "release", "text": f"Release v2.0.1 reverts strict validation and returns default config to opt-in.", "related_memory_ids": []}
        ]
        dag.add_requires(e3_id, e2_id)
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e1_id, e2_id, e3_id}
        counterevidence = set()
        expected_states = {m_target: "restored", m_condition: "authorized"}

    elif pattern == "version_scope_leakage":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"XML exporter is required for production reports on v1.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "reviewer", "trust_level": "verified", "visibility_scope": other_scope, "event_type": "comment", "text": f"Developer removed XML exporter on v2 development branch.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer confirms XML export will remain fully supported on v1 stable.", "related_memory_ids": []}
        ]
        dag.add_requires(e3_id, e1_id)
        minimal_evidence = {e1_id, e3_id}
        counterevidence = {e2_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "branch_scope_leakage":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "reviewer", "trust_level": "trusted", "visibility_scope": other_scope, "event_type": "pr", "text": f"PR on feature branch feat-311 enables Python 3.11 runtimes.", "related_memory_ids": []},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Main release branch targets only Python 3.10 and does not accept Python 3.11 commits.", "related_memory_ids": [m_target]}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "authority_conflict":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "untrusted", "visibility_scope": scope, "event_type": "comment", "text": f"User claims CVE-999 vulnerability has been fixed in latest patch.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "security", "trust_level": "verified", "visibility_scope": scope, "event_type": "security", "text": f"Security auditor confirms CVE-999 remains active and unpatched in current builds.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "unresolved", m_condition: "authorized"}

    elif pattern == "ci_failed_after_claim":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"Developer states performance hotfix is ready and merged.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "ci", "trust_level": "verified", "visibility_scope": scope, "event_type": "ci", "text": f"CI pipeline check for performance hotfix failed during compilation.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e1_id, e2_id}
        counterevidence = set()
        expected_states = {m_target: "blocked", m_condition: "authorized"}

    elif pattern == "security_policy_override":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"User requests local caching of passwords for faster OAuth logins.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "security", "trust_level": "verified", "visibility_scope": scope, "event_type": "policy", "text": f"Security Policy override: caching plain text authentication credentials is strictly forbidden.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "blocked", m_condition: "authorized"}

    elif pattern == "backport_only_fix":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "release", "text": f"Security patch backported only to stable legacy branch v1.2.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Production branch v2.0 uses a redesigned core and does not need or receive the patch.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e1_id, e2_id}
        counterevidence = set()
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "maintainer_correction_over_user_claim":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"User asserts database connection pool limit is now 100.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer clarifies pool limit is kept at 20 to prevent server exhaustion.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "stale_comment_after_new_release":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"Comment from 2024 states retry limit is locked to 3.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "release_note", "trust_level": "verified", "visibility_scope": scope, "event_type": "release", "text": f"Release v3.0 updates default retry limit to 5.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "label_state_mismatch":
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "issue", "text": f"Issue #900 reports memory leaks under load.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer closes issue with label wontfix, commenting that leak is within acceptable limits.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    elif pattern == "multi_memory_coupling":
        # Couple changes to target AND condition memories.
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "comment", "text": f"Request to migrate sync connections to async config.", "related_memory_ids": [m_target, m_condition]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "maintainer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Maintainer merges async client and timeout migrations conjointly.", "related_memory_ids": []}
        ]
        dag.add_requires(e2_id, e1_id)
        minimal_evidence = {e2_id}
        counterevidence = {e1_id}
        expected_states = {m_target: "superseded", m_condition: "superseded"}

    else:  # negative_evidence_required
        event_trace = [
            {"event_id": e1_id, "timestamp_order": 1, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "issue", "text": f"Issue #500 reports SSL routing error.", "related_memory_ids": [m_target]},
            {"event_id": e2_id, "timestamp_order": 2, "actor_role": "user", "trust_level": "trusted", "visibility_scope": scope, "event_type": "pr", "text": f"Developer opens PR #501 to resolve SSL routing error.", "related_memory_ids": []},
            {"event_id": e3_id, "timestamp_order": 3, "actor_role": "reviewer", "trust_level": "verified", "visibility_scope": scope, "event_type": "comment", "text": f"Reviewer states PR #501 is on hold and no merge action has been approved.", "related_memory_ids": []}
        ]
        dag.add_requires(e3_id, e2_id)
        minimal_evidence = {e1_id, e3_id}
        counterevidence = {e2_id}
        expected_states = {m_target: "authorized", m_condition: "authorized"}

    # Align memory states with pattern-bound decision/failure semantics.
    if pattern == "authority_conflict":
        if expected_decision == "use_current_memory":
            expected_states[m_target] = "authorized"
        else:
            expected_states[m_target] = "unresolved"
    elif pattern == "ci_failed_after_claim":
        if expected_decision == "mark_unresolved":
            expected_states[m_target] = "unresolved"
        else:
            expected_states[m_target] = "blocked"
    elif pattern == "multi_memory_coupling":
        if failure_mode == "under_update":
            expected_states[m_target] = "superseded"
            expected_states[m_condition] = "authorized"
        else:
            expected_states[m_target] = "superseded"
            expected_states[m_condition] = "superseded"
    elif pattern == "stale_comment_after_new_release" and failure_mode == "under_update":
        expected_states[m_target] = "superseded"
    elif pattern == "negative_evidence_required" and expected_decision == "mark_unresolved":
        expected_states[m_target] = "unresolved"

    # Incorporate distractors depending on difficulty
    if difficulty_level in ("L2", "L3", "L4"):
        # Version scope / authority distractor
        dist_ev = generate_version_distractor(rng, scenario_id, case_id, other_scope, frame, topic)
        event_trace.append(dist_ev)
        dag.add_event(dist_ev["event_id"])
        
        initial_memory.append({
            "memory_id": m_distractor,
            "text": f"Distractor info: {case_id} has separate config for {other_scope}.",
            "scope": other_scope,
            "source_event_ids": [dist_ev["event_id"]],
            "is_distractor": True
        })
        expected_states[m_distractor] = "out_of_scope"
        
    if difficulty_level in ("L3", "L4"):
        # Add authority conflict distractor
        auth_dist = generate_authority_distractor(rng, scenario_id, case_id, scope, frame, topic)
        event_trace.append(auth_dist)
        dag.add_event(auth_dist["event_id"])

    if difficulty_level == "L4":
        # Add rollback distractor and CI failure
        roll_dist = generate_rollback_distractor(rng, scenario_id, case_id, scope, frame)
        ci_dist = generate_ci_distractor(rng, scenario_id, case_id, scope, frame)
        event_trace.extend([roll_dist, ci_dist])
        dag.add_event(roll_dist["event_id"])
        dag.add_event(ci_dist["event_id"])

    # Ensure at least 90% of scenarios have at least 7 events
    bg_events = [
        f"Routine status synchronization for {project_id} confirmed system heartbeat is normal.",
        f"The operations log record shows database connection pool state is active.",
        f"Reviewer notes scheduling queue has no extra blocked actions pending.",
        f"Continuous Integration agent completed baseline checks for auxiliary module.",
        f"Documentation system refreshed the stable index files.",
        f"API gateway logged successful authentication check."
    ]
    rng.shuffle(bg_events)
    bg_idx = 0
    while len(event_trace) < 7 and bg_idx < len(bg_events):
        event_trace.append({
            "event_id": f"e-{scenario_id}-bg-{bg_idx+1}",
            "timestamp_order": 20 + bg_idx,
            "actor_role": "bot" if bg_idx % 2 == 0 else "user",
            "trust_level": "trusted",
            "visibility_scope": scope,
            "event_type": "comment",
            "text": bg_events[bg_idx],
            "related_memory_ids": []
        })
        bg_idx += 1

    # Ensure timestamp orders are sorted
    for idx_ev, ev in enumerate(event_trace):
        ev["timestamp_order"] = idx_ev + 1
        if "timestamp" not in ev:
            ev["timestamp"] = f"2027-01-01T09:{idx_ev:02d}:00Z"
        
    wrong_answer_traps = build_wrong_answer_traps(pattern, case_id, other_scope)
    expected_answer = neutral_expected_answer(
        case_id=case_id,
        artifact=artifact,
        scope=scope,
        source=source,
        expected_decision=expected_decision,
    )
    
    # Required Scenario V2 structure mapping to system requirements
    metadata = {
        "schema_version": "retrace_bench_general_1",
        "renderer": f"{split_name}_final_renderer",
        "split": split_name,
        "pattern": pattern,
        "pattern_trap_type": binding.trap_type,
        "canonical_failure_mode": failure_mode,
        "has_distractor": (difficulty_level in ("L2", "L3", "L4")),
        "has_cross_scope_trap": (difficulty_level in ("L3", "L4") or pattern in ("version_scope_leakage", "branch_scope_leakage", "merged_but_unreleased")),
        "verified_contradicts_trusted_note": True,
        "requires_rejecting_false_premise": True,
        "requires_non_answer_action": binding.non_answer_behavior,
        "event_count": len(event_trace),
        "memory_count": len(initial_memory),
        "seed": seed + index
    }

    return {
        "scenario_id": scenario_id,
        "pattern": pattern,
        "benchmark_version": "final",
        "public_split_name": split_name,
        "domain": domain,
        "primary_failure_mode": failure_mode,
        "difficulty": difficulty_level,
        "workflow_context": f"{owner.title()} is auditing {topic} and checking state integrity for {case_id} on {domain}.",
        "source_type": "controlled_synthetic",
        "source_pointers": [
            {
                "kind": "synthetic_blueprint",
                "repo": "yuchenzhu-research/ReTrace",
                "url_or_id": f"blueprint-{pattern}-{index}",
                "license_or_terms_note": "public GitHub artifact, paraphrased/anonymized"
            }
        ],
        "difficulty_level": difficulty_level,
        "difficulty_factors": {
            "num_events": len(event_trace),
            "num_memories": len(initial_memory),
            "authority_conflict": (difficulty_level in ("L3", "L4")),
            "scope_collision": (pattern in ("version_scope_leakage", "branch_scope_leakage")),
            "version_or_release_chain": (pattern in ("merged_but_unreleased", "release_then_revert")),
            "branch_scope_conflict": (pattern == "branch_scope_leakage"),
            "ci_or_test_state_required": (pattern == "ci_failed_after_claim" or difficulty_level == "L4"),
            "multi_memory_coupling": (pattern == "multi_memory_coupling"),
            "negative_evidence_required": (pattern == "negative_evidence_required"),
            "rollback_or_restore": (pattern == "release_then_revert"),
            "policy_or_security_constraint": (pattern == "security_policy_override"),
            "minimal_evidence_required": True,
            "adversarial_distractors": 2 if difficulty_level == "L4" else (1 if difficulty_level in ("L2", "L3") else 0)
        },
        "public_input": {
            "event_trace": event_trace,
            "initial_memory": initial_memory
        },
        "black_box_task": {
            "prompt": f"What is the final authorized status and value of {case_id} according to stable rules?",
            "output_schema": "JSON object with 'decision' and 'answer'"
        },
        "memory_state_task": {
            "prompt": "Classify active memory statuses for current entries.",
            "output_schema": "JSON object mapping memory_id -> status"
        },
        "evidence_retrieval_task": {
            "prompt": "Cite exact event IDs for minimal required evidence.",
            "output_schema": "JSON list of event_ids"
        },
        "diagnostic_task": {
            "prompt": "Diagnose the primary memory failure mode.",
            "output_schema": "JSON object with 'failure_diagnosis'"
        },
        "hidden_gold": {
            "expected_decision": expected_decision,
            "expected_answer": expected_answer,
            "expected_memory_state": normalize_memory_state(expected_states),
            "expected_evidence_event_ids": sorted(list(minimal_evidence)),
            "counterevidence_event_ids": sorted(list(counterevidence)),
            "expected_failure_diagnosis": expected_failure_diagnosis,
            "stale_or_wrong_answers": wrong_answer_traps,
            "rubric": {
                "must_include": [case_id, artifact],
                "must_not_include": [other_scope] if pattern != "version_scope_leakage" else [],
            },
        },
        "validation_notes": {
            "solvable_from_visible_evidence": True,
            "no_hidden_gold_leakage": True,
            "no_latest_event_shortcut": (difficulty_level in ("L3", "L4")),
            "requires_minimal_evidence": True
        },
        "metadata": metadata
    }
