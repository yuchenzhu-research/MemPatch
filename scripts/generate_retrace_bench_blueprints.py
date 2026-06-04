#!/usr/bin/env python3
"""Generate deterministic hidden blueprints for the general English benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import DIFFICULTIES, DOMAINS, FAILURE_MODES, PATTERNS
from benchmark.retrace_bench.generation.pattern_spec import resolve_pattern_binding


STATUS_BY_MODE = {
    "stale_memory_reuse": "outdated",
    "under_update": "outdated",
    "over_update": "current",
    "conflict_collapse": "unresolved",
    "scope_leakage": "out_of_scope",
    "policy_violation": "should_not_store",
    "wrong_source_attribution": "current",
    "memory_hallucination": "should_not_store",
    "unnecessary_memory_write": "should_not_store",
    "failure_to_forget": "deleted",
    "failure_to_release_or_restore": "restored",
}

NON_ANSWER_BY_MODE = {
    "policy_violation": "refuse_due_to_policy",
    "conflict_collapse": "mark_unresolved",
    "scope_leakage": "escalate",
    "memory_hallucination": "ask_clarification",
}


def choose_difficulty(index: int) -> str:
    bucket = index % 20
    if bucket < 3:
        return DIFFICULTIES[0]
    if bucket < 9:
        return DIFFICULTIES[1]
    if bucket < 16:
        return DIFFICULTIES[2]
    return DIFFICULTIES[3]


def blueprint(index: int, rng: random.Random) -> dict:
    domain = DOMAINS[index % len(DOMAINS)]
    # Bind every blueprint to a canonical workflow pattern so the rendered
    # scenario passes ``validate_pattern_semantics``: the pattern determines the
    # allowed (failure_mode, expected_decision) pair via ``resolve_pattern_binding``.
    pattern = PATTERNS[index % len(PATTERNS)]
    binding = resolve_pattern_binding(pattern, index)
    primary = binding.failure_mode
    secondary_pool = [m for m in FAILURE_MODES if m != primary]
    secondary_count = index % 4
    secondaries = rng.sample(secondary_pool, secondary_count)
    difficulty = choose_difficulty(index)
    scenario_id = f"rb-en-{index + 1:05d}"
    scope = f"workspace-{chr(65 + index % 4)}"
    other_scope = f"workspace-{chr(65 + (index + 1) % 4)}"
    case_id = f"C-{1000 + index:04d}"
    project_id = f"PROJ-{chr(65 + index % 26)}{17 + index % 83}"
    employee_id = f"EMP-{index % 900 + 100:03d}"
    base_status = STATUS_BY_MODE[primary]
    event_count = 6 + (index % 5)
    if index % 5 != 0:
        event_count = max(event_count, 7)
    memory_count = 2 + (index % 4)
    include_distractor = index % 10 < 5
    include_cross_scope = index % 10 < 4 or difficulty == "L4_cross_scope_adversarial_audit"
    verified_over_trusted = index % 4 == 0 or primary in {"stale_memory_reuse", "under_update", "failure_to_release_or_restore"}
    false_premise = index % 5 == 0 or primary in {"memory_hallucination", "wrong_source_attribution"}
    non_answer = index % 5 == 1 or primary in NON_ANSWER_BY_MODE
    target_memory = f"m-{scenario_id}-target"
    replacement_memory = f"m-{scenario_id}-replacement"
    memory_ids = [target_memory, replacement_memory]
    for n in range(memory_count - 2):
        memory_ids.append(f"m-{scenario_id}-ctx-{n + 1}")
    if include_distractor:
        memory_ids.append(f"m-{scenario_id}-distractor")
    evidence_event = f"e-{scenario_id}-04"
    if verified_over_trusted:
        evidence_event = f"e-{scenario_id}-05"
    expected_decision = binding.expected_decision
    non_answer = non_answer or binding.non_answer_behavior or expected_decision != "use_current_memory"
    return {
        "schema_version": "retrace_bench_general_blueprint_1",
        "scenario_id": scenario_id,
        "pattern": pattern,
        "domain": domain,
        "primary_failure_mode": primary,
        "secondary_failure_modes": secondaries,
        "difficulty": difficulty,
        "scope": scope,
        "other_scope": other_scope,
        "case_id": case_id,
        "project_id": project_id,
        "employee_id": employee_id,
        "target_memory_id": target_memory,
        "replacement_memory_id": replacement_memory,
        "memory_ids": memory_ids,
        "expected_status": base_status,
        "expected_decision": expected_decision,
        "event_count": event_count,
        "memory_count": len(memory_ids),
        "evidence_event_id": evidence_event,
        "include_distractor": include_distractor,
        "include_cross_scope_trap": include_cross_scope,
        "verified_contradicts_trusted_note": verified_over_trusted,
        "requires_rejecting_false_premise": false_premise,
        "requires_non_answer_action": non_answer,
        "seed_marker": rng.randint(100000, 999999),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(json.dumps(blueprint(i, rng), ensure_ascii=False) + "\n")
    print(f"Wrote {args.count} blueprints to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

