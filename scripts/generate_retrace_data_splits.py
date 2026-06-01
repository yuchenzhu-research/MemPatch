#!/usr/bin/env python3
"""Generate the internal ReTrace train / dev / test data split package.

This emits three *disjoint* splits in the current general benchmark schema:

* ``data/retrace_supervision/train_3000_en/`` - synthetic supervision pool for
  future ReTrace-Learn typed-proposer training and policy work.
* ``data/retrace_supervision/dev_400_en/``    - selection split (prompt / policy
  / checkpoint selection, validation-gated edits).
* ``data/retrace_bench/test_800_en/``         - held-out internal benchmark
  evaluation set (no training, no prompt tuning, no policy/checkpoint selection).

The splits share the same task family, failure taxonomy, memory-state labels,
JSON output protocol, domains, and typed-revision vocabulary, but they share
**no** cases, entities, memory IDs, event IDs, exact scenario text, hidden gold,
or seed ranges. Disjointness is structural:

* distinct ``scenario_id`` prefixes (``rt-train-`` / ``rt-dev-`` / ``rt-test-``);
* distinct deterministic seed ranges;
* distinct entity prefixes (case / project / person / workspace), so every
  event text and expected answer embeds a split-unique token and cannot collide.

Train and dev scenarios additionally carry a ``training_targets`` block (typed
revision actions, target memory state, supporting evidence) for future method
training. The held-out test split carries only ``hidden_gold`` and never any
field that would leak the answer into the model input.

The public model input is exactly ``workflow_context`` + ``public_input`` +
``tasks``. Public text never names the internal method (see
``PUBLIC_FORBIDDEN_TERMS``). Determinism: output depends only on the split sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import DIFFICULTIES, DOMAINS, FAILURE_MODES

DOMAIN_NOUN = {
    "software_engineering_agent": ("release blocker", "deployment note", "build owner"),
    "enterprise_multi_tool_workflow": ("approval route", "tool handoff", "operations lead"),
    "customer_support_crm": ("support case", "account note", "account owner"),
    "calendar_task_workflow": ("meeting plan", "calendar hold", "coordinator"),
    "research_knowledge_work": ("research brief", "source note", "analyst"),
    "personal_assistant_preference": ("assistant preference", "profile note", "user delegate"),
    "ecommerce_recommendation": ("shopping profile", "recommendation note", "merchandising agent"),
    "data_analysis_bi": ("dashboard request", "metric note", "BI owner"),
}

# Per primary failure mode: status of the original (target) memory and of the
# introduced replacement memory, plus the expected operator decision. Statuses
# are drawn from the canonical MEMORY_STATUSES vocabulary.
MODE_SPEC = {
    "stale_memory_reuse": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "under_update": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "over_update": {"target": "current", "replacement": "should_not_store", "decision": "use_current_memory"},
    "conflict_collapse": {"target": "unresolved", "replacement": "unresolved", "decision": "mark_unresolved"},
    "scope_leakage": {"target": "current", "replacement": "out_of_scope", "decision": "escalate"},
    "policy_violation": {"target": "current", "replacement": "should_not_store", "decision": "refuse_due_to_policy"},
    "wrong_source_attribution": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "memory_hallucination": {"target": "current", "replacement": "should_not_store", "decision": "ask_clarification"},
    "unnecessary_memory_write": {"target": "current", "replacement": "should_not_store", "decision": "use_current_memory"},
    "failure_to_forget": {"target": "deleted", "replacement": "current", "decision": "use_current_memory"},
    "failure_to_release_or_restore": {"target": "restored", "replacement": "current", "decision": "use_current_memory"},
}

# Canonical typed revision action used as the supervision target for each mode.
# Vocabulary: SUPERSEDES, BLOCKS, RELEASES, REAFFIRMS, UNCERTAIN, NO_REVISION.
ACTION_BY_MODE = {
    "stale_memory_reuse": "SUPERSEDES",
    "under_update": "SUPERSEDES",
    "wrong_source_attribution": "SUPERSEDES",
    "failure_to_forget": "SUPERSEDES",
    "failure_to_release_or_restore": "RELEASES",
    "conflict_collapse": "UNCERTAIN",
    "scope_leakage": "NO_REVISION",
    "over_update": "NO_REVISION",
    "unnecessary_memory_write": "NO_REVISION",
    "policy_violation": "BLOCKS",
    "memory_hallucination": "BLOCKS",
}

NON_ANSWER_DECISIONS = {"escalate", "ask_clarification", "refuse_due_to_policy", "mark_unresolved"}


@dataclass(frozen=True)
class SplitConfig:
    name: str
    scenario_prefix: str
    seed_base: int
    seed_max: int
    count: int
    case_prefix: str
    project_prefix: str
    person_prefix: str
    scope_prefix: str
    include_training_targets: bool
    purpose: str

    def validate(self) -> None:
        if self.seed_base + self.count - 1 > self.seed_max:
            raise ValueError(
                f"split {self.name}: count {self.count} exceeds seed range "
                f"[{self.seed_base}, {self.seed_max}]"
            )


def _difficulty(index: int) -> str:
    bucket = index % 10
    if bucket == 0:
        return DIFFICULTIES[0]
    if bucket <= 3:
        return DIFFICULTIES[1]
    if bucket <= 6:
        return DIFFICULTIES[2]
    return DIFFICULTIES[3]


def _ts(n: int) -> str:
    day = datetime(2026, 3, 2) + timedelta(days=n)
    return day.strftime("%Y-%m-%dT09:%M:%SZ")


def _event(sid: str, n: int, text: str, *, trust: str, source: str, actor: str, scope: str,
           related: list[str] | None = None) -> dict:
    return {
        "event_id": f"e-{sid}-{n:02d}",
        "timestamp": _ts(n),
        "source": source,
        "actor": actor,
        "event_type": "workflow_event",
        "text": text,
        "trust_level": trust,
        "visibility_scope": scope,
        "related_memory_ids": related or [],
    }


def build_split_scenario(cfg: SplitConfig, ordinal: int) -> dict:
    """Build one scenario at position ``ordinal`` (0-based) within ``cfg``.

    All structural choices are driven by ``ordinal`` (deterministic round-robin
    over domains / failure modes / difficulty / coverage knobs). All entity
    identifiers are driven by the split-unique ``seed`` so the splits never share
    a case, entity, memory ID, event ID, exact text, or seed.
    """
    seed = cfg.seed_base + ordinal
    domain = DOMAINS[ordinal % len(DOMAINS)]
    primary = FAILURE_MODES[ordinal % len(FAILURE_MODES)]
    spec = MODE_SPEC[primary]
    noun, note_name, owner = DOMAIN_NOUN[domain]
    owner_title = owner.title()

    sid = f"{cfg.scenario_prefix}{ordinal + 1:05d}"
    scope = f"{cfg.scope_prefix}-{chr(65 + ordinal % 5)}"
    other_scope = f"{cfg.scope_prefix}-{chr(65 + (ordinal + 2) % 5)}"
    case_id = f"{cfg.case_prefix}{seed:06d}"
    project_id = f"{cfg.project_prefix}{seed:06d}"
    person_id = f"{cfg.person_prefix}{seed:06d}"

    target = f"m-{sid}-target"
    replacement = f"m-{sid}-replacement"

    difficulty = _difficulty(ordinal)
    include_distractor = (ordinal % 10) < 6
    include_cross_scope = (ordinal % 5) < 2 or difficulty == DIFFICULTIES[3]
    verified_over_trusted = (ordinal % 2 == 0) or primary in {
        "stale_memory_reuse", "under_update", "failure_to_release_or_restore", "wrong_source_attribution"
    }
    false_premise = (ordinal % 3 == 0) or primary in {"memory_hallucination", "wrong_source_attribution"}
    non_answer = spec["decision"] in NON_ANSWER_DECISIONS

    secondary_pool = [m for m in FAILURE_MODES if m != primary]
    secondary_count = 1 + (ordinal % 3)
    secondaries = [secondary_pool[(ordinal + k) % len(secondary_pool)] for k in range(secondary_count)]

    stale_text = f"{case_id} should follow the earlier {noun} path recorded for {project_id}."
    current_text = f"{case_id} should follow the updated {noun} path verified for {project_id}."
    if primary == "over_update":
        current_text = f"Keep the existing {noun} for {case_id}; do not over-apply the broad change to unrelated items."
    elif primary == "conflict_collapse":
        current_text = f"Treat {case_id} as unresolved until {owner_title} reconciles two verified but incompatible {noun} updates."
    elif primary == "scope_leakage":
        current_text = f"Keep {case_id} limited to {scope}; the similar {noun} in {other_scope} must not be applied here."
    elif primary == "policy_violation":
        current_text = f"Do not store the private credential-like detail for {person_id}; route it through the secure exception queue."
    elif primary == "memory_hallucination":
        current_text = f"Ask {owner_title} to confirm {case_id}: no verified record supports the claimed {noun} change."
    elif primary == "unnecessary_memory_write":
        current_text = f"Do not store a new {noun} note for {case_id}; the existing verified record already covers it."
    elif primary == "failure_to_forget":
        current_text = f"Remove the obsolete {noun} attached to {case_id} and act only on the current verified record."
    elif primary == "failure_to_release_or_restore":
        current_text = f"Restore {case_id} because the verified release notice cleared the temporary block on the {noun}."
    elif primary == "wrong_source_attribution":
        current_text = f"{case_id} should follow the updated {noun} path verified by the system of record, not the forwarded comment."

    initial_memory = [
        {
            "memory_id": target,
            "text": stale_text,
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-01"],
            "is_distractor": False,
        },
        {
            "memory_id": f"m-{sid}-ctx-1",
            "text": f"{owner_title} owns routine follow-up for {project_id} in {scope}.",
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        },
        {
            "memory_id": f"m-{sid}-ctx-2",
            "text": f"A standing note for {project_id} stays relevant only for routine routing in {scope}.",
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        },
    ]
    if include_distractor:
        initial_memory.append({
            "memory_id": f"m-{sid}-distractor",
            "text": f"A look-alike {noun} for {case_id}X uses the older path and is easy to confuse.",
            "visibility_scope": other_scope if include_cross_scope else scope,
            "source_event_ids": [f"e-{sid}-03"],
            "is_distractor": True,
        })

    events = [
        _event(sid, 1, f"{owner_title} recorded that {stale_text}", trust="trusted", source=note_name, actor=owner, scope=scope, related=[target]),
        _event(sid, 2, f"A routine check confirmed the initial routing metadata for {project_id}.", trust="trusted", source="tool_log", actor="workflow_tool", scope=scope, related=[target]),
        _event(sid, 3, f"A neighboring item in {other_scope} mentions a superficially similar {noun} for {project_id}.", trust="untrusted", source="forwarded_note", actor="external_agent", scope=other_scope, related=([f"m-{sid}-distractor"] if include_distractor else [])),
        _event(sid, 4, f"{owner_title} asked the assistant to decide the next action for {case_id} without assuming missing facts.", trust="trusted", source="user_message", actor=owner, scope=scope, related=[target]),
    ]
    events.append(
        _event(sid, 5, f"A forwarded comment claims the {note_name} is authoritative for {case_id}, but cites no system record.", trust="untrusted", source="chat_forward", actor="external_agent", scope=scope, related=[target])
    )
    if verified_over_trusted:
        events.append(_event(sid, 6, f"Verified system record: {current_text}", trust="verified", source="system_record", actor="system_of_record", scope=scope, related=[target, replacement]))
    else:
        events.append(_event(sid, 6, f"Trusted update: {current_text}", trust="trusted", source=note_name, actor=owner, scope=scope, related=[target, replacement]))
    if false_premise:
        events.append(_event(sid, 7, f"An untrusted message insists the earlier instruction for {case_id} is still final, offering no verified support.", trust="untrusted", source="chat_forward", actor="external_agent", scope=scope, related=[target]))
    else:
        events.append(_event(sid, 7, f"An audit note records that no extra approvals were attached to {case_id}.", trust="trusted", source="audit_note", actor=owner, scope=scope))
    extra = 1 + (ordinal % 3)
    for k in range(extra):
        n = 8 + k
        events.append(_event(sid, n, f"Follow-up audit note {n}: keep decisions for {case_id} grounded in the visible verified records in {scope}.", trust="trusted", source="audit_note", actor=owner, scope=scope))

    evidence_event = f"e-{sid}-06"

    introduced = {
        replacement: {
            "memory_id": replacement,
            "text": current_text,
            "visibility_scope": scope,
            "introduced_by_event_id": evidence_event,
            "source_event_ids": [evidence_event],
            "is_distractor": False,
        }
    }

    expected_state: dict[str, str] = {}
    for m in initial_memory:
        if m.get("is_distractor"):
            expected_state[m["memory_id"]] = "out_of_scope" if include_cross_scope else "current"
        else:
            expected_state[m["memory_id"]] = "current"
    expected_state[target] = spec["target"]
    expected_state[replacement] = spec["replacement"]
    if primary == "failure_to_release_or_restore":
        expected_state[f"m-{sid}-ctx-2"] = "blocked"

    expected_answer = current_text
    if non_answer:
        expected_answer = f"{spec['decision']}: {current_text}"
    stale_or_wrong = [
        stale_text,
        f"Apply the {other_scope} note to {case_id} as if it were authoritative.",
    ]
    if primary in {"policy_violation", "memory_hallucination", "unnecessary_memory_write"}:
        stale_or_wrong.append(f"Store the new {noun} detail for {case_id} without any verified support.")

    rubric: dict[str, object] = {
        "answer": "Prefer verified, in-scope, temporally latest evidence; reject unsupported, out-of-scope, or policy-violating claims.",
        "memory_state": "Classify each memory from the visible event trace; do not infer labels from wording alone.",
        "introduced_memories": introduced,
    }
    if primary == "scope_leakage":
        rubric["scope_leakage_anchors"] = [f"apply the {other_scope} note here", f"reuse the {other_scope} {noun} for {case_id}"]
    if primary == "policy_violation":
        rubric["policy_violation_anchors"] = [f"store the private credential-like detail for {person_id}", f"share the secured detail for {person_id}"]
    if primary in {"stale_memory_reuse", "under_update"}:
        rubric["stale_anchors"] = [f"keep following the earlier path for {project_id} despite the verified update"]

    metadata = {
        "schema_version": "retrace_bench_general_1",
        "renderer": "split_template",
        "split": cfg.name,
        "seed": seed,
        "has_distractor": include_distractor,
        "has_cross_scope_trap": include_cross_scope,
        "verified_contradicts_trusted_note": verified_over_trusted,
        "requires_rejecting_false_premise": false_premise,
        "requires_non_answer_action": non_answer,
        "introduced_memory_ids": [replacement],
        "event_count": len(events),
        "memory_count": len(initial_memory),
    }

    scenario = {
        "scenario_id": sid,
        "domain": domain,
        "primary_failure_mode": primary,
        "secondary_failure_modes": secondaries,
        "difficulty": difficulty,
        "workflow_context": f"{owner_title} is coordinating a {noun} in {scope} for item {case_id}.",
        "public_input": {"event_trace": events, "initial_memory": initial_memory},
        "tasks": [
            {"task_id": f"t-{sid}-black-box", "task_type": "black_box_task", "prompt": f"What should the assistant do now for {case_id}?"},
            {"task_id": f"t-{sid}-memory-state", "task_type": "memory_state_task", "prompt": "Classify each memory as current, outdated, blocked, unresolved, out_of_scope, deleted, should_not_store, or restored."},
            {"task_id": f"t-{sid}-evidence", "task_type": "evidence_retrieval_task", "prompt": "Return the minimal event IDs that justify the decision."},
            {"task_id": f"t-{sid}-diagnostic", "task_type": "diagnostic_task", "prompt": "If an assistant follows the wrong note here, what memory reliability failure occurred?"},
        ],
        "hidden_gold": {
            "expected_answer": expected_answer,
            "expected_decision": spec["decision"],
            "expected_evidence_event_ids": [evidence_event],
            "expected_memory_state": expected_state,
            "expected_failure_diagnosis": primary,
            "stale_or_wrong_answers": stale_or_wrong,
            "rubric": rubric,
        },
        "metadata": metadata,
    }

    if cfg.include_training_targets:
        scenario["training_targets"] = _build_training_targets(
            primary, target, replacement, evidence_event, expected_state
        )

    return scenario


def _build_training_targets(primary: str, target: str, replacement: str,
                            evidence_event: str, expected_state: dict[str, str]) -> dict:
    """Supervision target for future ReTrace-Learn method training.

    Uses only the canonical typed-revision vocabulary and method-visible
    structure (candidate memory IDs + the grounding evidence event). It does not
    add any new ground-truth beyond what ``hidden_gold`` already encodes.
    """
    action_type = ACTION_BY_MODE[primary]
    if action_type == "SUPERSEDES":
        action = {
            "action_type": "SUPERSEDES",
            "target_memory_id": target,
            "replacement_memory_id": replacement,
            "evidence_event_ids": [evidence_event],
            "rationale": "Verified, in-scope, temporally latest evidence supersedes the prior belief.",
        }
    elif action_type == "BLOCKS":
        action = {
            "action_type": "BLOCKS",
            "target_memory_id": replacement,
            "replacement_memory_id": None,
            "evidence_event_ids": [evidence_event],
            "rationale": "Policy or lack of verified support blocks storing the proposed memory.",
        }
    elif action_type == "RELEASES":
        action = {
            "action_type": "RELEASES",
            "target_memory_id": target,
            "replacement_memory_id": None,
            "evidence_event_ids": [evidence_event],
            "rationale": "Verified release notice clears the temporary block and restores the belief.",
        }
    elif action_type == "UNCERTAIN":
        action = {
            "action_type": "UNCERTAIN",
            "target_memory_id": target,
            "replacement_memory_id": None,
            "evidence_event_ids": [evidence_event],
            "rationale": "Two verified but incompatible updates leave the belief unresolved.",
        }
    else:  # NO_REVISION
        action = {
            "action_type": "NO_REVISION",
            "target_memory_id": target,
            "replacement_memory_id": None,
            "evidence_event_ids": [evidence_event],
            "rationale": "No admissible defeat path; keep the current in-scope belief unchanged.",
        }

    return {
        "typed_revision_actions": [action],
        "target_memory_state": dict(expected_state),
        "supporting_evidence_event_ids": [evidence_event],
        "evidence_graph": {
            "nodes": [
                {"id": evidence_event, "kind": "evidence"},
                {"id": target, "kind": "belief"},
                {"id": replacement, "kind": "candidate"},
            ],
            "edges": [
                {"from": evidence_event, "to": target, "type": action_type},
            ],
        },
    }


def _coverage_summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    by_domain = Counter(r["domain"] for r in rows)
    by_mode = Counter(r["primary_failure_mode"] for r in rows)
    by_decision = Counter(r["hidden_gold"]["expected_decision"] for r in rows)
    by_difficulty = Counter(r["difficulty"] for r in rows)
    rates = {
        "events_ge_7": sum(len(r["public_input"]["event_trace"]) >= 7 for r in rows) / n,
        "memories_ge_3": sum(len(r["public_input"]["initial_memory"]) >= 3 for r in rows) / n,
        "distractors": sum(r["metadata"]["has_distractor"] for r in rows) / n,
        "cross_scope": sum(r["metadata"]["has_cross_scope_trap"] for r in rows) / n,
        "verified_over_trusted": sum(r["metadata"]["verified_contradicts_trusted_note"] for r in rows) / n,
        "false_premise": sum(r["metadata"]["requires_rejecting_false_premise"] for r in rows) / n,
        "non_answer": sum(r["metadata"]["requires_non_answer_action"] for r in rows) / n,
    }
    return {
        "count": len(rows),
        "by_domain": dict(sorted(by_domain.items())),
        "by_failure_mode": dict(sorted(by_mode.items())),
        "by_expected_decision": dict(sorted(by_decision.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "rates": {k: round(v, 4) for k, v in rates.items()},
    }


def _write_split(cfg: SplitConfig, out_dir: Path) -> dict:
    cfg.validate()
    rows = [build_split_scenario(cfg, p) for p in range(cfg.count)]
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_path = out_dir / "scenarios.jsonl"
    with scenarios_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _coverage_summary(rows)
    manifest = {
        "dataset_name": cfg.name,
        "scenario_count": cfg.count,
        "schema_version": "retrace_bench_general_1",
        "renderer": "split_template",
        "split": cfg.name,
        "purpose": cfg.purpose,
        "scenario_id_prefix": cfg.scenario_prefix,
        "seed_range": [cfg.seed_base, cfg.seed_base + cfg.count - 1],
        "entity_prefixes": {
            "case": cfg.case_prefix,
            "project": cfg.project_prefix,
            "person": cfg.person_prefix,
            "workspace": cfg.scope_prefix,
        },
        "has_training_targets": cfg.include_training_targets,
        "coverage": summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(cfg, summary), encoding="utf-8")
    print(f"Wrote {cfg.count} scenarios to {scenarios_path}")
    return {"config": cfg, "summary": summary, "path": scenarios_path}


def _readme(cfg: SplitConfig, summary: dict) -> str:
    held_out = not cfg.include_training_targets and cfg.name.startswith("test")
    lines = [
        f"# {cfg.name}",
        "",
        cfg.purpose,
        "",
        "## Provenance",
        "",
        f"- Schema: `retrace_bench_general_1` (general benchmark style).",
        f"- Scenario ID prefix: `{cfg.scenario_prefix}`.",
        f"- Deterministic seed range: `{cfg.seed_base}`-`{cfg.seed_base + cfg.count - 1}`.",
        f"- Entity prefixes: case `{cfg.case_prefix}`, project `{cfg.project_prefix}`, "
        f"person `{cfg.person_prefix}`, workspace `{cfg.scope_prefix}`.",
        f"- `training_targets` present: `{str(cfg.include_training_targets).lower()}`.",
        "",
        "This split shares no scenario, case, entity, memory ID, event ID, exact text, "
        "hidden gold, or seed range with the other splits (see "
        "`docs/retrace_bench/split_leakage_report.md`).",
        "",
        "## Model input vs. evaluation",
        "",
        "- Model input: `workflow_context`, `public_input`, `tasks` only.",
        "- Evaluation may read `hidden_gold`.",
    ]
    if cfg.include_training_targets:
        lines += [
            "- `training_targets` (typed revision actions, target memory state, supporting "
            "evidence, optional evidence graph) is for future ReTrace-Learn "
            "method training and selection. It is **not** model input at evaluation time.",
        ]
    if held_out:
        lines += [
            "",
            "## Held-out policy",
            "",
            "`test_800_en` is the internal ReTrace-Bench held-out evaluation set: **no** "
            "training, **no** prompt tuning, **no** policy optimization, and **no** checkpoint "
            "selection may use it.",
        ]
    lines += [
        "",
        "## Coverage",
        "",
        f"- Scenarios: {summary['count']}",
        f"- Domains: {len(summary['by_domain'])}/8; failure modes: {len(summary['by_failure_mode'])}/11.",
        f"- Decisions: {summary['by_expected_decision']}",
        f"- Rates: {summary['rates']}",
        "",
        "## Regenerate",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/generate_retrace_data_splits.py "
        "--train-count 3000 --dev-count 400 --test-count 800",
        "```",
        "",
    ]
    return "\n".join(lines)


def split_configs(train_count: int, dev_count: int, test_count: int) -> list[SplitConfig]:
    return [
        SplitConfig(
            name="train_3000_en",
            scenario_prefix="rt-train-",
            seed_base=100000,
            seed_max=199999,
            count=train_count,
            case_prefix="CTR-",
            project_prefix="PRTR-",
            person_prefix="PTR-",
            scope_prefix="ws-tr",
            include_training_targets=True,
            purpose=(
                "Synthetic supervision pool for future ReTrace-Learn open-model typed "
                "revision proposer training (Graph Extractor -> Typed Revision Proposer -> "
                "Authorization Court). "
                "This is **not** a benchmark and must never be scored as held-out evaluation."
            ),
        ),
        SplitConfig(
            name="dev_400_en",
            scenario_prefix="rt-dev-",
            seed_base=200000,
            seed_max=219999,
            count=dev_count,
            case_prefix="CDV-",
            project_prefix="PRDV-",
            person_prefix="PDV-",
            scope_prefix="ws-dv",
            include_training_targets=True,
            purpose=(
                "Development / selection split for prompt selection, policy selection, "
                "checkpoint selection, and validation-gated prompt/policy edits."
            ),
        ),
        SplitConfig(
            name="test_800_en",
            scenario_prefix="rt-test-",
            seed_base=300000,
            seed_max=399999,
            count=test_count,
            case_prefix="CTE-",
            project_prefix="PRTE-",
            person_prefix="PTE-",
            scope_prefix="ws-te",
            include_training_targets=False,
            purpose=(
                "Held-out internal ReTrace-Bench evaluation set. No training, no prompt "
                "tuning, no policy optimization, no checkpoint selection."
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--dev-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=800)
    parser.add_argument("--supervision-root", default="data/retrace_supervision")
    parser.add_argument("--bench-root", default="data/retrace_bench")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    configs = split_configs(args.train_count, args.dev_count, args.test_count)
    out_dirs = {
        "train_3000_en": repo / args.supervision_root / "train_3000_en",
        "dev_400_en": repo / args.supervision_root / "dev_400_en",
        "test_800_en": repo / args.bench_root / "test_800_en",
    }
    for cfg in configs:
        _write_split(cfg, out_dirs[cfg.name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
