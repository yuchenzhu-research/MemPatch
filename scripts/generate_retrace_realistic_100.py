#!/usr/bin/env python3
"""Generate the ``realistic_100_en`` ReTrace-Bench v1.0 scaffold (no gold).

``realistic_100_en`` (public split name: **`realistic`**) is a realistic-style
workflow split. Human annotation is **NOT** done in this pass:

* ``source_type = "realistic_style_synthetic"`` (generated from realistic
  workflow patterns, not collected from public sources).
* ``annotation_status = "pending"``.
* ``hidden_gold`` fields are intentionally left empty — there is **no synthetic
  gold and no fabricated human annotation**.
* An ``annotations_template.jsonl`` with one empty row per scenario is written
  for the human annotation pass the user will run later.

Each scenario still has realistic workflow texture, de-actionalized records (no
action-word leakage), no hidden labels in public text, and structural metadata
flags describing how the case was constructed (these describe construction, not
a validated label).

Category mix (100 total): 40 software/GitHub-style, 20 support/CRM, 15
research/citation, 15 calendar/task, 10 enterprise approval / multi-tool.

Determinism: output depends only on ``--count`` and ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import DIFFICULTIES, FAILURE_MODES
from benchmark.retrace_bench.generation.deactionalized import (
    DECISIONS_BY_MODE,
    DOMAIN_FRAMES,
    EVIDENCE_LABELS,
    _state_text,
)
from benchmark.retrace_bench.generation.release_manifest import (
    BENCHMARK_VERSION,
    build_manifest,
)

SEED = 5050000

# (domain, count) — sums to 100.
CATEGORY_MIX = (
    ("software_engineering_agent", 40),
    ("customer_support_crm", 20),
    ("research_knowledge_work", 15),
    ("calendar_task_workflow", 15),
    ("enterprise_multi_tool_workflow", 10),
)

CATEGORY_TEXTURE = {
    "software_engineering_agent": {
        "unit": "pull request", "system": "the repository's checks dashboard",
        "actors": ("a maintainer", "a reviewer", "the release bot"),
    },
    "customer_support_crm": {
        "unit": "support ticket", "system": "the CRM record of truth",
        "actors": ("a support agent", "the customer", "a team lead"),
    },
    "research_knowledge_work": {
        "unit": "citation entry", "system": "the reference manager of record",
        "actors": ("a co-author", "a librarian", "the corresponding author"),
    },
    "calendar_task_workflow": {
        "unit": "calendar task", "system": "the shared scheduling system",
        "actors": ("a teammate", "the organizer", "an assistant"),
    },
    "enterprise_multi_tool_workflow": {
        "unit": "approval request", "system": "the workflow system of record",
        "actors": ("an approver", "the requester", "a compliance reviewer"),
    },
}


def _domain_sequence(count: int) -> list[str]:
    seq: list[str] = []
    for domain, n in CATEGORY_MIX:
        seq.extend([domain] * n)
    # If a custom count is requested, pad/truncate against the mix proportions.
    if count <= len(seq):
        return seq[:count]
    base = seq[:]
    i = 0
    while len(base) < count:
        base.append(seq[i % len(seq)])
        i += 1
    return base


def _ts_sequence(rng: random.Random, count: int) -> list[str]:
    cur = datetime(2026, 5, 4, 9, 0) + timedelta(days=rng.randint(0, 9))
    out: list[str] = []
    for _ in range(count):
        cur += timedelta(minutes=rng.choice((8, 14, 22, 41, 70, 130, 240)))
        out.append(cur.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out


def build_scenario(index: int, domain: str, seed: int = SEED) -> dict:
    rng = random.Random(seed + index * 7331)
    tex = CATEGORY_TEXTURE[domain]
    frame = DOMAIN_FRAMES.get(domain, DOMAIN_FRAMES["software_engineering_agent"])
    topic = frame["topics"][(index // 3) % len(frame["topics"])]
    failure_mode = FAILURE_MODES[index % len(FAILURE_MODES)]
    # design-only decision used to shape de-actionalized state text; NOT stored
    # as gold (annotation is pending).
    design_decisions = DECISIONS_BY_MODE[failure_mode]
    design_decision = design_decisions[(index // len(FAILURE_MODES)) % len(design_decisions)]

    sid = f"rb-real-100-en-{index + 1:05d}"
    scope = f"team-RS{index % 13:02d}"
    other_scope = f"team-RS{(index + 4) % 13:02d}"
    case_id = f"RS-{300000 + index}"
    project_id = f"WORK-RS{(index * 37) % 887:03d}"
    unit = tex["unit"]
    system = tex["system"]
    actor_main, actor_other, actor_third = tex["actors"]

    target = f"m-{sid}-target"
    condition = f"m-{sid}-condition"
    distractor = f"m-{sid}-distractor"
    ctx1 = f"m-{sid}-ctx1"
    ctx2 = f"m-{sid}-ctx2"
    n_ctx = rng.randint(0, 3)
    ctx_extra = [f"m-{sid}-ctx{k}" for k in range(3, 3 + n_ctx)]

    include_cross = (index % 3) == 0  # ~33%, not universal
    requires_non_answer = (index % 9) < 5  # ~55%

    old_fact = f"{case_id} still points to the earlier {topic} for {project_id}."
    state = _state_text(
        failure_mode=failure_mode, decision=design_decision, case_id=case_id,
        project_id=project_id, scope=scope, other_scope=other_scope, frame=frame,
        topic=topic, sibling_id=condition,
    )
    label = EVIDENCE_LABELS[(index * 3 + 2) % len(EVIDENCE_LABELS)]

    initial_memory = [
        {"memory_id": target, "text": f"For {case_id}, {old_fact}", "visibility_scope": scope,
         "source_event_ids": [f"e-{sid}-01"], "is_distractor": False},
        {"memory_id": condition,
         "text": f"The {unit} {case_id} is valid only while its {topic} source stays current in {scope}.",
         "visibility_scope": scope, "source_event_ids": [f"e-{sid}-01"], "is_distractor": False},
        {"memory_id": ctx1,
         "text": f"{actor_main.title()} handles routine updates for {project_id} in {scope}.",
         "visibility_scope": scope, "source_event_ids": [f"e-{sid}-01"], "is_distractor": False},
        {"memory_id": distractor,
         "text": f"A similar {unit} in {other_scope} tracks a different {topic}.",
         "visibility_scope": other_scope, "source_event_ids": [f"e-{sid}-01"], "is_distractor": True},
    ]
    for mid in (ctx2, *ctx_extra):
        initial_memory.append({
            "memory_id": mid,
            "text": f"Background note {mid[-1]} for {project_id} stays relevant for routine routing in {scope}.",
            "visibility_scope": scope, "source_event_ids": [f"e-{sid}-01"], "is_distractor": False,
        })
    initial_memory = initial_memory[: rng.randint(4, 8)]
    memory_ids = {m["memory_id"] for m in initial_memory}

    # ---- natural workflow events ----------------------------------------
    raw: list[tuple[str, str, str, str, str, list[str]]] = []
    raw.append(("intro", f"{actor_main.title()} opened {unit} {case_id} in {scope}: {old_fact}", "trusted", "workflow_note", scope, [target]))
    raw.append(("status", f"{system} listed {case_id} with its prior {topic} still attached.", "trusted", "tool_snapshot", scope, [target, condition]))
    raw.append(("discussion", f"{actor_other.title()} asked {actor_main} to double-check the {topic} on {case_id} before relying on it.", "trusted", "thread", scope, [condition]))
    raw.append(("request", f"{actor_third.title()} now asks how to proceed on {case_id} using only what is visible in {scope}.", "trusted", "user_request", scope, [target]))
    raw.append(("verified", f"{label}: {state}", "verified", "system_of_record", scope, [target]))
    if include_cross:
        raw.append(("cross", f"A {other_scope} thread mentions a similar {topic} for a different {unit}.", "untrusted", "cross_team_forward", other_scope, [distractor]))
    raw.append(("late", f"A later forwarded message restates the original {topic} for {case_id} as final, without pointing to {system}.", "untrusted", "chat_forward", scope, [target]))

    target_len = rng.randint(15, 50)
    n_filler = max(0, target_len - len(raw))
    filler_lines = [
        f"A routine sync touched {project_id} without changing the {topic}.",
        f"{actor_main.title()} left a short status update on {case_id} for visibility.",
        f"An automated check posted a read-only summary for {scope}.",
        f"A teammate acknowledged the {unit} thread for {case_id}.",
        f"A reminder fired for routine follow-up on {project_id}.",
        f"A passive log entry recorded access to {case_id} in {scope}.",
    ]
    for k in range(n_filler):
        raw.append((f"f{k}", f"Update {k + 1}: {filler_lines[k % len(filler_lines)]}", "trusted", "activity_log", scope, []))

    timestamps = _ts_sequence(rng, len(raw))
    events = []
    key_to_id: dict[str, str] = {}
    for i, (key, text, trust, source, scp, related) in enumerate(raw, start=1):
        eid = f"e-{sid}-{i:02d}"
        key_to_id[key] = eid
        related = [m for m in related if m in memory_ids]
        events.append({
            "event_id": eid,
            "timestamp": timestamps[i - 1],
            "source": source,
            "actor": "system_of_record" if trust == "verified" else actor_main,
            "event_type": "workflow_event",
            "text": text,
            "trust_level": trust,
            "visibility_scope": scp,
            "related_memory_ids": related,
        })

    designed_evidence_count = rng.randint(1, 3)

    return {
        "scenario_id": sid,
        "split": "realistic",
        "domain": domain,
        "primary_failure_mode": failure_mode,
        "secondary_failure_modes": [],
        "difficulty": DIFFICULTIES[index % len(DIFFICULTIES)],
        "workflow_context": f"{actor_main.title()} is working through {unit} {case_id} and related notes in {scope}.",
        "public_input": {"event_trace": events, "initial_memory": initial_memory},
        "tasks": [
            {"task_id": f"t-{sid}-black-box", "task_type": "black_box_task",
             "prompt": f"What should the assistant do now for {case_id}?"},
            {"task_id": f"t-{sid}-memory-state", "task_type": "memory_state_task",
             "prompt": "Classify each memory as current, outdated, blocked, unresolved, out_of_scope, deleted, should_not_store, or restored."},
            {"task_id": f"t-{sid}-evidence", "task_type": "evidence_retrieval_task",
             "prompt": "Return the minimal event IDs that justify the decision."},
            {"task_id": f"t-{sid}-diagnostic", "task_type": "diagnostic_task",
             "prompt": "If the assistant mishandles the memory here, which single failure mode from the enum best describes it?"},
        ],
        # No synthetic gold and no fabricated human annotation: annotation pending.
        "hidden_gold": {
            "expected_answer": "",
            "expected_decision": "",
            "expected_evidence_event_ids": [],
            "expected_memory_state": {},
            "expected_failure_diagnosis": "",
            "rubric": {},
            "annotation_status": "pending",
        },
        "metadata": {
            "schema_version": "retrace_bench_general_1",
            "renderer": "realistic_v1_scaffold",
            "split": "realistic",
            "benchmark_version": BENCHMARK_VERSION,
            "source_type": "realistic_style_synthetic",
            "annotation_status": "pending",
            "category": domain,
            # structural construction flags (describe how the case was built, not
            # a validated human label):
            "has_distractor": True,
            "has_cross_scope_trap": include_cross,
            "verified_contradicts_trusted_note": True,
            "requires_rejecting_false_premise": True,
            "requires_non_answer_action": requires_non_answer,
            "event_count": len(events),
            "memory_count": len(initial_memory),
            "designed_required_evidence_count": designed_evidence_count,
            "design_failure_mode_hint": failure_mode,
            "seed": seed + index,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_annotations_template(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "scenario_id": row["scenario_id"],
                "annotator_id": "",
                "expected_decision": "",
                "memory_state": {},
                "evidence_event_ids": [],
                "failure_diagnosis": "",
                "ambiguous_or_invalid": None,
                "notes": "",
            }, ensure_ascii=False) + "\n")


def write_readme(out_dir: Path, manifest: dict) -> None:
    counts = manifest.get("category_counts", {})
    cat_lines = "\n".join(f"- `{k}`: {v}" for k, v in counts.items())
    text = f"""# ReTrace-Bench `realistic_100_en` (v{BENCHMARK_VERSION})

Realistic-style workflow split of ReTrace-Bench v1.0 (public split name:
**`realistic`**).

> **Annotation is NOT done in this pass.** These scenarios are
> `realistic_style_synthetic` (generated from realistic workflow patterns, not
> collected from public sources). `annotation_status` is **`pending`**.
> `hidden_gold` fields are intentionally empty — there is **no synthetic gold
> and no fabricated human annotation**. Use `annotations_template.jsonl` for the
> human annotation pass.

- **Scenarios:** {manifest['scenario_count']}
- **Events per scenario:** {manifest['min_event_count']}-{manifest['max_event_count']} (avg {manifest['avg_event_count']})
- **Memories per scenario:** {manifest['min_memory_count']}-{manifest['max_memory_count']}
- **Source type:** `{manifest['source_type']}`
- **Annotation status:** `{manifest['annotation_status']}`
- **Benchmark version:** `{manifest['version']}`

## Category mix

{cat_lines}

## Files

- `scenarios.jsonl` — realistic-style scenarios (de-actionalized, no gold).
- `annotations_template.jsonl` — one empty row per scenario for human annotation.
- `manifest.json` — split manifest + leakage audit summary.

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_realistic_100.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \\
  --data data/retrace_bench/realistic_100_en/scenarios.jsonl
```
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="data/retrace_bench/realistic_100_en/scenarios.jsonl")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    domains = _domain_sequence(args.count)
    rows = [build_scenario(i, domains[i], seed=args.seed) for i in range(args.count)]
    out = Path(args.out)
    write_jsonl(out, rows)
    write_annotations_template(out.parent.joinpath("annotations_template.jsonl"), rows)

    event_counts = [len(r["public_input"]["event_trace"]) for r in rows]
    memory_counts = [len(r["public_input"]["initial_memory"]) for r in rows]
    category_counts: dict[str, int] = {}
    for r in rows:
        category_counts[r["domain"]] = category_counts.get(r["domain"], 0) + 1
    manifest = build_manifest(
        rows,
        split="realistic",
        source_type="realistic_style_synthetic",
        annotation_status="pending",
        role="Realistic-style workflow split; human annotation pending (no synthetic gold).",
        extra={
            "min_event_count": min(event_counts),
            "min_memory_count": min(memory_counts),
            "category_counts": category_counts,
            "annotations_template": "annotations_template.jsonl",
        },
    )
    out.parent.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_readme(out.parent, manifest)
    print(f"wrote {len(rows)} scenarios to {out}")
    print(f"category mix: {category_counts}")
    print(f"event range: {min(event_counts)}-{max(event_counts)}; memory range: {min(memory_counts)}-{max(memory_counts)}")
    print(f"leakage_audit clean={manifest['leakage_audit_summary']['clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
