#!/usr/bin/env python3
"""Build a formal Stage C Silver v1 SFT dataset for MLX-LM QLoRA training.

This script improves upon v0 by generating more complex reasoning episodes, 
enforcing structured coverage quotas, and verifying all properties with a strict 
Sanity Checker.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    TypedRevisionTarget,
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
    StageCTrainingExample,
    DownstreamTask,
    GoldSnapshotExpectation,
)
from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.export_stagec_sft import (
    SYSTEM_PROMPT,
    format_assistant_response,
    format_user_prompt,
)
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)

OUT_DIR = Path("outputs/local_training/stagec_qwen3_4b_silver_v1/data")
MANIFEST_PATH = Path("outputs/local_training/stagec_qwen3_4b_silver_v1/manifest.json")
RNG_SEED = 42

# Surface replacements for train-set augmentations
TRAIN_AUGMENTATIONS = [
    ("base", {}),
    ("surface_qa", {
        "Staging": "QA",
        "staging": "QA",
        "Production": "Live",
        "production": "live",
        "Module": "Service",
        "module": "service",
        "Project": "Workspace",
        "project": "workspace",
    }),
    ("surface_preprod", {
        "Staging": "Pre-production",
        "staging": "pre-production",
        "Production": "Production",
        "production": "production",
        "Module": "Component",
        "module": "component",
        "Experiment": "Evaluation run",
        "experiment": "evaluation run",
    }),
    ("surface_canary", {
        "Staging": "Canary",
        "staging": "canary",
        "Production": "Primary",
        "production": "primary",
        "build": "release",
        "Build": "Release",
        "study": "analysis",
        "Study": "Analysis",
    }),
    ("surface_sandbox", {
        "Staging": "Sandbox",
        "staging": "sandbox",
        "Production": "Serving",
        "production": "serving",
        "pipeline": "workflow",
        "Pipeline": "Workflow",
        "dataset": "corpus",
        "Dataset": "Corpus",
    }),
    ("surface_validation", {
        "Staging": "Validation",
        "staging": "validation",
        "Production": "Deployed",
        "production": "deployed",
        "server": "endpoint",
        "Server": "Endpoint",
        "parser": "processor",
        "Parser": "Processor",
    }),
    ("surface_testbed", {
        "Staging": "Testbed",
        "staging": "testbed",
        "Production": "Customer-facing",
        "production": "customer-facing",
        "database": "storage",
        "Database": "Storage",
        "report": "record",
        "Report": "Record",
    }),
    ("surface_shadow", {
        "Staging": "Shadow",
        "staging": "shadow",
        "Production": "Mainline",
        "production": "mainline",
        "configuration": "setting",
        "Configuration": "Setting",
        "release": "snapshot",
        "Release": "Snapshot",
    }),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_replacements(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(old, new)
    return out


def make_example(ep: FixedCandidateInputEpisode, sub: FixedCandidateSubmission, targets: Tuple[TypedRevisionTarget, ...]) -> StageCTrainingExample:
    return StageCTrainingExample(
        example_id=f"silver_{ep.episode_id}_{sub.submission_id}",
        episode_id=ep.episode_id,
        submission_id=sub.submission_id,
        method_visible_input=sub,
        targets=targets,
        split="silver_synthetic",
        domain=ep.domain,
        failure_type=ep.failure_type_public_or_controlled,
        label_source="template_authored_executable_silver_v1",
        metadata={
            "scientific_status": "silver_synthetic_training_only",
            "not_for_paper_main_results": True,
            "contains_gold_in_user_input": False,
        },
    )


def to_chat_row(ex: StageCTrainingExample, augmentation_name: str, mapping: Dict[str, str]) -> Dict[str, Any]:
    user_text = format_user_prompt(ex)
    assistant_text = format_assistant_response(ex)

    augmented_episode_id = f"{ex.episode_id}__{augmentation_name}"
    user_text = user_text.replace(ex.episode_id, augmented_episode_id)
    assistant_text = assistant_text.replace(ex.episode_id, augmented_episode_id)

    user_text = apply_replacements(user_text, mapping)
    assistant_text = apply_replacements(assistant_text, mapping)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "source_episode_id": ex.episode_id,
            "augmented_episode_id": augmented_episode_id,
            "source_submission_id": ex.submission_id,
            "domain": ex.domain,
            "failure_type": ex.failure_type,
            "augmentation": augmentation_name,
            "actions": [t.action_type for t in ex.targets],
            "scientific_status": "silver_synthetic_training_only",
            "not_for_paper_main_results": True,
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def generate_new_silver_v1_episodes() -> List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    """Generates 70 new base episodes targeting specific hard, reaffirms, and multi-action quotas."""
    episodes: List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]] = []
    domains = ["software_engineering", "research_workflow"]
    
    # 7 Failure/推理模式
    patterns = [
        "reaffirms_only",
        "grounding_hard",
        "multi_action_supersedes_blocks",
        "multi_action_supersedes_releases",
        "multi_action_blocks_uncertain",
        "evidence_conflict",
        "target_ambiguity",
    ]

    for domain in domains:
        for pat in patterns:
            for v in range(1, 6):  # 5 variants -> 70 episodes total
                episode_id = f"ep_expansion_v1_{domain}_{pat}_v{v}"
                
                meta = {
                    "review_status": "pending_human_review",
                    "training_eligible": False,
                    "scientific_status": "pipeline_validation_only",
                    "label_source": "template_authored_silver_v1",
                }

                sub1_id = f"sub_{episode_id}_1"
                sub2_id = f"sub_{episode_id}_2"
                ev1_id = f"ev_{episode_id}_1"
                ev2_id = f"ev_{episode_id}_2"

                # -------------------------------------------------------------
                # 1. REAFFIRMS CASES
                # -------------------------------------------------------------
                if pat == "reaffirms_only":
                    if domain == "software_engineering":
                        query = f"Validate firewall policy config {v}"
                        b_init_text = f"Firewall policy {v} is active and compliant."
                        ev_init_text = f"Security scan logs for policy {v} pass compliance checks."
                        ev_new_text = f"Compliance audit report #10{v} re-verified firewall policy {v} as fully compliant."
                    else:
                        query = f"Check baseline accuracy for algorithm {v}"
                        b_init_text = f"Algorithm {v} baseline accuracy is verified at 92%."
                        ev_init_text = f"Initial benchmarks show algorithm {v} accuracy is 92%."
                        ev_new_text = f"Independent peer review replication study confirms algorithm {v} accuracy at 92%."

                    b1_id = f"b_{episode_id}_1"
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), ())
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "REAFFIRMS", target_belief_id=b1_id, rationale="Re-verified to be compliant, reaffirm config.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}
                    downstream_id = b1_id
                    downstream_expected = "AUTHORIZED"

                # -------------------------------------------------------------
                # 2. GROUNDING HARD CASES (Similar names, precise ID check)
                # -------------------------------------------------------------
                elif pat == "grounding_hard":
                    if domain == "software_engineering":
                        query = f"Resolve config settings for server cluster {v}"
                        b_init_1 = f"Server 1 in cluster {v} has 16GB memory."
                        b_init_2 = f"Server 2 in cluster {v} has 32GB memory."
                        b_up_text = f"Server 1 in cluster {v} memory upgraded to 64GB."
                        ev_init_text = f"Hardware discovery database registers cluster {v} specifications."
                        ev_new_text = f"System administrator console log: server 1 upgraded to 64GB."
                    else:
                        query = f"Verify database sync counts for study {v}"
                        b_init_1 = f"Data source A in study {v} contains 500 records."
                        b_init_2 = f"Data source B in study {v} contains 1000 records."
                        b_up_text = f"Data source A in study {v} record count expanded to 2000."
                        ev_init_text = f"Metadata database registry records sizes for study {v}."
                        ev_new_text = f"Sync server transaction count log: source A verified at 2000 records."

                    b1_id = f"b_{episode_id}_server1"
                    b2_id = f"b_{episode_id}_server2"
                    b_up_id = f"b_{episode_id}_server1_up"

                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_1, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_init_2, (ev1_id,))
                    b_up = BeliefNode(b_up_id, b_up_text, (ev2_id,))

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1, b2), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1, b2), (b_up,))
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b_up_id, rationale="Server 1 upgraded. Server 2 unchanged.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", b_up_id: "AUTHORIZED"}
                    downstream_id = b_up_id
                    downstream_expected = "AUTHORIZED"

                # -------------------------------------------------------------
                # 3. MULTI-ACTION: SUPERSEDES + BLOCKS
                # -------------------------------------------------------------
                elif pat == "multi_action_supersedes_blocks":
                    if domain == "software_engineering":
                        query = f"Check container build stability for stack {v}"
                        b_init_1 = f"Container {v} runs software version 1.0."
                        b_init_2 = f"Downstream balancer {v} is authorized to route traffic."
                        b_up_text = f"Container {v} is upgraded to version 2.0."
                        ev_init_text = f"Registry logs version 1.0 verified stable for stack {v}."
                        ev_new_text = f"Upgrade notification: software version 2.0 active. Warning: software version 1.0 is deprecated and unsafe."
                        c_text = f"Software version 1.0 remains stable for stack {v}."
                    else:
                        query = f"Check protocol execution for analysis {v}"
                        b_init_1 = f"Clinical protocol {v} uses reagent batch A."
                        b_init_2 = f"Experiment trial run {v} is active."
                        b_up_text = f"Clinical protocol {v} is updated to reagent batch B."
                        ev_init_text = f"Inventory sheet registers reagent batch A usage."
                        ev_new_text = f"Lab alert: batch B verified active. Reagent batch A is expired and invalid."
                        c_text = f"Reagent batch A is active and validated for analysis {v}."

                    b1_id = f"b_{episode_id}_1"
                    b2_id = f"b_{episode_id}_2"
                    b_up_id = f"b_{episode_id}_up"
                    c1_id = f"c_{episode_id}_1"
                    dep = DependencyEdge(f"dep_{episode_id}_2", b2_id, c1_id, "system")

                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_1, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_init_2, (ev1_id,))
                    b_up = BeliefNode(b_up_id, b_up_text, (ev2_id,))
                    cond = ConditionNode(c1_id, f"scope_{episode_id}_1", c_text)

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1, b2), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b1, b2),
                        candidate_replacement_beliefs=(b_up,),
                        candidate_conditions_by_belief=((b2_id, (cond,)),),
                        dependency_edges_by_belief=((b2_id, (dep,)),),
                    )
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b_up_id, rationale="Upgrade protocol/software.", evidence_ids=(ev2_id,)),
                        TypedRevisionTarget(sub2_id, "BLOCKS", target_condition_id=c1_id, rationale="Block old version/reagent.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b_up_id: "AUTHORIZED", b2_id: "BLOCKED"}
                    downstream_id = b2_id
                    downstream_expected = "BLOCKED"

                # -------------------------------------------------------------
                # 4. MULTI-ACTION: SUPERSEDES + RELEASES
                # -------------------------------------------------------------
                elif pat == "multi_action_supersedes_releases":
                    if domain == "software_engineering":
                        query = f"Verify routing and settings for balancer {v}"
                        b_init_1 = f"Old server balancer {v} is configured active."
                        b_init_2 = f"Route gateway {v} routing is active."
                        b_up_text = f"New software load balancer {v} is deployed."
                        ev_init_text = f"Cluster registry records balancer specifications."
                        ev_new_text = f"Cluster console event: balancer upgraded to software. API link gateway connection recovered."
                        c_text = f"API connection to route gateway {v} is active."
                    else:
                        query = f"Verify study datasets for trial {v}"
                        b_init_1 = f"Old backup log directory {v} is active."
                        b_init_2 = f"Primary research database {v} is active."
                        b_up_text = f"New metadata cloud storage {v} is configured."
                        ev_init_text = f"Backup pipeline initializes log storage registry."
                        ev_new_text = f"Data sync confirmation: storage upgraded to cloud. Ethical study permission verified."
                        c_text = f"Ethics board permission for database {v} is active."

                    b1_id = f"b_{episode_id}_stale"
                    b2_id = f"b_{episode_id}_active"
                    b_up_id = f"b_{episode_id}_stale_up"
                    c1_id = f"c_{episode_id}_released"
                    dep = DependencyEdge(f"dep_{episode_id}_active", b2_id, c1_id, "system")

                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_1, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_init_2, (ev1_id,))
                    b_up = BeliefNode(b_up_id, b_up_text, (ev2_id,))
                    cond = ConditionNode(c1_id, f"scope_{episode_id}_1", c_text)

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1, b2), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b1, b2),
                        candidate_replacement_beliefs=(b_up,),
                        candidate_conditions_by_belief=((b2_id, (cond,)),),
                        dependency_edges_by_belief=((b2_id, (dep,)),),
                    )
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b_up_id, rationale="Upgrade balancer/storage.", evidence_ids=(ev2_id,)),
                        TypedRevisionTarget(sub2_id, "RELEASES", target_condition_id=c1_id, rationale="Release connection/permission blocker.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b_up_id: "AUTHORIZED", b2_id: "AUTHORIZED"}
                    downstream_id = b2_id
                    downstream_expected = "AUTHORIZED"

                # -------------------------------------------------------------
                # 5. MULTI-ACTION: BLOCKS + UNCERTAIN
                # -------------------------------------------------------------
                elif pat == "multi_action_blocks_uncertain":
                    if domain == "software_engineering":
                        query = f"Validate backend stability for deployment {v}"
                        b_init_1 = f"Kubernetes container {v} deployment is healthy."
                        b_init_2 = f"Cluster service balancer {v} is routing traffic."
                        ev_init_text = f"Initial standard orchestration reports container verified health."
                        ev_new_text = f"Console warnings: container {v} health report failed validation check. Balancer routing status unstable."
                        c_text = f"Orchestrator health check verified for container {v}."
                    else:
                        query = f"Check study participant status for trial {v}"
                        b_init_1 = f"Clinical trial {v} patient database metrics are stable."
                        b_init_2 = f"Statistical database run {v} is authorized to proceed."
                        ev_init_text = f"Initial ethics log confirms stable patient enrolment metrics."
                        ev_new_text = f"Clinical coordinator warning: patient metrics data drift detected. Run validation check failed."
                        c_text = f"Validation check verified for trial {v} metrics."

                    b1_id = f"b_{episode_id}_primary"
                    b2_id = f"b_{episode_id}_downstream"
                    c1_id = f"c_{episode_id}_downstream_cond"
                    dep = DependencyEdge(f"dep_{episode_id}_downstream", b2_id, c1_id, "system")

                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_1, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_init_2, (ev1_id,))
                    cond = ConditionNode(c1_id, f"scope_{episode_id}_1", c_text)

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1, b2), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b1, b2),
                        candidate_replacement_beliefs=(),
                        candidate_conditions_by_belief=((b2_id, (cond,)),),
                        dependency_edges_by_belief=((b2_id, (dep,)),),
                    )
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "UNCERTAIN", target_belief_id=b1_id, rationale="Container/trial data warnings create uncertainty.", evidence_ids=(ev2_id,)),
                        TypedRevisionTarget(sub2_id, "BLOCKS", target_condition_id=c1_id, rationale="Validation check failed, block downstream active routing.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED", b2_id: "BLOCKED"}
                    downstream_id = b1_id
                    downstream_expected = "UNRESOLVED"

                # -------------------------------------------------------------
                # 6. EVIDENCE CONFLICT (NO_REVISION target)
                # -------------------------------------------------------------
                elif pat == "evidence_conflict":
                    if domain == "software_engineering":
                        query = f"Validate CPU security logs for stack {v}"
                        b_init_text = f"CPU microcode configuration {v} is secure."
                        ev_init_text = f"Linter logs: security microcode setup validated."
                        ev_new_text = f"Duplicate event log: CPU security configuration {v} verification success."
                    else:
                        query = f"Verify replicability metadata for experiment {v}"
                        b_init_text = f"Replicability metrics for study {v} validation pass."
                        ev_init_text = f"Audit log: trial data metrics check pass."
                        ev_new_text = f"Duplicate validation sheet: study {v} metadata replication success."

                    b1_id = f"b_{episode_id}_1"
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt", metadata={"duplicate_record": True})
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), ())
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "NO_REVISION", rationale="Duplicate confirmation log warrants no revision.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}
                    downstream_id = b1_id
                    downstream_expected = "AUTHORIZED"

                # -------------------------------------------------------------
                # 7. TARGET AMBIGUITY (UNCERTAIN target)
                # -------------------------------------------------------------
                elif pat == "target_ambiguity":
                    if domain == "software_engineering":
                        query = f"Validate logic configuration for framework {v}"
                        b_init_text = f"Logic gates config {v} is active and valid."
                        ev_init_text = f"Static checker confirmation: logic gate structure clean."
                        ev_new_text = f"Hedged hardware error message: logic gates config {v} could experience transient signals."
                    else:
                        query = f"Check experimental sensor drift for run {v}"
                        b_init_text = f"Sensor calibration run {v} output is clean and consistent."
                        ev_init_text = f"Sensor benchmark log verifies stable telemetry."
                        ev_new_text = f"Hedged calibration sensor log: telemetry run {v} shows signs of a potential minor drift."

                    b1_id = f"b_{episode_id}_1"
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "silver_v1", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_new_text, "silver_v1", f"file:///src/{domain}/v{v}/2.txt")
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))

                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), ())
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "UNCERTAIN", target_belief_id=b1_id, rationale="Hedged/unconfirmed error warrants uncertainty classification.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED"}
                    downstream_id = b1_id
                    downstream_expected = "UNRESOLVED"

                # Common task construction
                tasks = (
                    DownstreamTask(
                        task_id=f"task_{episode_id}",
                        query=query,
                        expected_answer_or_action=downstream_expected,
                        relevant_belief_ids=(downstream_id,),
                    ),
                )

                gold_snapshot = GoldSnapshotExpectation(
                    belief_statuses=gold_statuses,
                    rationale=f"Deterministic gold expectations for silver v1 pattern {pat}.",
                )

                episode = FixedCandidateInputEpisode(
                    episode_id=episode_id,
                    domain=domain,
                    failure_type_public_or_controlled=pat,
                    subagent_roles=tuple(set(s.producer_role for s in submissions)),
                    submissions=submissions,
                    downstream_tasks=tasks,
                    split="development_candidate",
                    protocol_mode="fixed_candidate_revision",
                    proposal_source="template_authored",
                    metadata=meta,
                )

                gold_record = FixedCandidateGoldRecord(
                    episode_id=episode_id,
                    gold_snapshot=gold_snapshot,
                    gold_typed_targets=tuple(targets),
                    failure_type=pat,
                    metadata=meta,
                )

                episodes.append((episode, gold_record))
                
    return episodes


def run_sanity_checker(
    labeled_examples: List[StageCTrainingExample],
    base_episodes_with_gold: List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]],
    train_rows: List[Dict[str, Any]],
    valid_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
) -> None:
    """Rigorous sanity checks validating schema references, distributions, split integrity, and leakage."""
    print("[*] Running Stage C Silver v1 Sanity Checker...")

    # 1. Verify every target action references valid memory/evidence IDs
    for ex in labeled_examples:
        sub = ex.method_visible_input
        valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs} | {b.belief_id for b in sub.candidate_replacement_beliefs}
        valid_condition_ids = set()
        for _, conds in sub.candidate_conditions_by_belief:
            for c in conds:
                valid_condition_ids.add(c.condition_id)
        valid_evidence_ids = {e.evidence_id for e in sub.evidence_context} | {sub.new_evidence_id}

        for target in ex.targets:
            # Action type verification
            assert target.action_type in {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}, \
                f"Non-canonical action: {target.action_type} in {ex.example_id}"
                
            # Grounding ID validity verification
            if target.target_belief_id:
                assert target.target_belief_id in valid_belief_ids, \
                    f"Action {target.action_type} targets invalid belief ID '{target.target_belief_id}' in example {ex.example_id}"
            if target.replacement_belief_id:
                assert target.replacement_belief_id in valid_belief_ids, \
                    f"Action {target.action_type} replacements invalid belief ID '{target.replacement_belief_id}' in example {ex.example_id}"
            if target.target_condition_id:
                assert target.target_condition_id in valid_condition_ids, \
                    f"Action {target.action_type} targets invalid condition ID '{target.target_condition_id}' in example {ex.example_id}"
            for ev_id in target.evidence_ids:
                assert ev_id in valid_evidence_ids, \
                    f"Action {target.action_type} cites invalid evidence ID '{ev_id}' in example {ex.example_id}"

    # 2. Verify duplicate example or episode IDs
    episode_ids = [ep.episode_id for ep, _ in base_episodes_with_gold]
    assert len(episode_ids) == len(set(episode_ids)), "Duplicate episode_id detected in base dataset!"
    
    example_ids = [ex.example_id for ex in labeled_examples]
    assert len(example_ids) == len(set(example_ids)), "Duplicate example_id detected in SFT dataset!"

    # 3. Verify action distribution quotas
    total_actions = Counter()
    multi_action_count = 0
    for ex in labeled_examples:
        for t in ex.targets:
            total_actions[t.action_type] += 1
        if len(ex.targets) > 1:
            multi_action_count += 1

    print(f"    - Total target action counts: {dict(total_actions)}")
    print(f"    - Multi-action examples count: {multi_action_count}")

    # Enforce basic quotas for stable training
    required_actions = {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}
    for act in required_actions:
        assert total_actions[act] >= 5, f"Quotas warning: action '{act}' has less than 5 representations (got {total_actions[act]})"
    assert multi_action_count >= 10, f"Quotas failure: multi-action examples count is too low (got {multi_action_count})"

    # 4. Verify train/valid/test split integrity and no leakage
    train_eps = {r["metadata"]["source_episode_id"] for r in train_rows}
    valid_eps = {r["metadata"]["source_episode_id"] for r in valid_rows}
    test_eps = {r["metadata"]["source_episode_id"] for r in test_rows}

    # Intersections must be empty
    assert not (train_eps & valid_eps), "Leakage detected: train and valid splits overlap!"
    assert not (train_eps & test_eps), "Leakage detected: train and test splits overlap!"
    assert not (valid_eps & test_eps), "Leakage detected: valid and test splits overlap!"

    # Split characteristics
    for ep_id in train_eps:
        assert not ep_id.endswith("_v5"), f"Leakage: held-out v5 case '{ep_id}' leaked into train set!"
    for ep_id in (valid_eps | test_eps):
        assert ep_id.endswith("_v5"), f"Leakage: validation/test case '{ep_id}' must belong to held-out v5 family!"

    # 5. Verify NO STALE external validation leakage
    for ep_id in (train_eps | valid_eps | test_eps):
        # STALE gold benchmark namespaces must not leak (they use 'ep_stale_' or 'cupmem')
        assert not ep_id.lower().startswith("ep_stale_") and "cupmem" not in ep_id.lower(), \
            f"Leakage failure: STALE/CUPMem domain info '{ep_id}' leaked into Silver v1 training!"

    print("[+] Stage C Silver v1 Sanity Checker: All checks passed successfully.")


def main() -> None:
    # 1. Load existing v0 dev expansion episodes (70 episodes)
    v0_episodes_with_gold = generate_expanded_episodes()
    print(f"Loaded {len(v0_episodes_with_gold)} base episodes from v0 dev_expansion.")

    # 2. Generate new v1 episodes (70 episodes)
    new_v1_episodes_with_gold = generate_new_silver_v1_episodes()
    print(f"Generated {len(new_v1_episodes_with_gold)} new base episodes for v1.")

    # Combined episodes list (140 base episodes total)
    episodes_with_gold = v0_episodes_with_gold + new_v1_episodes_with_gold
    print(f"Total base episodes: {len(episodes_with_gold)}")

    # Convert base episodes into labeled Stage C SFT examples
    labeled_examples: List[StageCTrainingExample] = []
    for ep, gold in episodes_with_gold:
        targets_by_submission: Dict[str, List[TypedRevisionTarget]] = defaultdict(list)
        for target in gold.gold_typed_targets:
            targets_by_submission[target.submission_id].append(target)

        # Retain submissions with revision labels only (do not inject default NO_REVISION here to preserve exact label counts)
        for submission in ep.submissions:
            targets = tuple(targets_by_submission.get(submission.submission_id, ()))
            if not targets:
                continue
            labeled_examples.append(make_example(ep, submission, targets))

    # Split train vs held-out validation/test sources
    train_source = [ex for ex in labeled_examples if not ex.episode_id.endswith("_v5")]
    heldout_source = [ex for ex in labeled_examples if ex.episode_id.endswith("_v5")]

    # Render training rows with surface augmentations
    train_rows: List[Dict[str, Any]] = []
    for ex in train_source:
        for aug_name, mapping in TRAIN_AUGMENTATIONS:
            train_rows.append(to_chat_row(ex, aug_name, mapping))

    # Split validation and test sets cleanly by episode_id to prevent leakages
    heldout_episode_ids = sorted(list({ex.episode_id for ex in heldout_source}))
    rng = random.Random(RNG_SEED)
    rng.shuffle(heldout_episode_ids)
    midpoint = max(1, len(heldout_episode_ids) // 2)
    valid_eps_set = set(heldout_episode_ids[:midpoint])
    test_eps_set = set(heldout_episode_ids[midpoint:])

    valid_source = [ex for ex in heldout_source if ex.episode_id in valid_eps_set]
    test_source = [ex for ex in heldout_source if ex.episode_id in test_eps_set]

    valid_rows = [to_chat_row(ex, "heldout_base", {}) for ex in valid_source]
    test_rows = [to_chat_row(ex, "heldout_base", {}) for ex in test_source]

    # Run strict sanity checker before writing
    run_sanity_checker(labeled_examples, episodes_with_gold, train_rows, valid_rows, test_rows)

    # Write files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_hash = write_jsonl(OUT_DIR / "train.jsonl", train_rows)
    valid_hash = write_jsonl(OUT_DIR / "valid.jsonl", valid_rows)
    test_hash = write_jsonl(OUT_DIR / "test.jsonl", test_rows)

    # Count statistics for manifest
    action_counts = Counter()
    failure_counts = Counter()
    multi_action_counts = 0
    for ex in labeled_examples:
        failure_counts[ex.failure_type] += 1
        for target in ex.targets:
            action_counts[target.action_type] += 1
        if len(ex.targets) > 1:
            multi_action_counts += 1

    manifest = {
        "dataset_name": "retrace_stagec_silver_v1",
        "scientific_status": "silver_synthetic_training_only",
        "warning": "silver_synthetic_training_only / not_for_paper_main_results",
        "generator_version": "silver_v1_generator_v1.0",
        "random_seed": RNG_SEED,
        "base_episode_count": len(episodes_with_gold),
        "revision_labeled_source_examples": len(labeled_examples),
        "train_source_examples_before_augmentation": len(train_source),
        "heldout_source_examples": len(heldout_source),
        "train_row_count": len(train_rows),
        "valid_row_count": len(valid_rows),
        "test_row_count": len(test_rows),
        "train_augmentation_count": len(TRAIN_AUGMENTATIONS),
        "action_counts_before_augmentation": dict(sorted(action_counts.items())),
        "multi_action_counts_before_augmentation": multi_action_counts,
        "failure_type_counts_before_augmentation": dict(sorted(failure_counts.items())),
        "hashes": {
            "train_sha256": train_hash,
            "valid_sha256": valid_hash,
            "test_sha256": test_hash,
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        },
        "limitations": [
            "Synthetic silver data has not been individually human-approved for paper claims.",
            "Useful for exploratory adapter training, not final reported paper evaluations."
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nManifest summary:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nWrote MLX training data to: {OUT_DIR}")
    print(f"Wrote manifest to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
