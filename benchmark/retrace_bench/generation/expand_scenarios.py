import random
from typing import List, Dict, Any
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, RevisionActionType, FinalStatus
from benchmark.retrace_bench.schemas import Scenario, DialogueTurn, MemoryEntry, RevisionAction
from benchmark.retrace_bench.generation.seed_scenarios import SEED_TEMPLATES
from benchmark.retrace_bench.generation.render_queries import render_scenario_queries


DOMAINS = [
    Domain.CODING_AGENT_DEBUGGING,
    Domain.RESEARCH_AGENT_MEMORY,
    Domain.PERSONAL_PREFERENCE_MEMORY,
    Domain.CALENDAR_WORKFLOW,
    Domain.TOOL_USE_ASSISTANT,
    Domain.MULTI_AGENT_KNOWLEDGE_BASE
]

FAMILIES = [
    RevisionFamily.SUPERSEDES,
    RevisionFamily.BLOCKS,
    RevisionFamily.RELEASES,
    RevisionFamily.UNCERTAIN,
    RevisionFamily.REAFFIRMS,
    RevisionFamily.NO_REVISION,
    RevisionFamily.MIXED_MULTI_ACTION
]


def expand_templates(num_scenarios: int, seed: int = 7) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios = []

    # Map domains to their specific terminology to generate diverse scenario texts dynamically
    domain_configs = {
        Domain.CODING_AGENT_DEBUGGING: {
            "belief_prop": "The database connection leak is caused by unclosed connections in the user service.",
            "cond_text": "The user service is deployed with connection pool tracking enabled.",
            "supersede_prop": "The database connection leak is caused by unclosed connections in the payment gateway notification handler.",
            "supersede_evidence": "Payment gateway notification handler leaks connections, verified by telemetry logs.",
            "block_evidence": "Connection pool tracking is disabled in user service config.",
            "release_evidence": "Pool tracking is re-enabled in staging.",
            "uncertain_evidence": "Auto-commit reporting suggests potential lack of leak.",
            "reaffirm_evidence": "Local simulator confirmed user service leak of 50 connections.",
            "no_rev_evidence": "UI color changed to dark blue.",
            "mixed_prop": "Database leak is in payment gateway.",
            "mixed_evidence": "Telemetry shows user service tracking disabled and leak is in payment gateway.",
        },
        Domain.RESEARCH_AGENT_MEMORY: {
            "belief_prop": "LK-99 exhibits zero electrical resistance at ambient pressure below 400 Kelvin.",
            "cond_text": "The replication laboratory environment maintains a pure sample composition of copper-doped lead apatite.",
            "supersede_prop": "LK-99 is an insulator with ferromagnetic properties and does not exhibit superconductivity.",
            "supersede_evidence": "Nature report confirming LK-99 is a ferromagnetic insulator in pure form.",
            "block_evidence": "XRD analysis shows significant copper sulfide (Cu2S) impurity phase.",
            "release_evidence": "Pure copper-doped lead apatite crystal synthesis validation.",
            "uncertain_evidence": "Berlin lab reports noise issues in electrical transport measurement.",
            "reaffirm_evidence": "Evaluation committee formal data audit report.",
            "no_rev_evidence": "Lithium battery ambient temperature literature.",
            "mixed_prop": "LK-99 is an insulator with no zero resistance pathway.",
            "mixed_evidence": "DFT calculation proving insulator state and impurity analysis.",
        },
        Domain.PERSONAL_PREFERENCE_MEMORY: {
            "belief_prop": "The user prefers meeting in the morning before 10 AM.",
            "cond_text": "The user is in the EST timezone.",
            "supersede_prop": "The user prefers meeting in the afternoon after 2 PM.",
            "supersede_evidence": "Direct calendar survey showing user productivity spikes in afternoon.",
            "block_evidence": "User relocated to GMT timezone indefinitely.",
            "release_evidence": "User returned to EST timezone.",
            "uncertain_evidence": "User mentioned liking morning coffee, but not necessarily morning meetings.",
            "reaffirm_evidence": "User accepted three consecutive morning meeting invites.",
            "no_rev_evidence": "User preferred laptop model is MacBook Pro.",
            "mixed_prop": "User prefers afternoon meetings.",
            "mixed_evidence": "User relocated to GMT and rescheduled all meetings to afternoon.",
        },
        Domain.CALENDAR_WORKFLOW: {
            "belief_prop": "The weekly sync is scheduled on Wednesday at 2 PM.",
            "cond_text": "The conference room is reserved.",
            "supersede_prop": "The weekly sync is scheduled on Thursday at 11 AM.",
            "supersede_evidence": "Organizer rescheduled sync via email notification due to conflicts.",
            "block_evidence": "Conference room booking was canceled for maintenance.",
            "release_evidence": "Conference room maintenance completed and reservation re-activated.",
            "uncertain_evidence": "A tentative invite was sent for Thursday, but not confirmed.",
            "reaffirm_evidence": "Meeting room manager verified the Wednesday reservation.",
            "no_rev_evidence": "Lunch menu for the team meeting is selected.",
            "mixed_prop": "Weekly sync is scheduled on Thursday.",
            "mixed_evidence": "Room maintenance scheduled and sync moved to Thursday.",
        },
        Domain.TOOL_USE_ASSISTANT: {
            "belief_prop": "The agent must call calculate_tax tool with version v2.",
            "cond_text": "The tax database is updated for the current fiscal year.",
            "supersede_prop": "The agent must call calculate_tax tool with version v3.",
            "supersede_evidence": "API gateway doc updated deprecating v2 in favor of v3.",
            "block_evidence": "Tax database updates for current year are delayed by IRS.",
            "release_evidence": "IRS released the tax updates database schema.",
            "uncertain_evidence": "Vague slack message saying v2 might be slow.",
            "reaffirm_evidence": "Integration test verified v2 output matches standards.",
            "no_rev_evidence": "The logging level is set to DEBUG.",
            "mixed_prop": "Agent must call calculate_tax with v3.",
            "mixed_evidence": "Tax database delayed and API docs recommend v3.",
        },
        Domain.MULTI_AGENT_KNOWLEDGE_BASE: {
            "belief_prop": "The master branch code deployment is scheduled for Friday.",
            "cond_text": "All CI pipelines pass successfully.",
            "supersede_prop": "The master branch code deployment is scheduled for next Tuesday.",
            "supersede_evidence": "Release manager postponed release due to pending critical CVE fixes.",
            "block_evidence": "Linting pipeline broke because of node package dependency error.",
            "release_evidence": "Package lock file updated, fixing the CI pipeline.",
            "uncertain_evidence": "Developer mentioned test failures but didn't push commits.",
            "reaffirm_evidence": "Deployment board status moved to APPROVED by QA lead.",
            "no_rev_evidence": "Documentation format updated to sphinx standard.",
            "mixed_prop": "Code deployment is rescheduled to next Tuesday.",
            "mixed_evidence": "CI pipeline failed and release manager postponed release to next Tuesday.",
        }
    }

    speakers = ["subagent_A", "subagent_B", "agent_X", "agent_Y", "moderator"]

    for i in range(num_scenarios):
        # Deterministically select domain, family, and specific names
        domain = DOMAINS[i % len(DOMAINS)]
        family = FAMILIES[(i // len(DOMAINS)) % len(FAMILIES)]
        cfg = domain_configs[domain]

        # Suffix to ensure uniqueness across scenarios
        suffix = f"_{i}"
        b1_id = f"b1{suffix}"
        c1_id = f"c1{suffix}"
        b2_id = f"b2{suffix}"
        b3_id = f"b3{suffix}"
        e_trigger_id = f"e_trig{suffix}"

        # Generate dialogue speakers
        sp1 = rng.choice(speakers)
        sp2 = rng.choice([s for s in speakers if s != sp1])

        # Base snapshot
        memory_snapshot = [
            MemoryEntry(entry_id=b1_id, content=cfg["belief_prop"], entry_type="belief"),
            MemoryEntry(entry_id=c1_id, content=cfg["cond_text"], entry_type="condition")
        ]

        dialogue_history = []
        gold_actions = []
        gold_final_statuses = {}
        conflict_type = ""

        if family == RevisionFamily.SUPERSEDES:
            conflict_type = "supersede_revision"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"Actually, {cfg['supersede_evidence']}"),
                DialogueTurn(speaker=sp2, text="I will record the change in memory.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["supersede_evidence"], entry_type="evidence"))
            memory_snapshot.append(MemoryEntry(entry_id=b2_id, content=cfg["supersede_prop"], entry_type="belief"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.SUPERSEDES,
                    target_id=b1_id,
                    replacement_id=b2_id,
                    evidence_ids=[e_trigger_id],
                    rationale="New evidence supersedes previous belief."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.SUPERSEDED,
                b2_id: FinalStatus.AUTHORIZED
            }

        elif family == RevisionFamily.BLOCKS:
            conflict_type = "prerequisite_violated"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"Wait, {cfg['block_evidence']}"),
                DialogueTurn(speaker=sp2, text="This blocks our prerequisite condition.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["block_evidence"], entry_type="evidence"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.BLOCKS,
                    target_id=c1_id,
                    evidence_ids=[e_trigger_id],
                    rationale="Prerequisite condition is blocked by evidence."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.BLOCKED
            }

        elif family == RevisionFamily.RELEASES:
            conflict_type = "prerequisite_released"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"Good news, {cfg['release_evidence']}"),
                DialogueTurn(speaker=sp2, text="This releases the prerequisite block.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["release_evidence"], entry_type="evidence"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.RELEASES,
                    target_id=c1_id,
                    evidence_ids=[e_trigger_id],
                    rationale="Prerequisite condition is released."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.AUTHORIZED
            }

        elif family == RevisionFamily.UNCERTAIN:
            conflict_type = "uncertain_validity"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"I am not sure, {cfg['uncertain_evidence']}"),
                DialogueTurn(speaker=sp2, text="Let's flag the belief as uncertain.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["uncertain_evidence"], entry_type="evidence"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.UNCERTAIN,
                    target_id=b1_id,
                    evidence_ids=[e_trigger_id],
                    rationale="Status is uncertain due to conflicting input."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.UNRESOLVED
            }

        elif family == RevisionFamily.REAFFIRMS:
            conflict_type = "reaffirm_validity"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"I verified that {cfg['reaffirm_evidence']}"),
                DialogueTurn(speaker=sp2, text="This reaffirms our initial belief.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["reaffirm_evidence"], entry_type="evidence"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.REAFFIRMS,
                    target_id=b1_id,
                    evidence_ids=[e_trigger_id],
                    rationale="Evidence reaffirms the belief."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.AUTHORIZED
            }

        elif family == RevisionFamily.NO_REVISION:
            conflict_type = "unrelated_dialogue"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"Note: {cfg['no_rev_evidence']}"),
                DialogueTurn(speaker=sp2, text="Unrelated change, no revision required.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["no_rev_evidence"], entry_type="evidence"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.NO_REVISION,
                    target_id=b1_id,
                    evidence_ids=[e_trigger_id],
                    rationale="No revision needed."
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.AUTHORIZED
            }

        elif family == RevisionFamily.MIXED_MULTI_ACTION:
            conflict_type = "complex_multiple_actions"
            dialogue_history = [
                DialogueTurn(speaker=sp1, text=f"Update: {cfg['mixed_evidence']}"),
                DialogueTurn(speaker=sp2, text="We need to block the old condition and update the belief.")
            ]
            memory_snapshot.append(MemoryEntry(entry_id=e_trigger_id, content=cfg["mixed_evidence"], entry_type="evidence"))
            memory_snapshot.append(MemoryEntry(entry_id=b3_id, content=cfg["mixed_prop"], entry_type="belief"))
            gold_actions = [
                RevisionAction(
                    action_type=RevisionActionType.BLOCKS,
                    target_id=c1_id,
                    evidence_ids=[e_trigger_id]
                ),
                RevisionAction(
                    action_type=RevisionActionType.SUPERSEDES,
                    target_id=b1_id,
                    replacement_id=b3_id,
                    evidence_ids=[e_trigger_id]
                )
            ]
            gold_final_statuses = {
                b1_id: FinalStatus.SUPERSEDED,
                b3_id: FinalStatus.AUTHORIZED
            }

        requires_map = {b1_id: [c1_id]}
        memory_topology = {"requires": requires_map}

        # Render 4 Probe Queries
        probe_queries = render_scenario_queries(
            scenario_id=f"scen_{i}",
            domain=domain.value,
            family=family.value,
            memory_snapshot=memory_snapshot,
            gold_final_statuses=gold_final_statuses,
            evidence_trigger_id=e_trigger_id,
            requires_map=requires_map
        )

        scenarios.append(Scenario(
            scenario_id=f"scen_{i}",
            domain=domain,
            revision_family=family,
            conflict_type=conflict_type,
            memory_topology=memory_topology,
            dialogue_history=dialogue_history,
            memory_snapshot=memory_snapshot,
            gold_final_statuses=gold_final_statuses,
            gold_revision_actions=gold_actions,
            probe_queries=probe_queries,
            metadata={"index": i, "suffix": suffix}
        ))

    return scenarios
