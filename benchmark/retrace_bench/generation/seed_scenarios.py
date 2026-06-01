from typing import Any, Dict, List

# A list of blueprint dictionaries for seed scenarios across different domains and revision families.
SEED_TEMPLATES: List[Dict[str, Any]] = [
    {
        "domain": "coding_agent_debugging",
        "context": "A group of software agents are debugging a production database connection pool leak.",
        "base_beliefs": [
            {"id": "b1", "proposition": "The database connection leak is caused by unclosed connections in the user service."}
        ],
        "base_conditions": [
            {"id": "c1", "scope_id": "global", "text": "The user service is deployed with connection pool tracking enabled."}
        ],
        "topology": {
            "requires": {"b1": ["c1"]}
        },
        "families": {
            "supersedes": {
                "conflict_type": "obsolete_root_cause",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "I analyzed the connection pool logs. User service connection leakage is actually not the root cause."},
                    {"speaker": "subagent_B", "text": "Correct. We found that the leak is located in the payment gateway notification handler."}
                ],
                "new_entries": [
                    {"id": "e1", "type": "evidence", "content": "Payment gateway notification handler leaks connections, verified by telemetry logs."},
                    {"id": "b2", "type": "belief", "content": "The database connection leak is caused by unclosed connections in the payment gateway notification handler."}
                ],
                "gold_actions": [
                    {
                        "action_type": "SUPERSEDES",
                        "target_id": "b1",
                        "replacement_id": "b2",
                        "evidence_ids": ["e1"],
                        "rationale": "Payment gateway connection leak replaces the user service leak theory."
                    }
                ],
                "final_statuses": {
                    "b1": "SUPERSEDED",
                    "b2": "AUTHORIZED"
                },
                "evidence_trigger_id": "e1"
            },
            "blocks": {
                "conflict_type": "prerequisite_failed",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "Wait, connection pool tracking has been disabled in the user service config to save CPU overhead."},
                    {"speaker": "subagent_B", "text": "Yes, I confirmed connection pool tracking is now disabled."}
                ],
                "new_entries": [
                    {"id": "e2", "type": "evidence", "content": "Connection pool tracking is disabled in user service config."}
                ],
                "gold_actions": [
                    {
                        "action_type": "BLOCKS",
                        "target_id": "c1",
                        "evidence_ids": ["e2"],
                        "rationale": "Connection pool tracking is disabled, so we cannot verify the connection pool leak theory."
                    }
                ],
                "final_statuses": {
                    "b1": "BLOCKED"
                },
                "evidence_trigger_id": "e2"
            },
            "releases": {
                "conflict_type": "prerequisite_restored",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "We re-enabled pool tracking in user service on the staging branch."},
                    {"speaker": "subagent_B", "text": "Ok, the prerequisite is active again."}
                ],
                "new_entries": [
                    {"id": "e3", "type": "evidence", "content": "Pool tracking is re-enabled in staging."}
                ],
                "gold_actions": [
                    {
                        "action_type": "RELEASES",
                        "target_id": "c1",
                        "evidence_ids": ["e3"],
                        "rationale": "Re-enabling pool tracking restores eligibility."
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e3"
            },
            "uncertain": {
                "conflict_type": "partial_refutation",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "There are some reports that user service uses auto-commit, which might not leak."},
                    {"speaker": "subagent_B", "text": "This is unconfirmed, we are not sure yet."}
                ],
                "new_entries": [
                    {"id": "e4", "type": "evidence", "content": "Auto-commit reporting suggests potential lack of leak."}
                ],
                "gold_actions": [
                    {
                        "action_type": "UNCERTAIN",
                        "target_id": "b1",
                        "evidence_ids": ["e4"],
                        "rationale": "Potential auto-commit behavior makes the leak theory uncertain."
                    }
                ],
                "final_statuses": {
                    "b1": "UNRESOLVED"
                },
                "evidence_trigger_id": "e4"
            },
            "reaffirms": {
                "conflict_type": "confirm_validity",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "I re-ran the local simulator and user service leaked 50 pools in 5 minutes."},
                    {"speaker": "subagent_B", "text": "Awesome, the user service leak theory is strongly confirmed."}
                ],
                "new_entries": [
                    {"id": "e5", "type": "evidence", "content": "Local simulator confirmed user service leak of 50 connections."}
                ],
                "gold_actions": [
                    {
                        "action_type": "REAFFIRMS",
                        "target_id": "b1",
                        "evidence_ids": ["e5"],
                        "rationale": "Simulation reaffirms the user service leak."
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e5"
            },
            "no_revision": {
                "conflict_type": "irrelevant_update",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "By the way, the UI color scheme has been changed to dark blue."},
                    {"speaker": "subagent_B", "text": "Okay, that is completely unrelated to the DB leak."}
                ],
                "new_entries": [
                    {"id": "e6", "type": "evidence", "content": "UI color changed to dark blue."}
                ],
                "gold_actions": [
                    {
                        "action_type": "NO_REVISION",
                        "target_id": "b1",
                        "evidence_ids": ["e6"],
                        "rationale": "Unrelated UI update."
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e6"
            },
            "mixed_multi_action": {
                "conflict_type": "complex_reconfiguration",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "We realized the user service config tracking is disabled, and also the leak is actually in payment gateway."},
                    {"speaker": "subagent_B", "text": "Let's block the pool tracking condition and supersede the leak theory."}
                ],
                "new_entries": [
                    {"id": "e7", "type": "evidence", "content": "Telemetry shows user service tracking disabled and leak is in payment gateway."},
                    {"id": "b3", "type": "belief", "content": "Database leak is in payment gateway."}
                ],
                "gold_actions": [
                    {
                        "action_type": "BLOCKS",
                        "target_id": "c1",
                        "evidence_ids": ["e7"],
                        "rationale": "Disable user service pool tracking."
                    },
                    {
                        "action_type": "SUPERSEDES",
                        "target_id": "b1",
                        "replacement_id": "b3",
                        "evidence_ids": ["e7"],
                        "rationale": "Replaced by payment gateway."
                    }
                ],
                "final_statuses": {
                    "b1": "SUPERSEDED",
                    "b3": "AUTHORIZED"
                },
                "evidence_trigger_id": "e7"
            }
        }
    },
    {
        "domain": "research_agent_memory",
        "context": "Agents conducting a systematic literature review on room-temperature superconductivity.",
        "base_beliefs": [
            {"id": "b1", "proposition": "LK-99 exhibits zero electrical resistance at ambient pressure below 400 Kelvin."}
        ],
        "base_conditions": [
            {"id": "c1", "scope_id": "global", "text": "The replication laboratory environment maintains a pure sample composition of copper-doped lead apatite."}
        ],
        "topology": {
            "requires": {"b1": ["c1"]}
        },
        "families": {
            "supersedes": {
                "conflict_type": "scientific_refutation",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "Recent multi-institute study confirmed LK-99 is actually a semiconductor and zero resistance was a fluke."},
                    {"speaker": "subagent_B", "text": "Yes, they synthesized pure crystals showing it is an insulator at low temperatures."}
                ],
                "new_entries": [
                    {"id": "e1", "type": "evidence", "content": "Nature report confirming LK-99 is a ferromagnetic insulator in pure form."},
                    {"id": "b2", "type": "belief", "content": "LK-99 is an insulator with ferromagnetic properties and does not exhibit superconductivity."}
                ],
                "gold_actions": [
                    {
                        "action_type": "SUPERSEDES",
                        "target_id": "b1",
                        "replacement_id": "b2",
                        "evidence_ids": ["e1"]
                    }
                ],
                "final_statuses": {
                    "b1": "SUPERSEDED",
                    "b2": "AUTHORIZED"
                },
                "evidence_trigger_id": "e1"
            },
            "blocks": {
                "conflict_type": "experimental_flaw",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "The sample we analyzed has high levels of copper sulfide impurities."},
                    {"speaker": "subagent_B", "text": "Ah, so the pure sample condition c1 is violated."}
                ],
                "new_entries": [
                    {"id": "e2", "type": "evidence", "content": "XRD analysis shows significant copper sulfide (Cu2S) impurity phase."}
                ],
                "gold_actions": [
                    {
                        "action_type": "BLOCKS",
                        "target_id": "c1",
                        "evidence_ids": ["e2"]
                    }
                ],
                "final_statuses": {
                    "b1": "BLOCKED"
                },
                "evidence_trigger_id": "e2"
            },
            "releases": {
                "conflict_type": "experiment_restored",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "We successfully synthesized a pure apatite crystal without Cu2S impurities."},
                    {"speaker": "subagent_B", "text": "Excellent, the pure sample prerequisite is now met again."}
                ],
                "new_entries": [
                    {"id": "e3", "type": "evidence", "content": "Pure copper-doped lead apatite crystal synthesis validation."}
                ],
                "gold_actions": [
                    {
                        "action_type": "RELEASES",
                        "target_id": "c1",
                        "evidence_ids": ["e3"]
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e3"
            },
            "uncertain": {
                "conflict_type": "peer_review_conflict",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "A lab in Berlin reported negative resistivity, but their probe has noise calibration issues."},
                    {"speaker": "subagent_B", "text": "We must treat the zero-resistance claim as uncertain for now."}
                ],
                "new_entries": [
                    {"id": "e4", "type": "evidence", "content": "Berlin lab reports noise issues in electrical transport measurement."}
                ],
                "gold_actions": [
                    {
                        "action_type": "UNCERTAIN",
                        "target_id": "b1",
                        "evidence_ids": ["e4"]
                    }
                ],
                "final_statuses": {
                    "b1": "UNRESOLVED"
                },
                "evidence_trigger_id": "e4"
            },
            "reaffirms": {
                "conflict_type": "reaffirm_peer_review",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "The independent evaluation committee reviewed the raw data and found it completely solid."},
                    {"speaker": "subagent_B", "text": "That strengthens our initial belief b1."}
                ],
                "new_entries": [
                    {"id": "e5", "type": "evidence", "content": "Evaluation committee formal data audit report."}
                ],
                "gold_actions": [
                    {
                        "action_type": "REAFFIRMS",
                        "target_id": "b1",
                        "evidence_ids": ["e5"]
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e5"
            },
            "no_revision": {
                "conflict_type": "unrelated_lit",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "I found a paper discussing room-temperature lithium batteries."},
                    {"speaker": "subagent_B", "text": "Okay, interesting but not relevant to LK-99 superconductivity."}
                ],
                "new_entries": [
                    {"id": "e6", "type": "evidence", "content": "Lithium battery ambient temperature literature."}
                ],
                "gold_actions": [
                    {
                        "action_type": "NO_REVISION",
                        "target_id": "b1",
                        "evidence_ids": ["e6"]
                    }
                ],
                "final_statuses": {
                    "b1": "AUTHORIZED"
                },
                "evidence_trigger_id": "e6"
            },
            "mixed_multi_action": {
                "conflict_type": "scientific_refute_and_violate",
                "dialogue": [
                    {"speaker": "subagent_A", "text": "The sample has severe impurities, and new calculations show zero resistance is mathematically impossible."},
                    {"speaker": "subagent_B", "text": "Let's block the purity condition and replace the superconductivity claim with the insulator conclusion."}
                ],
                "new_entries": [
                    {"id": "e7", "type": "evidence", "content": "DFT calculation proving insulator state and impurity analysis."},
                    {"id": "b3", "type": "belief", "content": "LK-99 is an insulator with no zero resistance pathway."}
                ],
                "gold_actions": [
                    {
                        "action_type": "BLOCKS",
                        "target_id": "c1",
                        "evidence_ids": ["e7"]
                    },
                    {
                        "action_type": "SUPERSEDES",
                        "target_id": "b1",
                        "replacement_id": "b3",
                        "evidence_ids": ["e7"]
                    }
                ],
                "final_statuses": {
                    "b1": "SUPERSEDED",
                    "b3": "AUTHORIZED"
                },
                "evidence_trigger_id": "e7"
            }
        }
    }
]

# We can dynamically duplicate/tweak these templates to build 100/2500 scenarios
# by replacing terms (e.g. subagent names, minor detail strings, identifiers) deterministically.
