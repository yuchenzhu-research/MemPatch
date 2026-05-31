"""Stage SFT-2 eval: typed revision policy quality.

Metrics: valid_json, action_type_accuracy, target_grounding, evidence_grounding,
exact_action_match, and final_status_accuracy_after_DPA (the kernel-grounded
metric that actually matters). The policy is model-agnostic: pass a
``proposer_factory(example) -> proposer`` where ``proposer.propose(view)``
returns a :class:`ProposalOutput`. The default factory is the gold oracle, which
yields a perfect-score sanity check of the harness.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from retrace_learn.schemas import RevisionAction, TypedRevisionExample
from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.runtime.views import build_view
from retrace_learn.runtime.dpa_runtime import run_actions
from retrace_learn.runtime.learned_proposer import ScriptedProposer
from retrace_learn.eval.metrics import accuracy, mean


def view_from_example(ex: TypedRevisionExample):
    return build_view(
        instance_id=ex.metadata.get("instance_id", "instance_0"),
        query_id=ex.metadata.get("query_id", "query_0"),
        query=ex.metadata.get("query", "eval query"),
        evidence_context=ex.current_graph["evidence_nodes"],
        new_evidence_id=ex.new_evidence["evidence_id"],
        candidate_beliefs=ex.candidate_beliefs,
        candidate_replacement_beliefs=ex.candidate_replacement_beliefs,
        candidate_conditions_by_belief=ex.candidate_conditions_by_belief,
        dependency_edges_by_belief=ex.dependency_edges_by_belief,
    )


def _action_key(a: RevisionAction) -> tuple:
    return (
        a.action_type,
        a.target_belief_id,
        a.target_condition_id,
        a.replacement_belief_id,
    )


def _type_by_target(actions: list[RevisionAction]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in actions:
        target = a.target_belief_id or a.target_condition_id or "__none__"
        out[target] = a.action_type
    return out


def evaluate(
    examples: list[TypedRevisionExample],
    proposer_factory: Callable[[TypedRevisionExample], Any],
) -> dict[str, float]:
    per: dict[str, list[float]] = {
        "valid_json": [],
        "action_type_accuracy": [],
        "target_grounding": [],
        "evidence_grounding": [],
        "exact_action_match": [],
        "final_status_accuracy_after_DPA": [],
    }
    for ex in examples:
        view = view_from_example(ex)
        proposer = proposer_factory(ex)
        out = proposer.propose(view)
        per["valid_json"].append(1.0 if out.parsing_valid else 0.0)

        pred = list(out.actions)
        gold = ex.gold_action_objects()

        # action_type accuracy keyed by target id
        gold_types = _type_by_target(gold)
        pred_types = _type_by_target(pred)
        correct = sum(1 for t, at in gold_types.items() if pred_types.get(t) == at)
        per["action_type_accuracy"].append(accuracy(correct, len(gold_types)))

        # grounding over predicted revision actions
        belief_ids = {b["belief_id"] for b in ex.candidate_beliefs}
        belief_ids |= {b["belief_id"] for b in ex.candidate_replacement_beliefs}
        cond_ids = {
            c["condition_id"]
            for conds in ex.candidate_conditions_by_belief.values()
            for c in conds
        }
        ev_ids = {ex.new_evidence["evidence_id"]}
        ev_ids |= {e["evidence_id"] for e in ex.current_graph["evidence_nodes"]}
        revs = [a for a in pred if a.action_type != "NO_REVISION"]
        tg = sum(
            1
            for a in revs
            if (a.target_belief_id in belief_ids or a.target_belief_id is None)
            and (a.target_condition_id in cond_ids or a.target_condition_id is None)
            and (a.replacement_belief_id in belief_ids or a.replacement_belief_id is None)
        )
        eg = sum(1 for a in revs if a.evidence_ids and all(e in ev_ids for e in a.evidence_ids))
        per["target_grounding"].append(accuracy(tg, len(revs)))
        per["evidence_grounding"].append(accuracy(eg, len(revs)))

        per["exact_action_match"].append(
            1.0 if {_action_key(a) for a in pred} == {_action_key(a) for a in gold} else 0.0
        )

        result = run_actions(view, pred)
        gold_status = ex.gold_final_statuses
        matched = sum(
            1
            for bid, st in gold_status.items()
            if result.final_belief_statuses.get(bid) == st
        )
        per["final_status_accuracy_after_DPA"].append(
            accuracy(matched, len(gold_status))
        )
    return {k: mean(v) for k, v in per.items()}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    examples = [ep.to_typed_revision_example() for ep in build_synthetic_episodes()]
    report = evaluate(
        examples,
        proposer_factory=lambda ex: ScriptedProposer(ex.gold_action_objects()),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
