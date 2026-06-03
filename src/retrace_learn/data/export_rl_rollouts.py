"""Export ``dpa_rl_rollouts.jsonl`` (Stage 3 DPA-guided RSFT / DPO data).

For each episode we sample several candidate completions (the gold oracle plus
controlled perturbations), run each through parser + RevisionGate + DPA, and
score it with the DPA-in-the-loop reward. The result is a rollout row carrying
the reward breakdown, total reward, failure category, and audit trace — directly
consumable as DPO/GRPO preference data (gold rollout = chosen, perturbed = rejected).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retrace_learn.schemas import RLRolloutExample, RevisionAction
from retrace_learn.data.build_synthetic_raw_dialogue import SyntheticEpisode, build_synthetic_episodes
from retrace_learn.data.jsonl_io import write_jsonl
from retrace_learn.runtime.dpa_runtime import run_from_text
from retrace_learn.runtime.learned_proposer import actions_to_json, build_proposer_prompt
from retrace_learn.runtime.reward import compute_reward_for_view

DEFAULT_OUT = "outputs/retrace_learn/dpa_rl_rollouts.jsonl"


def _perturbations(ep: SyntheticEpisode) -> list[tuple[str, str]]:
    """Return (rollout_tag, raw_completion) candidates for an episode."""
    gold = list(ep.gold_actions)
    candidates: list[tuple[str, str]] = [("gold", actions_to_json(gold))]

    # Drop the last non-NO_REVISION action (e.g. RELEASES) -> under-update / stale.
    revisions = [a for a in gold if a.action_type != "NO_REVISION"]
    if len(revisions) >= 1:
        dropped = [a for a in gold if a is not revisions[-1]]
        candidates.append(("drop_last_revision", actions_to_json(dropped)))

    # Ungrounded target -> gate rejection / invalid target.
    bad_target = [
        RevisionAction(
            action_type="SUPERSEDES",
            target_belief_id="b_does_not_exist",
            replacement_belief_id="b_also_missing",
            evidence_ids=(ep.new_evidence_id,),
            rationale="invalid target perturbation",
        )
    ]
    candidates.append(("invalid_target", actions_to_json(bad_target)))

    # Malformed JSON -> parser error (fail-closed).
    candidates.append(("broken_json", "I think the answer is: not-json"))
    return candidates


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in build_synthetic_episodes():
        view = ep.build_view()
        prompt = build_proposer_prompt(view)
        gold_statuses = ep.gold_final_statuses()
        gold_actions = list(ep.gold_actions)
        for tag, completion in _perturbations(ep):
            result = run_from_text(view, completion)
            breakdown = compute_reward_for_view(
                view, result, gold_statuses, gold_actions=gold_actions
            )
            row = RLRolloutExample(
                example_id=f"{ep.example_id}_{tag}",
                prompt_input=prompt,
                sampled_actions=[a.to_dict() for a in result.parse_result.actions],
                parser_result=result.parse_result.to_dict(),
                gate_decisions=result.gate_decisions,
                dpa_final_statuses=result.final_belief_statuses,
                gold_final_statuses=gold_statuses,
                reward_breakdown=breakdown.reward_breakdown(),
                total_reward=breakdown.total_reward,
                failure_category=breakdown.failure_category,
                audit_trace=result.audit_trace,
                metadata={
                    "episode_id": ep.example_id,
                    "rollout_tag": tag,
                    "raw_completion": completion,
                },
            )
            row.validate()
            rows.append(row.to_dict())
    return rows


def build_preference_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rollouts by episode and emit (chosen, rejected) DPO pairs."""
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        base = row["metadata"]["episode_id"]
        by_episode.setdefault(base, []).append(row)
    pairs = []
    for base, group in by_episode.items():
        group_sorted = sorted(group, key=lambda r: r["total_reward"], reverse=True)
        chosen = group_sorted[0]
        for rejected in group_sorted[1:]:
            if rejected["total_reward"] < chosen["total_reward"]:
                pairs.append(
                    {
                        "example_id": base,
                        "prompt_input": chosen["prompt_input"],
                        "chosen": chosen["metadata"]["raw_completion"],
                        "rejected": rejected["metadata"]["raw_completion"],
                        "chosen_reward": chosen["total_reward"],
                        "rejected_reward": rejected["total_reward"],
                        "rejected_failure": rejected["failure_category"],
                    }
                )
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--pairs-out", default=None, help="optional DPO pairs JSONL path")
    args = parser.parse_args(argv)
    rows = build_rows()
    n = write_jsonl(Path(args.out), rows)
    print(f"wrote {n} dpa_rl_rollouts rows -> {args.out}")
    if args.pairs_out:
        pairs = build_preference_pairs(rows)
        m = write_jsonl(Path(args.pairs_out), pairs)
        print(f"wrote {m} DPO preference pairs -> {args.pairs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
