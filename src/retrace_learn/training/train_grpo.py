"""GRPO with the DPA-in-the-loop reward (online RL variant; future/optional).

Not part of the v1 three-stage method. v1 stage 3 is DPA-guided RSFT/DPO built
offline from exported rollouts; this online GRPO loop is a future extension.

GRPO samples multiple completions per prompt and ranks them by a reward function.
Here the reward function *is* the deterministic pipeline: each sampled completion
is run through parser + RevisionGate + DPA and scored against gold final statuses
via :func:`retrace_learn.runtime.reward.compute_reward_for_view`. This is the key
ReTrace-Learn property — the policy is optimized for DPA-correct, memory-safe
outcomes, not surface action mimicry.

The prompt dataset + reward function are runnable (``--dry-run``); ``train()``
lazily imports trl's GRPOTrainer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.runtime.dpa_runtime import run_from_text
from retrace_learn.runtime.learned_proposer import build_proposer_prompt
from retrace_learn.runtime.reward import compute_reward_for_view


def load_config(path: str | Path) -> dict[str, Any]:
    from retrace_learn.training import check_contamination
    check_contamination(path)
    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    check_contamination(config)
    return config


def _registry() -> dict[str, dict[str, Any]]:
    """Map example_id -> {view, gold_statuses, gold_actions, prompt}."""
    reg: dict[str, dict[str, Any]] = {}
    for ep in build_synthetic_episodes():
        view = ep.build_view()
        reg[ep.example_id] = {
            "view": view,
            "gold_statuses": ep.gold_final_statuses(),
            "gold_actions": list(ep.gold_actions),
            "prompt": build_proposer_prompt(view),
        }
    return reg


_REGISTRY = _registry()


def build_grpo_dataset() -> list[dict[str, str]]:
    return [
        {"example_id": eid, "prompt": entry["prompt"]}
        for eid, entry in _REGISTRY.items()
    ]


def score_completion(example_id: str, completion: str) -> float:
    """DPA-in-the-loop reward for one sampled completion (the GRPO reward signal)."""
    entry = _REGISTRY[example_id]
    view = entry["view"]
    result = run_from_text(view, completion)
    breakdown = compute_reward_for_view(
        view,
        result,
        entry["gold_statuses"],
        gold_actions=entry["gold_actions"],
    )
    return breakdown.total_reward


def make_reward_fn():
    """Return a trl-compatible reward function: (prompts, completions, example_id)."""

    def reward_fn(prompts, completions, example_id=None, **kwargs):  # noqa: ANN001
        ids = example_id or [None] * len(completions)
        return [score_completion(eid, comp) for eid, comp in zip(ids, completions)]

    return reward_fn


def train(config_path: str | Path) -> None:  # pragma: no cover - requires GPU/deps
    config = load_config(config_path)
    dataset_rows = build_grpo_dataset()
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "train() needs transformers+trl installed. Use --dry-run to inspect rewards."
        ) from exc

    model_cfg = config["model"]
    train_cfg = config["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"])
    dataset = Dataset.from_list(dataset_rows)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(),
        args=GRPOConfig(
            output_dir=train_cfg["output_dir"],
            learning_rate=train_cfg.get("learning_rate", 1.0e-6),
            num_generations=train_cfg.get("num_generations", 4),
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
        ),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run or args.config is None:
        rows = build_grpo_dataset()
        reward_fn = make_reward_fn()
        print(f"[dry-run] built {len(rows)} GRPO prompts")
        for row in rows:
            entry = _REGISTRY[row["example_id"]]
            from retrace_learn.runtime.learned_proposer import actions_to_json

            gold_completion = actions_to_json(entry["gold_actions"])
            r = reward_fn([row["prompt"]], [gold_completion], example_id=[row["example_id"]])
            print(f"  {row['example_id']:16s} gold-completion reward={r[0]:+.3f}")
        return 0
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
