"""LoRA SFT for the graph extractor (Stage SFT-1) and revision proposer (SFT-2).

The dataset builders are pure-stdlib and runnable (``--dry-run``). ``train()``
lazily imports transformers/peft/trl and is a thin skeleton: it is intentionally
left to wire up a concrete ``SFTTrainer`` once a GPU/runtime is available, so
this module never forces heavy deps at import time.

Loss: standard JSON-target cross-entropy on the completion (prompt is masked).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.runtime.graph_extractor import build_extraction_prompt
from retrace_learn.runtime.learned_proposer import actions_to_json, build_proposer_prompt
from retrace_learn.eval.eval_revision_policy import view_from_example


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_graph_sft_pairs() -> list[dict[str, str]]:
    """(prompt, completion) pairs for Stage SFT-1 graph extraction."""
    pairs = []
    for ep in build_synthetic_episodes():
        ex = ep.to_graph_extraction_example()
        pairs.append(
            {
                "prompt": build_extraction_prompt(ex.raw_dialogue, ex.subagent_roles),
                "completion": json.dumps(ex.output_graph, ensure_ascii=False),
            }
        )
    return pairs


def build_revision_sft_pairs() -> list[dict[str, str]]:
    """(prompt, completion) pairs for Stage SFT-2 typed revision."""
    pairs = []
    for ep in build_synthetic_episodes():
        ex = ep.to_typed_revision_example()
        view = view_from_example(ex)
        pairs.append(
            {
                "prompt": build_proposer_prompt(view),
                "completion": actions_to_json(ex.gold_action_objects()),
            }
        )
    return pairs


def build_sft_pairs(task: str) -> list[dict[str, str]]:
    if task == "graph":
        return build_graph_sft_pairs()
    if task == "revision":
        return build_revision_sft_pairs()
    raise ValueError(f"unknown SFT task '{task}' (expected 'graph' or 'revision')")


def train(config_path: str | Path) -> None:  # pragma: no cover - requires GPU/deps
    config = load_config(config_path)
    task = config.get("train", {}).get("task", "revision")
    pairs = build_sft_pairs(task)
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "train() needs torch+transformers+peft+trl installed. Use --dry-run "
            "to build/inspect the dataset without these dependencies."
        ) from exc

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"])
    dataset = Dataset.from_list(
        [{"text": p["prompt"] + p["completion"]} for p in pairs]
    )
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=train_cfg["output_dir"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            num_train_epochs=train_cfg["num_train_epochs"],
            seed=train_cfg.get("seed", 17),
        ),
    )
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["graph", "revision"], default="revision")
    parser.add_argument("--dry-run", action="store_true", help="build dataset only")
    args = parser.parse_args(argv)

    if args.dry_run or args.config is None:
        pairs = build_sft_pairs(args.task)
        print(f"[dry-run] task={args.task} built {len(pairs)} SFT pairs")
        if pairs:
            print(f"  sample prompt chars: {len(pairs[0]['prompt'])}")
            print(f"  sample completion: {pairs[0]['completion'][:160]}")
        return 0
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
