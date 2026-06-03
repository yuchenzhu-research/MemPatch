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
from retrace_learn.data.jsonl_io import read_jsonl
from retrace_learn.runtime.graph_extractor import build_extraction_prompt
from retrace_learn.runtime.learned_proposer import actions_to_json, build_proposer_prompt
from retrace_learn.eval.eval_revision_policy import view_from_example
from retrace_learn.schemas import GraphExtractionExample, TypedRevisionExample


def load_config(path: str | Path) -> dict[str, Any]:
    from retrace_learn.training import check_contamination
    check_contamination(path)
    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    check_contamination(config)
    return config


def _graph_pair(ex: GraphExtractionExample) -> dict[str, str]:
    return {
        "prompt": build_extraction_prompt(ex.raw_dialogue, ex.subagent_roles),
        "completion": json.dumps(ex.output_graph, ensure_ascii=False),
    }


def _revision_pair(ex: TypedRevisionExample) -> dict[str, str]:
    view = view_from_example(ex)
    return {
        "prompt": build_proposer_prompt(view),
        "completion": actions_to_json(ex.gold_action_objects()),
    }


def build_graph_sft_pairs(
    examples: list[GraphExtractionExample] | None = None,
) -> list[dict[str, str]]:
    """(prompt, completion) pairs for Stage SFT-1 graph extraction.

    With no argument, falls back to the *smoke* synthetic episodes (sanity only).
    """
    if examples is None:
        examples = [ep.to_graph_extraction_example() for ep in build_synthetic_episodes()]
    return [_graph_pair(ex) for ex in examples]


def build_revision_sft_pairs(
    examples: list[TypedRevisionExample] | None = None,
) -> list[dict[str, str]]:
    """(prompt, completion) pairs for Stage SFT-2 typed revision.

    With no argument, falls back to the *smoke* synthetic episodes (sanity only).
    """
    if examples is None:
        examples = [ep.to_typed_revision_example() for ep in build_synthetic_episodes()]
    return [_revision_pair(ex) for ex in examples]


def build_sft_pairs(task: str) -> list[dict[str, str]]:
    """Build SFT pairs from the smoke synthetic episodes (sanity/dry-run only)."""
    if task == "graph":
        return build_graph_sft_pairs()
    if task == "revision":
        return build_revision_sft_pairs()
    raise ValueError(f"unknown SFT task '{task}' (expected 'graph' or 'revision')")


def load_sft_pairs(task: str, input_path: str | Path) -> list[dict[str, str]]:
    """Build SFT pairs from explicit JSONL rows (real training input).

    ``graph`` rows are :class:`GraphExtractionExample`; ``revision`` rows are
    :class:`TypedRevisionExample`. Each row is validated before use. ReTrace-Bench
    paths are rejected by :func:`check_contamination`.
    """
    from retrace_learn.training import check_contamination
    check_contamination(input_path)
    rows = list(read_jsonl(input_path))
    if not rows:
        raise RuntimeError(f"no SFT rows found in '{input_path}'")
    if task == "graph":
        examples = [GraphExtractionExample.from_dict(r) for r in rows]
        for ex in examples:
            ex.validate()
        return build_graph_sft_pairs(examples)
    if task == "revision":
        examples = [TypedRevisionExample.from_dict(r) for r in rows]
        for ex in examples:
            ex.validate()
        return build_revision_sft_pairs(examples)
    raise ValueError(f"unknown SFT task '{task}' (expected 'graph' or 'revision')")


def resolve_sft_pairs(
    task: str,
    *,
    input_path: str | Path | None,
    config: dict[str, Any] | None,
    smoke: bool,
) -> list[dict[str, str]]:
    """Pick the SFT dataset source, refusing to silently train on smoke data.

    Order: explicit ``--input`` / config ``data.train_path`` -> real JSONL rows;
    ``--smoke`` -> synthetic episodes; otherwise a loud error.
    """
    resolved = input_path
    if resolved is None and config is not None:
        resolved = (config.get("data") or {}).get("train_path")
    if resolved:
        return load_sft_pairs(task, resolved)
    if smoke:
        return build_sft_pairs(task)
    raise RuntimeError(
        "refusing to run real SFT on smoke synthetic episodes. Pass --input <jsonl> "
        "or set data.train_path in the config to a real ReTrace-Learn dataset "
        "(e.g. under data/retrace_learn/v1_0/). Use --smoke only for a sanity/dry run."
    )


def train(
    config_path: str | Path,
    *,
    input_path: str | Path | None = None,
    smoke: bool = False,
) -> None:  # pragma: no cover - requires GPU/deps
    config = load_config(config_path)
    task = config.get("train", {}).get("task", "revision")
    pairs = resolve_sft_pairs(task, input_path=input_path, config=config, smoke=smoke)
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
    parser.add_argument(
        "--input",
        default=None,
        help="explicit JSONL dataset for real training (graph/revision SFT rows)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the smoke synthetic episodes as the dataset (sanity only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="build dataset only")
    args = parser.parse_args(argv)

    if args.dry_run or args.config is None:
        # Dry-run inspects whatever source is given: explicit --input if present,
        # otherwise the smoke synthetic episodes.
        pairs = load_sft_pairs(args.task, args.input) if args.input else build_sft_pairs(args.task)
        source = args.input if args.input else "smoke synthetic episodes"
        print(f"[dry-run] task={args.task} source={source} built {len(pairs)} SFT pairs")
        if pairs:
            print(f"  sample prompt chars: {len(pairs[0]['prompt'])}")
            print(f"  sample completion: {pairs[0]['completion'][:160]}")
        return 0
    train(args.config, input_path=args.input, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
