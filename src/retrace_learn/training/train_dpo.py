"""DPO over DPA-in-the-loop preference pairs (Stage RL-3, preference variant).

Preference pairs come from ``export_rl_rollouts.build_preference_pairs`` —
``chosen`` is the higher-reward completion, ``rejected`` the lower-reward one,
where reward is computed from parser + RevisionGate + DPA vs gold final statuses.
The dataset builder is runnable (``--dry-run``); ``train()`` lazily imports trl.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from retrace_learn.data.export_rl_rollouts import build_preference_pairs, build_rows


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_dpo_pairs() -> list[dict[str, Any]]:
    """(prompt, chosen, rejected) triples for DPO."""
    pairs = build_preference_pairs(build_rows())
    return [
        {
            "prompt": p["prompt_input"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]


def train(config_path: str | Path) -> None:  # pragma: no cover - requires GPU/deps
    config = load_config(config_path)
    pairs = build_dpo_pairs()
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "train() needs transformers+trl installed. Use --dry-run to inspect pairs."
        ) from exc

    model_cfg = config["model"]
    train_cfg = config["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"])
    dataset = Dataset.from_list(pairs)
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=train_cfg["output_dir"],
            learning_rate=train_cfg.get("learning_rate", 5.0e-6),
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
            beta=train_cfg.get("beta", 0.1),
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run or args.config is None:
        pairs = build_dpo_pairs()
        print(f"[dry-run] built {len(pairs)} DPO preference pairs")
        if pairs:
            print(f"  sample chosen:   {pairs[0]['chosen'][:120]}")
            print(f"  sample rejected: {pairs[0]['rejected'][:120]}")
        return 0
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
