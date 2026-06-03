"""DPO over DPA-guided preference pairs (Stage 3 protocol).

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
from retrace_learn.data.jsonl_io import read_jsonl


def load_config(path: str | Path) -> dict[str, Any]:
    from retrace_learn.training import check_contamination
    check_contamination(path)
    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    check_contamination(config)
    return config


def build_dpo_pairs() -> list[dict[str, Any]]:
    """(prompt, chosen, rejected) triples from the smoke rollouts (sanity only)."""
    pairs = build_preference_pairs(build_rows())
    return [
        {
            "prompt": p["prompt_input"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]


def load_dpo_pairs(input_path: str | Path) -> list[dict[str, Any]]:
    """Load (prompt, chosen, rejected) preference triples from explicit JSONL."""
    from retrace_learn.training import check_contamination
    check_contamination(input_path)
    pairs = []
    for r in read_jsonl(input_path):
        prompt = r.get("prompt") or r.get("prompt_input")
        if prompt is None or "chosen" not in r or "rejected" not in r:
            raise RuntimeError(
                "DPO rows require prompt/prompt_input, chosen, and rejected fields"
            )
        pairs.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    if not pairs:
        raise RuntimeError(f"no DPO preference pairs found in '{input_path}'")
    return pairs


def resolve_dpo_pairs(
    *,
    input_path: str | Path | None,
    config: dict[str, Any] | None,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Pick the DPO dataset source, refusing to silently train on smoke data."""
    resolved = input_path
    if resolved is None and config is not None:
        data_cfg = config.get("data") or {}
        resolved = data_cfg.get("preference_path") or data_cfg.get("train_path")
    if resolved:
        return load_dpo_pairs(resolved)
    if smoke:
        return build_dpo_pairs()
    raise RuntimeError(
        "refusing to run real DPO on smoke preference pairs. Pass --input <jsonl> "
        "or set data.preference_path in the config to real DPA-induced preference "
        "data (e.g. exported under data/retrace_learn/v1_0/). Use --smoke for a sanity run."
    )


def train(
    config_path: str | Path,
    *,
    input_path: str | Path | None = None,
    smoke: bool = False,
) -> None:  # pragma: no cover - requires GPU/deps
    config = load_config(config_path)
    pairs = resolve_dpo_pairs(input_path=input_path, config=config, smoke=smoke)
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
    parser.add_argument(
        "--input",
        default=None,
        help="explicit JSONL preference dataset (prompt/chosen/rejected rows)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the smoke rollout preference pairs as the dataset (sanity only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run or args.config is None:
        pairs = load_dpo_pairs(args.input) if args.input else build_dpo_pairs()
        source = args.input if args.input else "smoke rollout pairs"
        print(f"[dry-run] source={source} built {len(pairs)} DPO preference pairs")
        if pairs:
            print(f"  sample chosen:   {pairs[0]['chosen'][:120]}")
            print(f"  sample rejected: {pairs[0]['rejected'][:120]}")
        return 0
    train(args.config, input_path=args.input, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
