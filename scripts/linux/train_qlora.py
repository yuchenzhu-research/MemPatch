#!/usr/bin/env python3
"""QLoRA SFT for one k-fold (paper profile: rank-16 attn+mlp, 256 steps)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__)

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_example(row: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    messages = row["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def val_loss_from_checkpoint(ckpt_dir: Path) -> float | None:
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    eval_losses = [
        float(row["eval_loss"])
        for row in state.get("log_history", [])
        if "eval_loss" in row
    ]
    if not eval_losses:
        return None
    return eval_losses[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--save-steps", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-total-limit", type=int, default=8)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    train_rows = [format_example(r, tokenizer) for r in read_jsonl(args.train_data)]
    valid_rows = [format_example(r, tokenizer) for r in read_jsonl(args.valid_data)]
    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    warmup_steps = 0 if args.max_steps <= 16 else max(1, int(args.max_steps * 0.03))
    logging_steps = min(16, max(1, args.save_steps))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume and not Path(resume).is_dir():
        print(f"error: resume checkpoint not found: {resume}", file=sys.stderr)
        return 1
    trainer.train(resume_from_checkpoint=resume)

    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in sorted(args.output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        step_str = checkpoint.name.split("-", 1)[-1]
        if not step_str.isdigit():
            continue
        val_loss = val_loss_from_checkpoint(checkpoint)
        if val_loss is None:
            continue
        checkpoint_rows.append(
            {
                "step": int(step_str),
                "val_loss": val_loss,
                "checkpoint_dir": str(checkpoint),
            }
        )

    best_val = min((row["val_loss"] for row in checkpoint_rows), default=None)
    metrics_payload = {
        "model_id": args.model_id,
        "output_dir": str(args.output_dir),
        "best_val_loss": best_val,
        "checkpoints": sorted(checkpoint_rows, key=lambda r: r["step"]),
    }
    metrics_path = args.log_dir / "trainer_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
