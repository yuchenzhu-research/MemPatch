#!/usr/bin/env python3
"""QLoRA SFT for one fixed stratified train/val partition within train3500."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
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


def subsample_eval_dataset(dataset: Any, *, max_samples: int, seed: int) -> Any:
    """Cap in-train eval rows to reduce peak VRAM (checkpoint pick uses relative val_loss)."""
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    import random

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(sorted(indices[:max_samples]))


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
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-total-limit", type=int, default=8)
    parser.add_argument("--eval-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="Cap validation rows during in-train eval (0 = full valid set).",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer
    from trl import SFTConfig, SFTTrainer

    from scripts.linux.hf_hub import hub_kwargs, log_hub_config

    log_hub_config(args.model_id)
    hub = hub_kwargs(args.model_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    from scripts.linux.hf_inference import load_hf_tokenizer, suppress_bitsandbytes_warnings

    suppress_bitsandbytes_warnings()
    tokenizer = load_hf_tokenizer(args.model_id, hub)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        **hub,
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
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    train_rows = [format_example(r, tokenizer) for r in read_jsonl(args.train_data)]
    valid_rows = [format_example(r, tokenizer) for r in read_jsonl(args.valid_data)]
    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)
    valid_ds = subsample_eval_dataset(
        valid_ds,
        max_samples=args.eval_max_samples,
        seed=args.seed,
    )
    if args.eval_max_samples > 0:
        print(
            f"in-train eval capped to {len(valid_ds)} / {len(valid_rows)} validation rows",
            flush=True,
        )

    warmup_steps = 0 if args.max_steps <= 16 else max(1, int(args.max_steps * 0.03))
    logging_steps = min(16, max(1, args.save_steps))

    sft_parameters = inspect.signature(SFTConfig).parameters
    sft_config_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": True,
        "seed": args.seed,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if "per_device_eval_batch_size" in sft_parameters:
        sft_config_kwargs["per_device_eval_batch_size"] = 1
    if "eval_accumulation_steps" in sft_parameters:
        sft_config_kwargs["eval_accumulation_steps"] = max(1, args.eval_accumulation_steps)
    if "dataloader_pin_memory" in sft_parameters:
        sft_config_kwargs["dataloader_pin_memory"] = False
    if "dataloader_num_workers" in sft_parameters:
        sft_config_kwargs["dataloader_num_workers"] = 0
    if "gradient_checkpointing" in sft_parameters:
        sft_config_kwargs["gradient_checkpointing"] = True
    if "gradient_checkpointing_kwargs" in sft_parameters:
        sft_config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "prediction_loss_only" in sft_parameters:
        sft_config_kwargs["prediction_loss_only"] = True
    if "eval_use_gather_for_metrics" in sft_parameters:
        sft_config_kwargs["eval_use_gather_for_metrics"] = False
    if "eval_strategy" in sft_parameters:
        sft_config_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sft_parameters:
        sft_config_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("installed TRL SFTConfig has no supported evaluation-strategy option")
    if "save_strategy" in sft_parameters:
        sft_config_kwargs["save_strategy"] = "steps"
    if "load_best_model_at_end" in sft_parameters:
        sft_config_kwargs["load_best_model_at_end"] = True
    if "max_length" in sft_parameters:
        sft_config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_parameters:
        sft_config_kwargs["max_seq_length"] = args.max_seq_length
    else:
        raise RuntimeError("installed TRL SFTConfig has no supported sequence-length option")

    training_args = SFTConfig(
        **sft_config_kwargs,
    )

    resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    resume_global_step: int | None = None
    if resume and not Path(resume).is_dir():
        print(f"error: resume checkpoint not found: {resume}", file=sys.stderr)
        return 1
    if resume:
        state_path = Path(resume) / "trainer_state.json"
        if not state_path.is_file():
            print(f"error: resume checkpoint has no trainer_state.json: {resume}", file=sys.stderr)
            return 1
        resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        resume_global_step = int(resume_state.get("global_step", -1))
        if resume_global_step < 0 or resume_global_step >= args.max_steps:
            print(
                f"error: invalid resume global_step={resume_global_step} for max_steps={args.max_steps}",
                file=sys.stderr,
            )
            return 1

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": valid_ds,
    }
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("installed TRL SFTTrainer accepts neither processing_class nor tokenizer")

    from transformers import TrainerCallback

    class MemorySafeSFTTrainer(SFTTrainer):
        """Avoid TRL eval path that materializes full vocab logits for metrics."""

        def _loss_only_eval_enabled(self) -> bool:
            return bool(getattr(self.args, "prediction_loss_only", False))

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """TRL SFTTrainer.prediction_step still builds full logits; bypass it."""
            if not (prediction_loss_only or self._loss_only_eval_enabled()):
                return super().prediction_step(
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, return_outputs=False)
            return (loss.detach(), None, None)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss_only_eval = self._loss_only_eval_enabled() and not model.training
            if not loss_only_eval:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            eval_inputs = dict(inputs)
            eval_inputs["use_cache"] = False
            forward_params = inspect.signature(model.forward).parameters
            if "skip_logits" in forward_params:
                eval_inputs["skip_logits"] = True
            loss = Trainer.compute_loss(
                self,
                model,
                eval_inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )
            return (loss, None) if return_outputs else loss

    class ReleaseCudaCacheCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    trainer_kwargs["callbacks"] = [ReleaseCudaCacheCallback()]
    trainer = MemorySafeSFTTrainer(**trainer_kwargs)
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
        "training_config": {
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "load_best_model_at_end": bool(
                sft_config_kwargs.get("load_best_model_at_end", False)
            ),
            "max_seq_length": args.max_seq_length,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_rank * 2,
            "learning_rate": args.learning_rate,
            "eval_accumulation_steps": args.eval_accumulation_steps,
            "eval_max_samples": args.eval_max_samples,
            "gradient_checkpointing": True,
            "prediction_loss_only": True,
            "memory_safe_eval": True,
            "seed": args.seed,
            "resume_from_checkpoint": resume,
            "resume_global_step": resume_global_step,
            "final_global_step": int(trainer.state.global_step),
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets")
        },
    }
    metrics_path = args.log_dir / "trainer_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")

    del trainer
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
