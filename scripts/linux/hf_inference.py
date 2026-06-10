"""Shared Hugging Face 4-bit inference helpers for Linux eval scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.mlx_support.mlx_chat_utils import apply_chat_template_no_think


def load_hf_model(model_id: str, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from scripts.linux.hf_hub import hub_kwargs, log_hub_config

    log_hub_config(model_id)
    hub = hub_kwargs(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **hub,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_from_messages(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
    temp: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    import torch

    prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]
    input_ids, gen_meta = apply_chat_template_no_think(tokenizer, prompt_messages)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_tensor.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, gen_meta
