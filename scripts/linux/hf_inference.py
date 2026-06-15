"""Shared Hugging Face 4-bit inference helpers for Linux eval scripts."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from scripts.linux.hf_chat_utils import apply_chat_template_no_think

_WARNINGS_CONFIGURED = False


def suppress_bitsandbytes_warnings() -> None:
    """Hide noisy third-party FutureWarnings during 4-bit load/generate."""
    global _WARNINGS_CONFIGURED
    if _WARNINGS_CONFIGURED:
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*_check_is_size will be removed.*",
        category=FutureWarning,
    )
    _WARNINGS_CONFIGURED = True


def _is_mistral_model(model_id: str) -> bool:
    text = model_id.lower()
    return "mistral" in text or "nemo" in text


def tokenizer_load_kwargs(model_id: str, hub: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(hub or {})
    if _is_mistral_model(model_id):
        out.setdefault("fix_mistral_regex", True)
    return out


def load_hf_tokenizer(model_id: str, hub: dict[str, Any] | None = None) -> Any:
    from transformers import AutoTokenizer

    suppress_bitsandbytes_warnings()
    kwargs = {"trust_remote_code": True, **tokenizer_load_kwargs(model_id, hub)}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_model(model_id: str, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from scripts.linux.hf_hub import hub_kwargs, log_hub_config

    suppress_bitsandbytes_warnings()
    log_hub_config(model_id)
    hub = hub_kwargs(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = load_hf_tokenizer(model_id, hub)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
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
    json_prefill: str = "{",
) -> tuple[str, dict[str, Any]]:
    import torch

    suppress_bitsandbytes_warnings()
    prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]
    input_ids, gen_meta = apply_chat_template_no_think(
        tokenizer,
        prompt_messages,
        json_prefill=json_prefill,
    )
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor)
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_tensor.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, gen_meta
