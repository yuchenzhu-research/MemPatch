# API provider abstraction

ReTrace's evaluation runner is **provider-agnostic**. Proposers (Stage A / C)
and the DirectJudge baseline (Stage B) talk to an LLM through a small provider
layer in `src/retracemem/providers/`, so the runner never hard-codes one
vendor's request/response shape. The deterministic kernel (`authorize(...)`,
RevisionGate, DPA) is unaffected and remains API-free.

## Provider modes

| `mode` | Endpoint style | Examples |
| --- | --- | --- |
| `openai-chat` | `/v1/chat/completions` | OpenAI, SiliconFlow, DeepSeek |
| `custom-openai-compatible` | `/v1/chat/completions` | vLLM, SGLang, LM Studio, local servers |
| `anthropic-messages` | `/v1/messages` | Anthropic Claude |
| `ollama-chat` | `/api/chat` | local Ollama (no key required) |

`openai-chat` and `custom-openai-compatible` share the OpenAI-compatible request
shape; the second name only documents intent for self-hosted endpoints.

## Selecting a provider

Two interchangeable ways; `--model` always stays authoritative over the config:

```bash
# 1) Registry name (backward compatible) — resolves via configs/providers.yaml
python3 scripts/evaluate.py stage-a --live --provider siliconflow \
    --model deepseek-ai/DeepSeek-V3 --constrained

# 2) Single-provider config file — mode/base_url/api_key_env come from the file
python3 scripts/evaluate.py stage-a --live \
    --provider-config configs/providers/siliconflow.yaml \
    --model deepseek-ai/DeepSeek-V3 --constrained
```

## Config file schema

A single-provider config (`configs/providers/*.yaml`) is a flat mapping of
`ProviderConfig` fields (`src/retracemem/providers/config.py`):

```yaml
name: siliconflow
mode: openai-chat                                  # see modes table
base_url: https://api.siliconflow.cn/v1/chat/completions   # full endpoint URL
api_key_env: SILICONFLOW_API_KEY                   # env var NAME, never the key
model: deepseek-ai/DeepSeek-V3
timeout: 60
max_retries: 2
temperature: 0.0
max_tokens: 1024
extra_headers: {}                                  # optional
reasoning: false                                   # optional
stream: false                                      # forced off for reproducibility
```

The registry file `configs/providers.yaml` maps short names to defaults
(`mode`, `default_base_url`, `api_key_env`, optional `extra_headers`) and is what
`--provider <name>` resolves against.

## Secrets

* `base_url` is the **full** endpoint URL (including path), so one field works
  across every mode without mode-specific path assembly.
* API keys are **never** stored in configs — only the *name* of the environment
  variable (`api_key_env`) is. Keys are resolved at call time from the
  environment (e.g. a git-ignored `.env` at the repo root).
* If a key is required by the mode but the env var is unset, the runner fails
  closed with a clear `ProviderConfigError` naming the missing variable.
* Local modes (`ollama-chat`) require no key.
* Outgoing requests funnel through one transport helper
  (`providers/_transport.py`) that redacts known secret values from any error
  message or trace before it is logged.

## Example configs

`configs/providers/` ships:

* `siliconflow.yaml` — concrete OpenAI-compatible example (ready to use).
* `openai_compatible.yaml.example` — generic OpenAI-compatible template.
* `anthropic.yaml.example` — Anthropic Messages template.
* `ollama.yaml.example` — local Ollama template (no key).
* `lmstudio.yaml.example` — LM Studio (`custom-openai-compatible`) template.

Copy an `.example` file to a real `.yaml` (or add an entry to
`configs/providers.yaml`) and set the matching `api_key_env` in your `.env`.
