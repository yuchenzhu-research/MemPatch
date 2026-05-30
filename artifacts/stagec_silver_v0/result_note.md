# Stage C Silver v0 Result Note

Status: exploratory local QLoRA result
Scientific status: silver_synthetic_training_only
Not for paper main results: true

Base model:
- Qwen3-4B-4bit, local MLX

Adapter:
- outputs/local_training/stagec_qwen3_4b_silver/adapters/adapters.safetensors
- trained for 300 steps
- batch_size = 1
- num_layers = 1
- save_every = 15

Dataset:
- 512 train rows
- 8 valid rows
- 8 test rows
- generated from ReTrace synthetic dev expansion
- STALE not used for training

Held-out test result:

Adapter:
- valid_json = 8/8
- action_type_match = 7/8
- exact_match = 6/8

Base Qwen3-4B prompt-only:
- valid_json = 1/8
- action_type_match = 1/8
- exact_match = 1/8

Main finding:
- The LoRA adapter substantially improves structured typed-action proposal behavior.
- Base model often produces long reasoning without final parseable JSON.

Known failure cases:
- test_00: gold = SUPERSEDES + BLOCKS, adapter predicted only SUPERSEDES.
  This indicates multi-action composition needs stronger supervision.
- test_05: action type UNCERTAIN matched, but exact match failed, likely due to target/evidence grounding mismatch.

Next steps:
- Build Silver v1 with more multi-action examples.
- Add REAFFIRMS coverage.
- Add grounding-specific evaluation metrics.
- Later compare on human-reviewed dev/test data.
