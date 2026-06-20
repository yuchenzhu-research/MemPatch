# MemPatch

Benchmark and reference runtime for post-admission memory revision in persistent LLM-agent memory.

Main code:

- `benchmark/`: benchmark API, generator, scorer
- `mempatch/`: revision runtime and Guard
- `experiments/`: current evaluation campaign
- `scripts/`: dataset, scoring, and campaign wrapper

Minimal commands:

```bash
pip install -e ".[dev]"
python scripts/evaluate_mempatch_predictions.py --help
scripts/run_paper_campaign.sh smoke qwen3_14b
```

Blind-review paper material is under `Montreal/`. The ChatGPT review bundle is `review.zip`.
