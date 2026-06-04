# ReTrace-Bench — Hugging Face v1.1 Release Plan

How to publish the canonical ReTrace-Bench release to Hugging Face. **Nothing is
published by this cleanup pass.** These are the manual steps for the user.

> Public release name is simply **ReTrace-Bench** (the internal "v1.1" label is
> not used in the public card). The current HF repo holds the v1.0 legacy/pilot
> dataset; archive/tag it before overwriting.

## Bundle location

Built locally by:

```bash
python scripts/build_hf_release_v1_1.py            # public bundle (3780 cases)
python scripts/build_hf_release_v1_1.py --include-private   # also stages private/ (local only)
```

Output: `hf_release/retrace_bench_v1_1/`

```
hf_release/retrace_bench_v1_1/
  main/scenarios.jsonl            (3000)   [local only, git-ignored]
  hard/scenarios.jsonl            (500)    [local only, git-ignored]
  realistic/scenarios.jsonl       (200)    [local only, git-ignored]
  calibration/scenarios.jsonl     (80)     [local only, git-ignored]
  <split>/manifest.json           [tracked]
  README.md                       [tracked] dataset card
  dataset_info.json               [tracked]
  DATASET_LICENSE.md              [tracked] CC BY 4.0
  checksums.json                  [tracked] sha256 of each scenarios.jsonl
  manifest.json                   [tracked] splits, counts, seed, licenses
  VERSION                         [tracked] 1.1.0
```

The full `scenarios.jsonl` files are intentionally **git-ignored** (size +
dataset-license hosting policy): GitHub keeps code, validators, manifests,
checksums, and the dataset card; the full data is distributed on Hugging Face.
Regenerate the JSONL locally with the command above before uploading.

## Pre-flight checks

```bash
# 1. Splits validate clean (realistic warns: synthetic_gold_unreviewed)
for s in main hard realistic calibration; do
  python scripts/validate_retrace_bench_dataset.py \
    --data hf_release/retrace_bench_v1_1/$s/scenarios.jsonl
done

# 2. Gold oracle = 1.0 on core metrics (self-consistency)
python scripts/check_retrace_bench_gold_oracle.py \
  --data hf_release/retrace_bench_v1_1/hard/scenarios.jsonl \
  --out /tmp/oracle_hard.json

# 3. Checksums match the staged files
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("hf_release/retrace_bench_v1_1")
want = json.loads((root / "checksums.json").read_text())
for rel, h in want.items():
    got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    assert got == h, f"checksum mismatch: {rel}"
print("checksums OK")
PY
```

## Upload steps (user runs these)

1. **Archive v1.0 first.** In the existing dataset repo
   `https://huggingface.co/datasets/Sylvan-Vale-Moon/ReTrace-Bench`, tag or move
   the current v1.0 files (e.g. into a `legacy_v1_0/` folder or a `v1.0` git tag)
   so the pilot data is not lost or silently overwritten.

2. **Authenticate.**
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login        # paste an HF write token; do NOT hardcode it
   ```

3. **Upload the public bundle.**
   ```bash
   huggingface-cli upload Sylvan-Vale-Moon/ReTrace-Bench \
     hf_release/retrace_bench_v1_1 . --repo-type dataset
   ```
   (Do **not** upload the `private/` subtree to the public repo. If a private
   evaluation set is desired, push it to a separate **private** dataset repo.)

4. **Verify on the Hub.**
   - Dataset card renders; license shows CC BY 4.0.
   - All four configs (`main`, `hard`, `realistic`, `calibration`) load in the
     viewer with the expected row counts (3000 / 500 / 200 / 80).
   - `realistic` card text still says `synthetic_gold_unreviewed`.
   - `calibration` still carries the smoke-only warning.
   - Re-download and re-check `checksums.json`.

## Integrity reminders

- Do not publish `private_hidden`.
- Do not put any API key / token in the card, manifest, commit, or logs.
- Do not advertise `realistic` as human-validated until
  `human_validation_status.md` says so.
- Do not present the lost-run hard500 API numbers as official results.
