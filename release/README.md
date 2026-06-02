# ReTrace Release

- `release/` stores exact release-package snapshots, such as the Hugging Face
  dataset upload bundle for ReTrace-Bench.
- It is **not** canonical source data. All canonical source data lives under
  `data/`.
- Hugging Face packages should be generated or refreshed from canonical data
  using scripts (e.g. `scripts/package_hf_retrace_bench.py`) before updating a
  tracked release snapshot.
- Do not add ad hoc duplicate datasets under `release/`; keep tracked release
  packages limited to intentional public-release snapshots.
