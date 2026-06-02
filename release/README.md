# ReTrace Release

- `release/` is a local generated packaging area for releases (e.g., Hugging Face dataset uploads).
- It is **not** canonical source data.
- All canonical source data lives under `data/`.
- Hugging Face packages should be generated or refreshed from canonical data using scripts (e.g. `scripts/package_hf_retrace_bench.py`).
- Do not commit large duplicate datasets under `release/` to GitHub.
