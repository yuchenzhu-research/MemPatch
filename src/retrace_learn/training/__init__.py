"""Training entrypoints for ReTrace-Learn (LoRA SFT, DPO, GRPO).

Heavy ML dependencies (torch / transformers / peft / trl) are imported *lazily*
inside the ``train_*`` functions so this package imports cleanly in a CPU-only,
dependency-light environment. The dataset-building halves are pure-stdlib and
fully runnable/testable via ``--dry-run``.
"""
