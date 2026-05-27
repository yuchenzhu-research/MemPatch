from __future__ import annotations

import argparse

from retracemem.adapters.memora_adapter import MemoraAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReTrace Memora smoke check.")
    parser.add_argument("--reference-root", default="reference/Memora")
    args = parser.parse_args()

    adapter = MemoraAdapter(args.reference_root)
    if not adapter.exists():
        raise SystemExit(f"Memora reference root not found: {args.reference_root}")
    print(f"Memora reference root: {args.reference_root}")


if __name__ == "__main__":
    main()
