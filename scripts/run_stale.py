from __future__ import annotations

import argparse

from retracemem.adapters.stale_adapter import StaleAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReTrace STALE smoke check.")
    parser.add_argument("--reference-root", default="reference/STALE")
    args = parser.parse_args()

    adapter = StaleAdapter(args.reference_root)
    if not adapter.exists():
        raise SystemExit(f"STALE reference root not found: {args.reference_root}")
    print(f"STALE reference root: {args.reference_root}")


if __name__ == "__main__":
    main()
