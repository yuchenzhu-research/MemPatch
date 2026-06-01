#!/usr/bin/env python3
import sys
from pathlib import Path

# Add root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Please install it using pip install PyYAML.")
    sys.exit(1)


def main():
    root = Path(__file__).resolve().parent.parent
    agent_mem_reg = root / "references" / "agent_memory" / "registry.yaml"
    top_bench_reg = root / "references" / "top_benchmarks" / "registry.yaml"

    print(f"{'Name':<22} | {'Category':<15} | {'Repo Pointer':<55}")
    print("-" * 100)

    for path, cat in [(agent_mem_reg, "Agent Memory"), (top_bench_reg, "Top Benchmarks")]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                records = yaml.safe_load(f) or []
                for r in records:
                    name = r.get("name", "Unknown")
                    repo = r.get("repo", "N/A")
                    print(f"{name:<22} | {cat:<15} | {repo:<55}")


if __name__ == "__main__":
    main()
