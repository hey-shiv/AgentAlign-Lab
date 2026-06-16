#!/usr/bin/env python3
"""Script 01: Generate the full task suite.

Loads task family counts from configs/agent.yaml and generates
all tasks with a 70/15/15 train/val/test split.

Usage:
    python scripts/01_generate_tasks.py
"""

import json
from collections import Counter
from pathlib import Path

import yaml

from agentalign.tasks.generate import generate_all_tasks


def main() -> None:
    """Generate tasks and print a summary."""
    # Load config for task family counts (optional overrides)
    config_path = Path("configs/agent.yaml")
    config: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
        # The agent.yaml doesn't have per-family counts, so use defaults
        config = raw.get("task_counts", {})

    print("=" * 60)
    print("AgentAlign Lab — Task Generation")
    print("=" * 60)

    tasks = generate_all_tasks(config)

    # Print summary
    family_counts = Counter(t.family for t in tasks)
    print(f"\nGenerated {len(tasks)} tasks total:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")

    # Print split info
    tasks_dir = Path("data/tasks")
    for split in ["train", "val", "test"]:
        split_path = tasks_dir / f"{split}.jsonl"
        if split_path.exists():
            n = sum(1 for line in split_path.read_text().splitlines() if line.strip())
            print(f"  {split}: {n} tasks")

    print(f"\nTask files saved to {tasks_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
