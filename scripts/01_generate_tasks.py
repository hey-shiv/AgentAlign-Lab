#!/usr/bin/env python
"""Generate tasks for the lab.

This script generates the initial set of tasks for agent training and evaluation.
"""

import argparse
from pathlib import Path
from agentalign.tasks.generate import generate_tasks
from agentalign.tasks.load import save_tasks_to_jsonl


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate tasks")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--template", type=str, default="default")
    parser.add_argument("--output", type=Path, default="data/tasks/generated.jsonl")

    args = parser.parse_args()

    print(f"Generating {args.num_tasks} tasks...")
    tasks = generate_tasks(args.num_tasks, args.template)

    print(f"Saving to {args.output}...")
    save_tasks_to_jsonl(tasks, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
