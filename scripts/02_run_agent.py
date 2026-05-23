#!/usr/bin/env python
"""Run agent on tasks and collect trajectories.

This script runs the agent on generated tasks to collect trajectories.
"""

import argparse
from pathlib import Path
from agentalign.agent.loop import Agent
from agentalign.tasks.load import load_tasks_from_jsonl
from agentalign.data.trajectories import save_trajectories


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run agent on tasks")
    parser.add_argument("--tasks", type=Path, default="data/tasks/generated.jsonl")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--output", type=Path, default="data/trajectories/raw")

    args = parser.parse_args()

    print(f"Loading tasks from {args.tasks}...")
    tasks = load_tasks_from_jsonl(args.tasks)

    print(f"Running agent ({args.model}) on {len(tasks)} tasks...")
    agent = Agent(model_name=args.model)
    trajectories = [agent.run(task) for task in tasks]

    print(f"Saving trajectories to {args.output}...")
    save_trajectories(trajectories, args.output / "trajectories.jsonl")

    print("Done!")


if __name__ == "__main__":
    main()
