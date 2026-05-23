#!/usr/bin/env python
"""Evaluate trained models.

This script evaluates models on the test set.
"""

import argparse
from pathlib import Path
from agentalign.eval.run import Evaluator
from agentalign.tasks.load import load_tasks_from_jsonl


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--model", type=str, default="outputs/adapters/dpo")
    parser.add_argument("--tasks", type=Path, default="data/tasks/test.jsonl")
    parser.add_argument("--output", type=Path, default="outputs/evals")
    parser.add_argument("--num-samples", type=int, default=100)

    args = parser.parse_args()

    print(f"Loading tasks from {args.tasks}...")
    tasks = load_tasks_from_jsonl(args.tasks)

    print(f"Evaluating {args.model}...")
    evaluator = Evaluator(args.output)
    results = evaluator.evaluate(args.model, tasks, args.num_samples)

    print(f"Evaluation results: {results}")
    print("Done!")


if __name__ == "__main__":
    main()
