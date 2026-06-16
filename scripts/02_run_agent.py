#!/usr/bin/env python3
"""Script 02: Run the agent on tasks and collect trajectories.

Loads a model (or uses a scripted baseline/bad_model callable), runs
each task through the agent loop, and saves trajectories.

Usage:
    python scripts/02_run_agent.py --split train --agent baseline --model dummy --out runs/train --repetitions 8 --clear
    python scripts/02_run_agent.py --split train --agent bad_model --model bad_dummy --out runs/train --repetitions 8 --clear
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path

from agentalign.agent.loop import run_agent_loop
from agentalign.data.trajectories import save_trajectory
from agentalign.tasks.load import load_task, load_tasks_from_jsonl


# ---------------------------------------------------------------------------
# Scripted model callables (no GPU needed)
# ---------------------------------------------------------------------------

def _baseline_callable(prompt: str) -> str:
    """A scripted 'good' agent that reads files, runs tests, fixes bugs.

    This deterministic baseline produces valid JSON actions that follow
    a simple strategy: list files, read the source, attempt a fix, run tests.
    """
    prompt_lower = prompt.lower()

    # Determine which step we're on based on history
    if "step 1" not in prompt_lower and "history:" not in prompt_lower:
        # First step: list files
        return json.dumps({
            "thought": "Let me start by listing the files in the workspace.",
            "action": "list_files",
            "args": {},
        })

    if "list_files" in prompt and "read_file" not in prompt:
        # After listing, read the main file
        # Try to find a non-test .py file or the main task file
        return json.dumps({
            "thought": "I'll read the main source file to understand the bug.",
            "action": "read_file",
            "args": {"path": "stats.py"},
        })

    if "read_file" in prompt and "run_command" not in prompt:
        # After reading, run tests to see the error
        return json.dumps({
            "thought": "Let me run the tests to see the failure.",
            "action": "run_command",
            "args": {"cmd": "pytest -q"},
        })

    # Default: submit final answer
    return json.dumps({
        "thought": "I'll submit my final answer.",
        "action": "final_answer",
        "args": {"answer": "Attempted to fix the task."},
    })


def _bad_model_callable(prompt: str) -> str:
    """A scripted 'bad' agent that produces invalid JSON or forbidden commands."""
    import random
    r = random.random()
    if r < 0.3:
        # Invalid JSON
        return "I think we should fix the bug by... {invalid json"
    elif r < 0.5:
        # Forbidden command
        return json.dumps({
            "thought": "Let me install a package to help.",
            "action": "run_command",
            "args": {"cmd": "pip install numpy"},
        })
    elif r < 0.7:
        # Try to delete files
        return json.dumps({
            "thought": "Let me clean up.",
            "action": "run_command",
            "args": {"cmd": "rm -rf test_stats.py"},
        })
    else:
        # Wrong answer without testing
        return json.dumps({
            "thought": "I think the answer is obvious.",
            "action": "final_answer",
            "args": {"answer": "The code looks fine to me."},
        })


_MODEL_CALLABLES = {
    "dummy": _baseline_callable,
    "bad_dummy": _bad_model_callable,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent on tasks")
    parser.add_argument("--split", default="train", help="Task split to run (train/val/test)")
    parser.add_argument("--agent", default="baseline", help="Agent ID (baseline/bad_model)")
    parser.add_argument("--model", default="dummy", help="Model name or 'dummy'/'bad_dummy'")
    parser.add_argument("--out", default="runs/train", help="Output directory for trajectories")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per task")
    parser.add_argument("--clear", action="store_true", help="Clear output directory first")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks_path = Path(f"data/tasks/{args.split}.jsonl")
    if tasks_path.exists():
        tasks = load_tasks_from_jsonl(tasks_path)
    else:
        # Fallback: load individual JSON files
        tasks_dir = Path("data/tasks")
        tasks = [load_task(p) for p in sorted(tasks_dir.glob("*.json"))]

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"Running agent '{args.agent}' on {len(tasks)} tasks x {args.repetitions} reps")

    # Get model callable
    model_callable = _MODEL_CALLABLES.get(args.model)
    if model_callable is None:
        print(f"[WARN] Model '{args.model}' not available as scripted callable.")
        print("[WARN] Using dummy baseline instead. For real models, use HF inference.")
        model_callable = _baseline_callable

    success_count = 0
    total_count = 0

    for task in tasks:
        for rep in range(args.repetitions):
            total_count += 1
            try:
                trajectory = run_agent_loop(
                    task=task,
                    model_callable=model_callable,
                    agent_id=args.agent,
                    max_steps=8,
                    model_name=args.model,
                )
                save_trajectory(trajectory, out_dir)
                if trajectory.success:
                    success_count += 1
                status = "✓" if trajectory.success else "✗"
                score = trajectory.score
                print(f"  {status} {task.task_id} rep={rep} score={score:.1f}")
            except Exception as exc:
                print(f"  ✗ {task.task_id} rep={rep} ERROR: {exc}")

    print(f"\nDone. {success_count}/{total_count} succeeded.")
    print(f"Trajectories saved to {out_dir}/")


if __name__ == "__main__":
    main()
