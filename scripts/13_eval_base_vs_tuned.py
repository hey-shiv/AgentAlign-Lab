#!/usr/bin/env python3
"""Script 13: Full Evaluation Pipeline (Base vs Tuned).

Runs both base and tuned models on the test split, scores trajectories,
computes metrics, and compares them side-by-side.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

def run_cmd(cmd: list[str]) -> None:
    print(f"\n> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs tuned model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model")
    parser.add_argument("--adapter", default="outputs/adapters/qwen_dpo_final", help="Tuned adapter path")
    parser.add_argument("--split", default="test", help="Task split")
    parser.add_argument("--reps", type=int, default=4, help="Repetitions per task")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to evaluate")
    args = parser.parse_args()

    python_bin = sys.executable
    
    agents = [
        {"id": "qwen_base", "adapter": None},
        {"id": "qwen_dpo_tuned", "adapter": args.adapter},
    ]

    runs_dir = f"runs/llm_{args.split}"
    scored_dir = f"data/trajectories/scored_llm_{args.split}"
    
    # 1. Run both agents
    for agent in agents:
        print(f"\n{'='*50}\nRunning Agent: {agent['id']}\n{'='*50}")
        cmd_run = [
            python_bin, "scripts/02_run_agent.py",
            "--split", args.split,
            "--agent", agent["id"],
            "--model", args.model,
            "--out", runs_dir,
            "--repetitions", str(args.reps),
        ]
        if agent["adapter"]:
            cmd_run.extend(["--adapter", agent["adapter"]])
        if args.max_tasks:
            cmd_run.extend(["--max-tasks", str(args.max_tasks)])
            
        run_cmd(cmd_run)
        
    # 2. Score trajectories
    print(f"\n{'='*50}\nScoring Trajectories\n{'='*50}")
    run_cmd([
        python_bin, "scripts/03_score_trajectories.py",
        "--runs-dir", runs_dir,
        "--out-dir", scored_dir,
        "--clear"
    ])
    
    # 3. Evaluate and compare
    print(f"\n{'='*50}\nComparing Metrics\n{'='*50}")
    run_cmd([
        python_bin, "scripts/06_eval_models.py",
        "--runs-dir", scored_dir,
        "--split", args.split,
        "--agent", "qwen_base",
        "--compare-agent", "qwen_dpo_tuned"
    ])
    
    print("\nEvaluation Pipeline Complete!")
    print(f"Check outputs/evals/comparison.json for details.")

if __name__ == "__main__":
    main()
