#!/usr/bin/env python3
"""Script 11: End-to-end LLM rollout pipeline.

Runs real HF model inference on train/val/test tasks, scores them,
and builds preference pairs automatically.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str]) -> None:
    print(f"\n> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end LLM rollout pipeline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="HF model name")
    parser.add_argument("--reps", type=int, default=4, help="Repetitions per task")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Which split to run")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run per split")
    parser.add_argument("--agent", default="qwen_base", help="Agent identifier")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    python_bin = sys.executable

    for split in splits:
        print(f"\n{'='*50}\nProcessing split: {split}\n{'='*50}")
        
        runs_dir = f"runs/llm_{split}"
        scored_dir = f"data/trajectories/scored_llm_{split}"
        pref_out = f"data/preferences/dpo_llm_{split}.jsonl"

        # 1. Run agent
        cmd_run = [
            python_bin, "scripts/02_run_agent.py",
            "--split", split,
            "--agent", args.agent,
            "--model", args.model,
            "--out", runs_dir,
            "--repetitions", str(args.reps),
            "--clear"
        ]
        if args.max_tasks:
            cmd_run.extend(["--max-tasks", str(args.max_tasks)])
        
        run_cmd(cmd_run)
        
        # 2. Score trajectories
        cmd_score = [
            python_bin, "scripts/03_score_trajectories.py",
            "--runs-dir", runs_dir,
            "--out-dir", scored_dir,
            "--clear"
        ]
        run_cmd(cmd_score)
        
        # 3. Build preference pairs
        cmd_pref = [
            python_bin, "scripts/04_build_preferences.py",
            "--runs-dir", scored_dir,
            "--out", pref_out,
            "--split", split,
            "--min-margin", "2.0"
        ]
        run_cmd(cmd_pref)

    print("\n✅ LLM Rollouts Pipeline Complete!")

if __name__ == "__main__":
    main()
