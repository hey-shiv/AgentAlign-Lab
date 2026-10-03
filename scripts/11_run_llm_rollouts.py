#!/usr/bin/env python3
"""Script 11: End-to-end LLM rollout pipeline.

Runs real HF model inference on train/val/test tasks, scores them,
and builds preference pairs automatically.

Supports checkpoint/resume so GPU runs can be restarted safely.
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HF model name")
    parser.add_argument("--reps", type=int, default=6,
                        help="Repetitions per task")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "all"],
                        help="Which split to run")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to run per split")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max agent steps per trajectory")
    parser.add_argument("--agent", default="qwen_base",
                        help="Agent identifier")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda/mps/cpu)")
    parser.add_argument("--adapter", default=None,
                        help="Path to PEFT adapter (for eval)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks with existing trajectories")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    python_bin = sys.executable

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

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
            "--temperature", str(args.temperature),
            "--device", args.device,
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
        ]
        if args.adapter:
            cmd_run.extend(["--adapter", args.adapter])
        if args.max_tasks:
            cmd_run.extend(["--max-tasks", str(args.max_tasks)])
        if args.resume:
            cmd_run.append("--resume")
        else:
            cmd_run.append("--clear")

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

    # Print summary
    for split in splits:
        scored_dir = Path(f"data/trajectories/scored_llm_{split}")
        pref_out = Path(f"data/preferences/dpo_llm_{split}.jsonl")
        n_traj = sum(1 for _ in scored_dir.glob("*.jsonl")) if scored_dir.exists() else 0
        n_pairs = sum(1 for l in pref_out.read_text().splitlines() if l.strip()) if pref_out.exists() else 0
        print(f"  {split}: {n_traj} trajectories, {n_pairs} preference pairs")


if __name__ == "__main__":
    main()
