#!/usr/bin/env python3
"""Script 03: Score raw trajectories with the deterministic verifier.

Loads all raw trajectories, re-runs the verifier and score computation,
and saves the scored trajectories.

Usage:
    python scripts/03_score_trajectories.py --runs-dir runs/train --out-dir data/trajectories/scored_train --clear
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from agentalign.data.trajectories import load_all_trajectories, save_trajectory
from agentalign.verifier.score import calculate_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw trajectories")
    parser.add_argument("--runs-dir", default="runs/train", help="Directory of raw trajectories")
    parser.add_argument("--out-dir", default="data/trajectories/scored_train", help="Output directory")
    parser.add_argument("--clear", action="store_true", help="Clear output directory first")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all raw trajectories
    trajectories = load_all_trajectories(args.runs_dir)
    print(f"Loaded {len(trajectories)} trajectories from {args.runs_dir}")

    if not trajectories:
        print("No trajectories found. Run scripts/02_run_agent.py first.")
        return

    # Score each trajectory
    scores: list[float] = []
    for traj in trajectories:
        calculate_score(traj)
        save_trajectory(traj, out_dir)
        scores.append(traj.score)

    # Print summary
    print(f"\nScored {len(trajectories)} trajectories:")
    print(f"  Min score:  {min(scores):.2f}")
    print(f"  Max score:  {max(scores):.2f}")
    print(f"  Mean score: {sum(scores) / len(scores):.2f}")

    # Failure label distribution
    label_counts = Counter(
        t.verifier_result.failure_label
        for t in trajectories
        if t.verifier_result and t.verifier_result.failure_label
    )
    print("\nFailure label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    print(f"\nScored trajectories saved to {out_dir}/")


if __name__ == "__main__":
    main()
