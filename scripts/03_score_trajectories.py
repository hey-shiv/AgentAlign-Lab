#!/usr/bin/env python
"""Score agent trajectories.

This script evaluates and scores collected trajectories.
"""

import argparse
from pathlib import Path
from agentalign.data.trajectories import load_trajectories, save_trajectories
from agentalign.verifier.score import SuccessScorer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Score trajectories")
    parser.add_argument("--input", type=Path, default="data/trajectories/raw")
    parser.add_argument("--output", type=Path, default="data/trajectories/scored")

    args = parser.parse_args()

    print(f"Loading trajectories from {args.input}...")
    trajectories = load_trajectories(args.input)

    print(f"Scoring {len(trajectories)} trajectories...")
    scorer = SuccessScorer()
    for traj in trajectories:
        traj.score = scorer.score(traj)

    print(f"Saving scored trajectories to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    save_trajectories(trajectories, args.output / "scored.jsonl")

    print("Done!")


if __name__ == "__main__":
    main()
