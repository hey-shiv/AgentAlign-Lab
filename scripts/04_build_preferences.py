#!/usr/bin/env python3
"""Script 04: Build DPO preference pairs from scored trajectories.

Groups scored trajectories by task_id, creates chosen/rejected pairs,
filters by minimum score margin, and saves to JSONL.

Usage:
    python scripts/04_build_preferences.py --runs-dir data/trajectories/scored_train --out data/preferences/dpo_train.jsonl --split train --min-margin 2.0
"""

import argparse
from pathlib import Path

from agentalign.data.preferences import build_preference_pairs, save_preference_pairs
from agentalign.data.trajectories import load_all_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument(
        "--runs-dir", default="data/trajectories/scored_train",
        help="Directory of scored trajectories",
    )
    parser.add_argument(
        "--out", default="data/preferences/dpo_train.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument("--split", default="train", help="Split name for logging")
    parser.add_argument(
        "--min-margin", type=float, default=2.0,
        help="Minimum score margin for a valid pair",
    )
    args = parser.parse_args()

    # Load scored trajectories
    trajectories = load_all_trajectories(args.runs_dir)
    print(f"Loaded {len(trajectories)} scored trajectories from {args.runs_dir}")

    if not trajectories:
        print("No trajectories found. Run scripts/03_score_trajectories.py first.")
        return

    # Build pairs
    pairs = build_preference_pairs(trajectories, min_score_margin=args.min_margin)
    print(f"\nBuilt {len(pairs)} preference pairs (min margin = {args.min_margin})")

    if pairs:
        margins = [p.score_margin for p in pairs]
        print(f"  Min margin:  {min(margins):.2f}")
        print(f"  Max margin:  {max(margins):.2f}")
        print(f"  Mean margin: {sum(margins) / len(margins):.2f}")

    # Save
    save_preference_pairs(pairs, args.out)
    print(f"\nPreference pairs saved to {args.out}")


if __name__ == "__main__":
    main()
