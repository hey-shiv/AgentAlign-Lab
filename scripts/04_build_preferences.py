#!/usr/bin/env python
"""Build preference pairs from trajectories.

This script creates preference pairs for DPO training.
"""

import argparse
from pathlib import Path
from agentalign.data.trajectories import load_trajectories
from agentalign.data.preferences import build_preferences_from_trajectories, save_preferences


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build preference pairs")
    parser.add_argument("--input", type=Path, default="data/trajectories/scored")
    parser.add_argument("--output", type=Path, default="data/preferences")

    args = parser.parse_args()

    print(f"Loading trajectories from {args.input}...")
    trajectories = load_trajectories(args.input)

    # Split by success
    good = [t for t in trajectories if t.success]
    bad = [t for t in trajectories if not t.success]

    print(f"Building preference pairs ({len(good)} good, {len(bad)} bad)...")
    preferences = build_preferences_from_trajectories(good, bad)

    print(f"Saving preferences to {args.output}...")
    save_preferences(preferences, args.output / "preferences.jsonl")

    print("Done!")


if __name__ == "__main__":
    main()
