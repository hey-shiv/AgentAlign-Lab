#!/usr/bin/env python3
"""Script 06: Evaluate models and compare baseline vs tuned.

Loads scored trajectories, computes metrics for each agent, and
prints a comparison table.

Usage:
    python scripts/06_eval_models.py --runs-dir data/trajectories/scored_train --split train --agent baseline --compare-agent bad_model
"""

import argparse
import json
from pathlib import Path

from agentalign.data.trajectories import load_all_trajectories
from agentalign.eval.failure_labels import print_failure_report
from agentalign.eval.metrics import compare_models, compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument(
        "--runs-dir", default="data/trajectories/scored_train",
        help="Directory of scored trajectories",
    )
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--agent", default="baseline", help="Primary agent to evaluate")
    parser.add_argument("--compare-agent", default=None, help="Agent to compare against")
    args = parser.parse_args()

    # Load all trajectories
    all_trajs = load_all_trajectories(args.runs_dir)
    print(f"Loaded {len(all_trajs)} trajectories from {args.runs_dir}")

    if not all_trajs:
        print("No trajectories found.")
        return

    # Split by agent
    agent_trajs = [t for t in all_trajs if t.agent_id == args.agent]
    print(f"\n{args.agent}: {len(agent_trajs)} trajectories")
    agent_metrics = compute_metrics(agent_trajs)

    # Print metrics
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for key, val in agent_metrics.items():
        if key not in ("failure_label_distribution", "total_trajectories"):
            print(f"  {key:<28} {val:>10}")

    # Compare if requested
    if args.compare_agent:
        compare_trajs = [t for t in all_trajs if t.agent_id == args.compare_agent]
        print(f"\n{args.compare_agent}: {len(compare_trajs)} trajectories")
        compare_metrics = compute_metrics(compare_trajs)

        comparison = compare_models(compare_metrics, agent_metrics)

        print(f"\n{'Metric':<25} {'Compare':>10} {'Agent':>10} {'Delta':>10} {'Dir':>10}")
        print("-" * 67)
        for key, info in comparison.items():
            print(
                f"  {key:<23} "
                f"{info['baseline']:>10.4f} "
                f"{info['tuned']:>10.4f} "
                f"{info['delta']:>+10.4f} "
                f"{info['direction']:>10}"
            )

        # Save comparison
        output_dir = Path("outputs/evals")
        output_dir.mkdir(parents=True, exist_ok=True)
        comp_file = output_dir / "comparison.json"
        comp_file.write_text(json.dumps({
            "split": args.split,
            "agent": args.agent,
            "compare_agent": args.compare_agent,
            "agent_metrics": agent_metrics,
            "compare_metrics": compare_metrics,
            "comparison": comparison,
        }, indent=2, default=str))
        print(f"\nComparison saved to {comp_file}")

    # Print failure report
    print_failure_report(agent_trajs)


if __name__ == "__main__":
    main()
