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
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument("--adapter", default=None, help="Path to PEFT adapter")
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
        if key not in ("failure_label_distribution", "total_trajectories", "pass_rate_by_family", "avg_score_by_family"):
            print(f"  {key:<28} {val:>10}")

    print("\nPer-Family Pass Rate:")
    for fam, pr in agent_metrics.get("pass_rate_by_family", {}).items():
        print(f"  {fam:<28} {pr:>10.4f}")

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
        
        # --- Generate Beautiful Plots ---
        print("\nGenerating Evaluation Plots...")
        metrics_to_plot = ["pass_rate", "avg_score", "invalid_action_rate"]
        titles = ["Task Success Rate", "Average Score", "Invalid Action (JSON Error) Rate"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        sns.set_theme(style="whitegrid")
        
        for ax, metric, title in zip(axes, metrics_to_plot, titles):
            if metric in comparison:
                base_val = comparison[metric]["baseline"]
                tuned_val = comparison[metric]["tuned"]
                
                sns.barplot(
                    x=["Base Model", "Tuned Model"], 
                    y=[base_val, tuned_val], 
                    ax=ax,
                    palette=["#ff9999", "#66b3ff"]
                )
                ax.set_title(title, fontsize=14, pad=15)
                ax.set_ylabel("Value")
                
                # Add value labels on top of bars
                for i, v in enumerate([base_val, tuned_val]):
                    ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
        plt.tight_layout()
        plot_file = output_dir / "comparison_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")

    # Print failure report
    print_failure_report(agent_trajs)


if __name__ == "__main__":
    main()
