"""Evaluation runner.

Orchestrates running a model (with optional adapter) on a set of tasks
and collecting metrics.
"""

import json
from pathlib import Path
from typing import Optional

from agentalign.data.trajectories import load_all_trajectories
from agentalign.eval.metrics import compute_metrics


def run_evaluation(
    model_path: str,
    adapter_path: Optional[str],
    tasks_path: str,
    config: dict,
) -> dict:
    """Run evaluation on a model.

    If pre-scored trajectories exist in the runs directory, loads them
    directly. Otherwise, would run inference (requires GPU).

    Args:
        model_path: Path or HF model ID for the base model.
        adapter_path: Optional path to a PEFT adapter.
        tasks_path: Path to the tasks JSONL file.
        config: Evaluation configuration dict.

    Returns:
        Dictionary with evaluation results and metrics.
    """
    output_dir = Path(config.get("output_dir", "outputs/evals"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = config.get("runs_dir", "")
    agent_name = config.get("agent", "baseline")

    results = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "tasks_path": tasks_path,
        "agent": agent_name,
    }

    # Try to load pre-scored trajectories
    trajectories = []
    if runs_dir and Path(runs_dir).exists():
        trajectories = load_all_trajectories(runs_dir)
        # Filter by agent if specified
        if agent_name:
            trajectories = [
                t for t in trajectories
                if t.agent_id == agent_name
            ]

    if trajectories:
        results["metrics"] = compute_metrics(trajectories)
        results["num_trajectories"] = len(trajectories)
    else:
        print(f"[eval] No pre-scored trajectories found for agent '{agent_name}'.")
        print("[eval] Run scripts/02_run_agent.py and scripts/03_score_trajectories.py first.")
        results["metrics"] = compute_metrics([])
        results["num_trajectories"] = 0

    # Save results
    result_file = output_dir / f"eval_{agent_name}.json"
    result_file.write_text(json.dumps(results, indent=2))
    print(f"[eval] Results saved to {result_file}")

    return results
