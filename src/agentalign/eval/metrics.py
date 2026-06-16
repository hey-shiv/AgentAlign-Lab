"""Evaluation metrics for agent trajectories.

Computes aggregate statistics over a collection of trajectories.
"""

from collections import Counter

from agentalign.schemas import Trajectory


def compute_metrics(trajectories: list[Trajectory]) -> dict:
    """Compute evaluation metrics over a list of trajectories.

    Metrics computed:
    - task_success_rate: fraction of trajectories where the verifier passed
    - avg_verifier_score: mean verifier score
    - avg_steps: mean number of steps per trajectory
    - invalid_action_rate: fraction of steps that were invalid
    - unsafe_action_rate: fraction of steps with forbidden commands
    - failure_label_distribution: counts per failure label

    Args:
        trajectories: List of scored trajectories.

    Returns:
        Dictionary of metric name to value.
    """
    if not trajectories:
        return {
            "task_success_rate": 0.0,
            "avg_verifier_score": 0.0,
            "avg_steps": 0.0,
            "invalid_action_rate": 0.0,
            "unsafe_action_rate": 0.0,
            "failure_label_distribution": {},
            "total_trajectories": 0,
        }

    n = len(trajectories)
    passed = sum(1 for t in trajectories if t.verifier_result and t.verifier_result.passed)
    total_score = sum(
        t.verifier_result.score for t in trajectories if t.verifier_result
    )
    total_steps = sum(len(t.steps) for t in trajectories)
    total_invalid = sum(
        (t.verifier_result.invalid_actions if t.verifier_result else 0)
        for t in trajectories
    )
    total_unsafe = sum(
        (t.verifier_result.unsafe_actions if t.verifier_result else 0)
        for t in trajectories
    )

    # Failure label distribution
    label_counter: Counter = Counter()
    for t in trajectories:
        if t.verifier_result and t.verifier_result.failure_label:
            label_counter[t.verifier_result.failure_label] += 1

    return {
        "task_success_rate": round(passed / n, 4),
        "avg_verifier_score": round(total_score / n, 4),
        "avg_steps": round(total_steps / n, 2),
        "invalid_action_rate": round(total_invalid / max(total_steps, 1), 4),
        "unsafe_action_rate": round(total_unsafe / max(total_steps, 1), 4),
        "failure_label_distribution": dict(label_counter.most_common()),
        "total_trajectories": n,
    }


def compare_models(baseline_metrics: dict, tuned_metrics: dict) -> dict:
    """Compare baseline vs tuned model metrics.

    Computes deltas for each numeric metric and flags improvements
    and regressions.

    Args:
        baseline_metrics: Metrics dict from compute_metrics for the baseline.
        tuned_metrics: Metrics dict from compute_metrics for the tuned model.

    Returns:
        Dictionary with baseline, tuned, delta, and direction for each metric.
    """
    comparison: dict = {}
    numeric_keys = [
        "task_success_rate", "avg_verifier_score", "avg_steps",
        "invalid_action_rate", "unsafe_action_rate",
    ]
    # Higher is better for these metrics
    higher_is_better = {"task_success_rate", "avg_verifier_score"}

    for key in numeric_keys:
        base_val = baseline_metrics.get(key, 0.0)
        tuned_val = tuned_metrics.get(key, 0.0)
        delta = round(tuned_val - base_val, 4)

        if key in higher_is_better:
            direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
        else:
            direction = "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"

        comparison[key] = {
            "baseline": base_val,
            "tuned": tuned_val,
            "delta": delta,
            "direction": direction,
        }

    return comparison
