"""DPO preference pair construction from scored trajectories.

Builds chosen/rejected pairs for Direct Preference Optimization training
by comparing trajectories on the same task.
"""

import json
import uuid
from collections import defaultdict
from pathlib import Path

from agentalign.schemas import Preference, PreferencePair, Step, Trajectory


def trajectory_to_text(trajectory: Trajectory) -> str:
    """Format a trajectory as a readable string of thought/action/observation steps.

    This is used as the 'chosen' or 'rejected' text in a preference pair.

    Args:
        trajectory: The trajectory to format.

    Returns:
        Multi-line string representing the trajectory.
    """
    parts: list[str] = []
    for step in trajectory.steps:
        block = json.dumps(
            {"thought": step.thought, "action": step.action, "args": step.args},
            indent=2,
        )
        parts.append(block)
        if step.error:
            parts.append(f"Error: {step.error}")
        elif step.observation:
            parts.append(f"Observation:\n{step.observation}")
    return "\n\n".join(parts)


def build_preference_pairs(
    trajectories: list[Trajectory],
    min_score_margin: float = 2.0,
) -> list[PreferencePair]:
    """Build DPO preference pairs from scored trajectories.

    Groups trajectories by task_id, sorts each group by verifier score,
    and creates chosen/rejected pairs from highest vs lowest scoring.
    Filters out pairs with insufficient score margin.

    Args:
        trajectories: Scored trajectories (must have verifier_result).
        min_score_margin: Minimum score difference for a valid pair.

    Returns:
        List of PreferencePair objects.
    """
    # Group by task_id
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        if traj.verifier_result is not None:
            grouped[traj.task_id].append(traj)

    pairs: list[PreferencePair] = []
    pair_counter = 0

    for task_id, trajs in sorted(grouped.items()):
        if len(trajs) < 2:
            continue

        # Sort by score descending
        trajs.sort(key=lambda t: t.verifier_result.score, reverse=True)

        # Build prompt from task instruction
        prompt = f"Task: {task_id}"

        # Create pairs: best vs each worse trajectory
        best = trajs[0]
        for worse in trajs[1:]:
            margin = best.verifier_result.score - worse.verifier_result.score
            if margin >= min_score_margin:
                pair_counter += 1
                pairs.append(PreferencePair(
                    pair_id=f"pref_{pair_counter:06d}",
                    task_id=task_id,
                    prompt=prompt,
                    chosen=trajectory_to_text(best),
                    rejected=trajectory_to_text(worse),
                    chosen_score=best.verifier_result.score,
                    rejected_score=worse.verifier_result.score,
                    score_margin=round(margin, 2),
                ))
                break  # One pair per task to avoid imbalance

    return pairs


def save_preference_pairs(pairs: list[PreferencePair], path: str | Path) -> None:
    """Save preference pairs to a JSONL file.

    Args:
        pairs: The pairs to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    """Load preference pairs from a JSONL file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of PreferencePair objects.
    """
    path = Path(path)
    if not path.exists():
        return []
    pairs: list[PreferencePair] = []
    for line in path.read_text().splitlines():
        if line.strip():
            pairs.append(PreferencePair.model_validate_json(line))
    return pairs


def pairs_to_hf_dataset(pairs: list[PreferencePair]):
    """Convert preference pairs to a HuggingFace Dataset.

    Requires the 'datasets' package (optional dependency).

    Args:
        pairs: The preference pairs to convert.

    Returns:
        A HuggingFace Dataset with prompt/chosen/rejected columns.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install with: uv pip install -e '.[ml]'"
        )

    return Dataset.from_dict({
        "prompt": [p.prompt for p in pairs],
        "chosen": [p.chosen for p in pairs],
        "rejected": [p.rejected for p in pairs],
    })


# ---------------------------------------------------------------------------
# Legacy helpers (backward compat)
# ---------------------------------------------------------------------------

def generate_preference_pairs(
    trajectories: list[Trajectory],
    tasks_dict: dict,
    min_margin: float = 2.0,
) -> list[PreferencePair]:
    """Legacy API — delegates to build_preference_pairs."""
    return build_preference_pairs(trajectories, min_score_margin=min_margin)


def format_trajectory_actions(traj: Trajectory) -> str:
    """Legacy alias for trajectory_to_text."""
    return trajectory_to_text(traj)


def save_preferences(pairs: list[PreferencePair], path: Path | str) -> None:
    """Legacy alias for save_preference_pairs."""
    save_preference_pairs(pairs, path)


def build_preferences_from_trajectories(
    good_trajectories: list[Trajectory],
    bad_trajectories: list[Trajectory],
) -> list[Preference]:
    """Build simple Preference triples from paired good/bad trajectories."""
    preferences: list[Preference] = []
    for good, bad in zip(good_trajectories, bad_trajectories):
        if good.task_id != bad.task_id:
            continue
        preferences.append(Preference(
            prompt=f"Task: {good.task_id}",
            chosen=trajectory_to_text(good) or good.final_answer or "",
            rejected=trajectory_to_text(bad) or bad.final_answer or "",
        ))
    return preferences
