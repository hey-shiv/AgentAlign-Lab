"""Preference data handling.

Utilities for creating and managing preference pairs for DPO training.
"""

import json
from pathlib import Path
from agentalign.schemas import Preference, Trajectory


def build_preferences_from_trajectories(
    good_trajectories: list[Trajectory],
    bad_trajectories: list[Trajectory],
) -> list[Preference]:
    """Build preference pairs from trajectories.

    Args:
        good_trajectories: High-quality trajectories
        bad_trajectories: Low-quality trajectories

    Returns:
        List of preference pairs
    """
    preferences = []

    for good, bad in zip(good_trajectories, bad_trajectories):
        # Create preference from trajectory pair
        pref = Preference(
            prompt=f"Task: {good.task_id}",
            chosen=f"Trajectory {good.trajectory_id}: Success",
            rejected=f"Trajectory {bad.trajectory_id}: Failed",
            metadata={"good_id": good.trajectory_id, "bad_id": bad.trajectory_id},
        )
        preferences.append(pref)

    return preferences


def save_preferences(preferences: list[Preference], output_path: Path | str) -> None:
    """Save preferences to JSONL file.

    Args:
        preferences: List of preferences to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pref in preferences:
            f.write(pref.model_dump_json() + "\n")


def load_preferences(file_path: Path | str) -> list[Preference]:
    """Load preferences from JSONL file.

    Args:
        file_path: Path to preferences file

    Returns:
        List of loaded preferences
    """
    file_path = Path(file_path)
    preferences = []

    if not file_path.exists():
        return preferences

    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pref = Preference(**data)
                preferences.append(pref)

    return preferences
