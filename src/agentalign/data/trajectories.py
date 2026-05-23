"""Trajectory data handling.

Utilities for loading, saving, and manipulating trajectory data.
"""

import json
from pathlib import Path
from agentalign.schemas import Trajectory


def load_trajectories(directory: Path | str) -> list[Trajectory]:
    """Load trajectories from directory.

    Args:
        directory: Directory containing trajectory files

    Returns:
        List of loaded trajectories
    """
    directory = Path(directory)
    trajectories = []

    if not directory.exists():
        return trajectories

    for file in directory.glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    trajectory = Trajectory(**data)
                    trajectories.append(trajectory)

    return trajectories


def save_trajectories(
    trajectories: list[Trajectory], output_path: Path | str
) -> None:
    """Save trajectories to file.

    Args:
        trajectories: List of trajectories to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")
