"""Trajectory persistence utilities.

Save and load agent trajectories in JSONL format, and group them by task_id.
"""

import json
from collections import defaultdict
from pathlib import Path

from agentalign.schemas import Trajectory


def save_trajectory(
    trajectory: Trajectory,
    base_dir: str | Path = "data/trajectories/raw",
) -> Path:
    """Append a trajectory to a JSONL file named after its run_id.

    Args:
        trajectory: The trajectory to save.
        base_dir: Directory to write into.

    Returns:
        Path to the written file.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{trajectory.run_id}.jsonl"
    with path.open("a") as f:
        f.write(trajectory.model_dump_json() + "\n")
    return path


def load_trajectory(path: str | Path) -> Trajectory:
    """Load a single trajectory from a JSONL file (first line).

    Args:
        path: Path to the .jsonl file.

    Returns:
        The first Trajectory in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    for line in path.read_text().splitlines():
        if line.strip():
            return Trajectory.model_validate_json(line)
    raise ValueError(f"Empty trajectory file: {path}")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load all trajectories from a JSONL file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of Trajectory objects (empty list if file missing).
    """
    path = Path(path)
    if not path.exists():
        return []
    trajectories: list[Trajectory] = []
    for line in path.read_text().splitlines():
        if line.strip():
            trajectories.append(Trajectory.model_validate_json(line))
    return trajectories


def load_all_trajectories(directory: str | Path) -> list[Trajectory]:
    """Load all trajectories from every .jsonl file in a directory.

    Args:
        directory: Directory to scan for .jsonl files.

    Returns:
        Flat list of all Trajectory objects found.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    all_trajs: list[Trajectory] = []
    for path in sorted(directory.glob("*.jsonl")):
        all_trajs.extend(load_trajectories(path))
    return all_trajs


def group_trajectories_by_task(
    trajectories: list[Trajectory],
) -> dict[str, list[Trajectory]]:
    """Group trajectories by their task_id.

    Args:
        trajectories: Flat list of trajectories.

    Returns:
        Dictionary mapping task_id to list of trajectories.
    """
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        grouped[traj.task_id].append(traj)
    return dict(grouped)
