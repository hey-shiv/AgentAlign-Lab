from pathlib import Path

from agentalign.schemas import Trajectory


def save_trajectory(trajectory: Trajectory, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(trajectory.model_dump_json() + '\n')

def load_trajectories(path: Path | str) -> list[Trajectory]:
    path = Path(path)
    if not path.exists():
        return []
    trajectories = []
    with path.open('r') as f:
        for line in f:
            if line.strip():
                trajectories.append(Trajectory.model_validate_json(line))
    return trajectories
