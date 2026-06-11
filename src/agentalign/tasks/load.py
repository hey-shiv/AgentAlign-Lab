import json
from pathlib import Path

from agentalign.schemas import Task


def load_task(path: Path | str) -> Task:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    data = json.loads(path.read_text())
    return Task.model_validate(data)

def save_task(task: Task, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task.model_dump_json(indent=2))
