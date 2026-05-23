import json
from pathlib import Path

from agentalign.schemas import Task


def load_task(path: str) -> Task:
    task_path = Path(path)

    data = json.loads(task_path.read_text())

    task = Task(**data)

    return task