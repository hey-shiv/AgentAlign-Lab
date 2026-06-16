"""Task serialization and deserialization utilities.

Provides functions to load tasks from JSON/JSONL files and save them back.
"""

import json
from pathlib import Path

from agentalign.schemas import Task


def load_task_from_dict(data: dict) -> Task:
    """Deserialize a dictionary into a Task.

    Args:
        data: Dictionary with task fields.

    Returns:
        Validated Task instance.
    """
    return Task.model_validate(data)


def load_task(path: Path | str) -> Task:
    """Load a single task from a JSON file.

    Args:
        path: Path to a .json file containing one task.

    Returns:
        Validated Task instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    data = json.loads(path.read_text())
    return Task.model_validate(data)


def load_tasks_from_jsonl(path: str | Path) -> list[Task]:
    """Read a JSONL file where each line is a JSON task object.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of validated Task instances.
    """
    path = Path(path)
    if not path.exists():
        return []
    tasks: list[Task] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            tasks.append(Task.model_validate(json.loads(line)))
    return tasks


def save_tasks_to_jsonl(tasks: list[Task], path: str | Path) -> None:
    """Serialize a list of Tasks to a JSONL file.

    Args:
        tasks: Tasks to save.
        path: Destination .jsonl file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")


def save_task(task: Task, path: Path | str) -> None:
    """Save a single task to a JSON file.

    Args:
        task: The task to save.
        path: Destination .json file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task.model_dump_json(indent=2))


def load_template_tasks() -> list[Task]:
    """Load the 5 built-in template tasks.

    Returns:
        The five hand-written template tasks, one per family.
    """
    from agentalign.tasks.templates import load_template_tasks as _load
    return _load()
