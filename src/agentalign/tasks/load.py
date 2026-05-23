"""Task loading and I/O utilities.

Functions for loading tasks from various data sources.
"""

import json
from pathlib import Path
from agentalign.schemas import Task


def load_tasks_from_jsonl(file_path: Path | str) -> list[Task]:
    """Load tasks from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of loaded tasks
    """
    tasks = []
    file_path = Path(file_path)

    if not file_path.exists():
        return tasks

    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task = Task(**data)
                tasks.append(task)

    return tasks


def save_tasks_to_jsonl(tasks: list[Task], file_path: Path | str) -> None:
    """Save tasks to a JSONL file.

    Args:
        tasks: List of tasks to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
