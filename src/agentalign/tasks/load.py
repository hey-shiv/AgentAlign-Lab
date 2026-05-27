from pathlib import Path
import json

from agentalign.schemas import Task

def load_task(path: Path) ->Task:
    with open(path) as f:
        data = json.load(f)
    return Task(**data)

task = load_task(Path("data/tasks/task.json"))
print(task)