from pathlib import Path
from tempfile import TemporaryDirectory

from agentalign.schemas import Task


def create_workspace(task: Task):
    temp_dir = TemporaryDirectory()

    workspace = Path(temp_dir.name)

    for file in task.files:
        file_path = workspace / file.path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(file.content)

    return workspace, temp_dir