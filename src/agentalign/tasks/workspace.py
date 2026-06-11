import shutil
import tempfile
from pathlib import Path
from typing import Any

from agentalign.schemas import Task


class TaskWorkspace:
    def __init__(self, task: Task):
        self.task = task
        self.temp_dir = tempfile.mkdtemp(prefix=f"agentalign_task_{task.task_id}_")
        self.path = Path(self.temp_dir)

    def setup(self) -> Path:
        for file in self.task.files:
            file_path = self.path / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)
        return self.path

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> Path:
        return self.setup()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
