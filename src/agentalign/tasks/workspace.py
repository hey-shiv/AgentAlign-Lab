"""Task workspace management.

Creates isolated temporary directories for agent execution and cleans them up.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

from agentalign.schemas import Task


class TaskWorkspace:
    """Context manager that creates an isolated workspace for a task.

    On enter, writes all task files into a fresh temp directory.
    On exit, removes the directory.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.temp_dir = tempfile.mkdtemp(prefix=f"agentalign_{task.task_id}_")
        self.path = Path(self.temp_dir)

    def setup(self) -> Path:
        """Write task files into the workspace and return its path."""
        for file in self.task.files:
            file_path = self.path / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)
        return self.path

    def cleanup(self) -> None:
        """Remove the temporary workspace directory."""
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> Path:
        return self.setup()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
