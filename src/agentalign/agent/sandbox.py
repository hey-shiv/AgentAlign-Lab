"""Sandboxed command execution and workspace management.

Provides a security layer between the agent and the host OS:
- Allowlisted binaries only
- No shell=True
- Stdout/stderr capped at 4 000 chars
- Configurable timeouts
"""

import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agentalign.schemas import Task

# Only these binaries may be executed by the agent.
ALLOWED_BINARIES = [
    "python", "python3", "pytest",
    "ls", "pwd", "cat", "sed",
    "head", "tail", "grep", "wc",
]

_MAX_OUTPUT = 4000  # characters


def _normalize_command(parts: list[str]) -> list[str]:
    """Replace 'python' and 'pytest' with the current interpreter."""
    if not parts:
        return parts
    if parts[0] in ("python", "python3"):
        return [sys.executable, *parts[1:]]
    if parts[0] == "pytest":
        return [sys.executable, "-m", "pytest", *parts[1:]]
    return parts


def create_workspace(task: Task) -> str:
    """Create a temporary directory and populate it with the task's files.

    Args:
        task: The task whose files should be written.

    Returns:
        Absolute path to the new workspace directory.
    """
    workspace = tempfile.mkdtemp(prefix=f"agentalign_{task.task_id}_")
    for f in task.files:
        path = Path(workspace) / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f.content)
    return workspace


def run_safe(
    cmd: str,
    cwd: str,
    timeout: int = 8,
    forbidden_commands: list[str] | None = None,
) -> dict:
    """Execute *cmd* inside *cwd* after safety checks.

    Args:
        cmd: The shell-style command string (split with shlex).
        cwd: Working directory for the subprocess.
        timeout: Maximum wall-clock seconds.
        forbidden_commands: Extra binaries that are disallowed for this task.

    Returns:
        dict with keys: ok (bool), returncode (int), stdout (str),
        stderr (str), error (str | None).
    """
    parts = shlex.split(cmd)
    if not parts:
        return {"ok": False, "error": "empty_command"}

    binary = parts[0]

    # Check task-specific forbidden commands
    forbidden = set(forbidden_commands or [])
    if binary in forbidden:
        return {"ok": False, "error": "forbidden_command"}

    # Check global allowlist
    if binary not in ALLOWED_BINARIES:
        return {"ok": False, "error": "forbidden_command"}

    parts = _normalize_command(parts)

    try:
        result = subprocess.run(
            parts,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-_MAX_OUTPUT:],
            "stderr": result.stderr[-_MAX_OUTPUT:],
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def cleanup_workspace(path: str) -> None:
    """Remove a workspace directory created by *create_workspace*."""
    shutil.rmtree(path, ignore_errors=True)
