"""Tool implementations for the agent sandbox.

Each tool function takes workspace-relative arguments and returns an
observation string. The dispatch_tool function routes an Action to the
correct handler.
"""

import shlex
import subprocess
import sys
from pathlib import Path

from agentalign.schemas import Action

# Only these binaries may be invoked by the agent.
ALLOWED_BINARIES = [
    "python", "python3", "pytest",
    "ls", "pwd", "cat", "sed",
    "head", "tail", "grep", "wc",
]

_MAX_OUTPUT = 4000  # characters


def _normalize_command(parts: list[str]) -> list[str]:
    """Map python/pytest to the current interpreter."""
    if not parts:
        return parts
    if parts[0] in ("python", "python3"):
        return [sys.executable, *parts[1:]]
    if parts[0] == "pytest":
        return [sys.executable, "-m", "pytest", *parts[1:]]
    return parts


# ---------------------------------------------------------------------------
# Individual tool functions
# ---------------------------------------------------------------------------

def tool_list_files(cwd: str) -> str:
    """List all files in the workspace directory.

    Args:
        cwd: Workspace root path.

    Returns:
        Newline-separated listing of files, or an error message.
    """
    workspace = Path(cwd)
    try:
        files = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file())
        if not files:
            return "(empty workspace)"
        return "\n".join(files)
    except Exception as exc:
        return f"Error listing files: {exc}"


def tool_read_file(path: str, cwd: str, max_chars: int = 3000) -> str:
    """Read a file from the workspace.

    Args:
        path: Relative path within the workspace.
        cwd: Workspace root path.
        max_chars: Maximum characters to return.

    Returns:
        File contents (possibly truncated), or an error message.
    """
    workspace = Path(cwd)
    file_path = workspace / path

    # Security: ensure path stays inside workspace
    try:
        file_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return "Error: Path outside workspace."

    if not file_path.exists():
        return f"Error: File not found: {path}"

    try:
        content = file_path.read_text()
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... (truncated at {max_chars} chars)"
        return content
    except Exception as exc:
        return f"Error reading file: {exc}"


def tool_write_file(path: str, content: str, cwd: str) -> str:
    """Write content to a file in the workspace.

    Args:
        path: Relative path within the workspace.
        content: The content to write.
        cwd: Workspace root path.

    Returns:
        Success confirmation or an error message.
    """
    workspace = Path(cwd)
    file_path = workspace / path

    # Security: ensure path stays inside workspace
    try:
        file_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return "Error: Path outside workspace."

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return "File written successfully."
    except Exception as exc:
        return f"Error writing file: {exc}"


def tool_run_command(
    cmd: str,
    cwd: str,
    forbidden: list[str] | None = None,
    timeout: int = 8,
) -> str:
    """Execute a command in the workspace sandbox.

    Args:
        cmd: The command string.
        cwd: Workspace root path.
        forbidden: Extra forbidden binaries for this task.
        timeout: Maximum execution time in seconds.

    Returns:
        Formatted output string with exit code, stdout, and stderr.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        return f"Error: Malformed command: {exc}"

    if not parts:
        return "Error: Empty command."

    binary = parts[0]

    # Check forbidden commands
    if binary in set(forbidden or []):
        return f"Error: forbidden_command ({binary})"

    # Check allowlist
    if binary not in ALLOWED_BINARIES:
        return f"Error: forbidden_command ({binary})"

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
        stdout = result.stdout[-_MAX_OUTPUT:] if result.stdout else ""
        stderr = result.stderr[-_MAX_OUTPUT:] if result.stderr else ""
        return f"Exit code: {result.returncode}\nStdout:\n{stdout}\nStderr:\n{stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as exc:
        return f"Error: {exc}"


def dispatch_tool(
    action: Action,
    cwd: str,
    forbidden_commands: list[str] | None = None,
) -> str:
    """Route an Action to the appropriate tool function.

    Args:
        action: The parsed Action from the agent.
        cwd: Workspace root path.
        forbidden_commands: Binaries the agent must not invoke.

    Returns:
        Observation string from the executed tool.
    """
    name = action.action
    args = action.args

    if name == "list_files":
        return tool_list_files(cwd)
    elif name == "read_file":
        return tool_read_file(args.get("path", ""), cwd)
    elif name == "write_file":
        return tool_write_file(args.get("path", ""), args.get("content", ""), cwd)
    elif name == "run_command":
        return tool_run_command(args.get("cmd", ""), cwd, forbidden=forbidden_commands)
    elif name == "final_answer":
        return "Agent submitted final answer."
    else:
        return f"Error: Unknown action '{name}'"


# ---------------------------------------------------------------------------
# Legacy ToolExecutor class (backward compat with existing agent loop)
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Stateful tool executor bound to a workspace."""

    def __init__(self, workspace_path: Path, forbidden_commands: list[str] | None = None) -> None:
        self.workspace_path = workspace_path
        self.forbidden_commands = forbidden_commands or []

    def execute(self, action_name: str, args: dict) -> str:
        """Execute an action by name with the given arguments."""
        action = Action(action=action_name, args=args)
        return dispatch_tool(
            action,
            str(self.workspace_path),
            forbidden_commands=self.forbidden_commands,
        )
