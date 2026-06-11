import shlex
import subprocess
import sys
from pathlib import Path

ALLOWED_BINARIES = ["python", "pytest", "ls", "pwd", "cat", "sed", "head", "tail", "grep", "rg", "wc"]


def normalize_command(parts: list[str]) -> list[str]:
    if parts[0] == "pytest":
        return [sys.executable, "-m", "pytest", *parts[1:]]
    if parts[0] == "python":
        return [sys.executable, *parts[1:]]
    return parts


def run_safe(cmd: str, cwd: Path, timeout: int = 8, forbidden_commands: list[str] | None = None) -> dict:
    parts = shlex.split(cmd)
    if not parts:
        return {"ok": False, "error": "empty_command"}

    forbidden = set(forbidden_commands or [])
    if parts[0] in forbidden:
        return {"ok": False, "error": "forbidden_command"}

    if parts[0] not in ALLOWED_BINARIES:
        return {"ok": False, "error": "forbidden_command"}

    parts = normalize_command(parts)

    try:
        result = subprocess.run(
            parts,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class ToolExecutor:
    def __init__(self, workspace_path: Path, forbidden_commands: list[str] | None = None):
        self.workspace_path = workspace_path
        self.forbidden_commands = forbidden_commands or []

    def execute(self, action: str, args: dict) -> str:
        if action == "run_command":
            cmd = args.get("cmd", "")
            res = run_safe(cmd, self.workspace_path, forbidden_commands=self.forbidden_commands)
            if res.get("error"):
                return f"Error: {res['error']}"
            return f"Exit code: {res.get('returncode')}\nStdout:\n{res.get('stdout')}\nStderr:\n{res.get('stderr')}"
        elif action == "read_file":
            path = self.workspace_path / args.get("path", "")
            try:
                path.resolve().relative_to(self.workspace_path.resolve())
            except ValueError:
                return "Error: Path outside workspace"
            if not path.exists():
                return "Error: File not found"
            return path.read_text()[:4000]
        elif action == "write_file":
            path = self.workspace_path / args.get("path", "")
            try:
                path.resolve().relative_to(self.workspace_path.resolve())
            except ValueError:
                return "Error: Path outside workspace"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.get("content", ""))
            return "File written successfully."
        elif action == "list_files":
            res = run_safe("ls -la", self.workspace_path)
            return res.get("stdout", "Error listing files")
        elif action == "final_answer":
            return "Run ended."
        return f"Error: Unknown action {action}"
