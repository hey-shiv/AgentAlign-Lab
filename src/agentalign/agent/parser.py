"""LLM output parser for the ReAct agent loop.

Extracts structured Action objects from raw model text that may contain
markdown fences, preamble text, or malformed JSON.
"""

import json
import re
from typing import Optional

from pydantic import ValidationError

from agentalign.schemas import Action


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_action(raw: str) -> tuple[Optional[Action], str]:
    """Parse raw LLM output into an Action.

    Handles:
    - Markdown code fences around JSON
    - Extra text before/after the JSON object
    - Missing fields with sensible defaults

    Args:
        raw: The raw string output from the model.

    Returns:
        (Action, "") on success, or (None, error_message) on failure.
    """
    cleaned = _strip_markdown_fences(raw)

    # Find the first { and last } to extract the JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or start > end:
        return None, "No JSON object found in output."

    json_str = cleaned[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"

    if not isinstance(data, dict):
        return None, "JSON is not an object."

    # Accept 'name' as an alias for 'action'
    if "action" not in data and "name" in data:
        data["action"] = data.pop("name")

    if "action" not in data:
        return None, "JSON missing 'action' field."

    try:
        action = Action.model_validate(data)
    except ValidationError as exc:
        return None, f"Schema validation failed: {exc}"

    return action, ""


def validate_action(
    action: Action,
    forbidden_commands: list[str],
    workspace: str = "",
) -> tuple[bool, str]:
    """Validate that an action is safe to execute.

    Checks:
    1. action.action is a recognised tool name.
    2. run_command does not invoke a forbidden binary.
    3. File paths do not escape the workspace via '../'.

    Args:
        action: The parsed Action to validate.
        forbidden_commands: Binaries the agent must not call.
        workspace: Workspace root path for escape detection.

    Returns:
        (True, "") if valid, or (False, reason) if invalid.
    """
    valid_actions = {"list_files", "read_file", "write_file", "run_command", "final_answer"}
    if action.action not in valid_actions:
        return False, f"Unknown action: {action.action}"

    # Check run_command for forbidden binaries
    if action.action == "run_command":
        import shlex
        cmd = action.args.get("cmd", "")
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return False, "Malformed command string."
        if parts and parts[0] in set(forbidden_commands):
            return False, f"Forbidden command: {parts[0]}"

    # Check file paths for workspace escape
    if action.action in ("read_file", "write_file"):
        path = action.args.get("path", "")
        if ".." in path:
            return False, f"Path escape attempt: {path}"

    return True, ""


# ---------------------------------------------------------------------------
# Backward-compatible OutputParser for older tests
# ---------------------------------------------------------------------------

class OutputParser:
    """Simple text-output parser kept for backward compatibility."""

    @staticmethod
    def parse_action(output: str) -> str:
        return output.strip().splitlines()[0] if output.strip() else ""

    @staticmethod
    def is_valid_output(output: str) -> bool:
        return bool(output.strip())

    @staticmethod
    def parse_reasoning(output: str) -> str:
        return output.strip()
