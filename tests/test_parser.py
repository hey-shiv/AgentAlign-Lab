"""Tests for the agent parser module.

Covers: valid JSON action parsing, malformed JSON handling, forbidden command
detection, path escape detection, markdown code fence stripping.
"""

import json

import pytest

from agentalign.agent.parser import OutputParser, parse_action, validate_action
from agentalign.schemas import Action


class TestParseAction:
    """Tests for parse_action function."""

    def test_valid_json_action(self):
        """Parse a well-formed JSON action."""
        raw = json.dumps({
            "thought": "I should list the files",
            "action": "list_files",
            "args": {},
        })
        action, error = parse_action(raw)
        assert error == ""
        assert action is not None
        assert action.action == "list_files"
        assert action.thought == "I should list the files"

    def test_json_with_extra_text(self):
        """Parse JSON embedded in surrounding text."""
        raw = 'Sure, let me do that.\n{"thought": "ok", "action": "read_file", "args": {"path": "x.py"}}\nDone.'
        action, error = parse_action(raw)
        assert error == ""
        assert action is not None
        assert action.action == "read_file"

    def test_markdown_code_fences(self):
        """Strip markdown code fences around JSON."""
        raw = '```json\n{"thought": "fix", "action": "write_file", "args": {"path": "a.py", "content": "x"}}\n```'
        action, error = parse_action(raw)
        assert error == ""
        assert action is not None
        assert action.action == "write_file"

    def test_malformed_json_returns_error(self):
        """Return error for invalid JSON."""
        raw = '{"thought": "broken, "action": "read_file"'
        action, error = parse_action(raw)
        assert action is None
        assert "Invalid JSON" in error or "No JSON" in error

    def test_no_json_returns_error(self):
        """Return error when no JSON object is found."""
        raw = "I think we should fix the bug by editing the file."
        action, error = parse_action(raw)
        assert action is None
        assert "No JSON" in error

    def test_missing_action_field(self):
        """Return error when 'action' field is missing."""
        raw = json.dumps({"thought": "hmm", "args": {}})
        action, error = parse_action(raw)
        assert action is None
        assert "action" in error.lower()

    def test_name_alias_for_action(self):
        """Accept 'name' as alias for 'action'."""
        raw = json.dumps({"thought": "ok", "name": "list_files", "args": {}})
        action, error = parse_action(raw)
        assert error == ""
        assert action is not None
        assert action.action == "list_files"


class TestValidateAction:
    """Tests for validate_action function."""

    def test_valid_action(self):
        action = Action(action="read_file", args={"path": "main.py"})
        ok, reason = validate_action(action, forbidden_commands=["rm", "curl"])
        assert ok is True
        assert reason == ""

    def test_forbidden_command_detected(self):
        action = Action(action="run_command", args={"cmd": "rm -rf /"})
        ok, reason = validate_action(action, forbidden_commands=["rm", "curl"])
        assert ok is False
        assert "Forbidden" in reason or "forbidden" in reason.lower()

    def test_path_escape_detected(self):
        action = Action(action="read_file", args={"path": "../../etc/passwd"})
        ok, reason = validate_action(action, forbidden_commands=[])
        assert ok is False
        assert "escape" in reason.lower()

    def test_unknown_action_rejected(self):
        action = Action(action="delete_everything", args={})
        ok, reason = validate_action(action, forbidden_commands=[])
        assert ok is False
        assert "Unknown" in reason


class TestOutputParser:
    """Tests for backward-compatible OutputParser."""

    def test_parse_action_text(self):
        output = "Search Google for information\nOther output"
        action = OutputParser.parse_action(output)
        assert action == "Search Google for information"

    def test_is_valid_output(self):
        assert OutputParser.is_valid_output("some output") is True
        assert OutputParser.is_valid_output("") is False
        assert OutputParser.is_valid_output("   ") is False

    def test_parse_reasoning(self):
        output = "Let me think about this"
        reasoning = OutputParser.parse_reasoning(output)
        assert "think" in reasoning
