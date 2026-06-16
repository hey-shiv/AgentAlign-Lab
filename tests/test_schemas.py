"""Tests for schemas module.

Covers: Task validation, Action parsing, VerifierResult score bounds,
PreferencePair score_margin calculation, and backward compatibility.
"""

import pytest
from pydantic import ValidationError

from agentalign.schemas import (
    Action,
    Preference,
    PreferencePair,
    Step,
    Task,
    TaskFile,
    Trajectory,
    VerifierConfig,
    VerifierResult,
)


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------

class TestTask:
    """Tests for Task model validation."""

    def test_task_creation_valid(self):
        """Create a valid task with all required fields."""
        task = Task(
            task_id="test_001",
            instruction="Fix the bug in stats.py",
            family="python_bugfix",
        )
        assert task.task_id == "test_001"
        assert task.instruction == "Fix the bug in stats.py"
        assert task.description == "Fix the bug in stats.py"  # property alias

    def test_task_with_description_alias(self):
        """Accept 'description' as alias for 'instruction'."""
        task = Task(task_id="t1", description="Do something")
        assert task.instruction == "Do something"

    def test_task_empty_id_raises(self):
        """Reject empty task_id."""
        with pytest.raises(ValidationError):
            Task(task_id="", instruction="valid")

    def test_task_empty_instruction_raises(self):
        """Reject empty instruction."""
        with pytest.raises(ValidationError):
            Task(task_id="t1", instruction="")

    def test_task_whitespace_only_raises(self):
        """Reject whitespace-only task_id and instruction."""
        with pytest.raises(ValidationError):
            Task(task_id="   ", instruction="valid")

    def test_task_default_forbidden_commands(self):
        """Default forbidden commands include common dangerous binaries."""
        task = Task(task_id="t1", instruction="fix it")
        assert "rm" in task.forbidden_commands
        assert "curl" in task.forbidden_commands

    def test_task_with_files(self):
        """Task with files list."""
        task = Task(
            task_id="t1",
            instruction="fix it",
            files=[
                TaskFile(path="main.py", content="print('hello')"),
                TaskFile(path="test_main.py", content="def test(): pass"),
            ],
        )
        assert len(task.files) == 2
        assert task.files[0].path == "main.py"


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------

class TestAction:
    """Tests for Action model."""

    def test_action_valid(self):
        action = Action(action="read_file", args={"path": "main.py"})
        assert action.action == "read_file"

    def test_action_with_thought(self):
        action = Action(thought="Let me read", action="read_file", args={"path": "x.py"})
        assert action.thought == "Let me read"

    def test_action_name_compat(self):
        """Support legacy 'name' field as alias for 'action'."""
        action = Action.model_validate({"name": "list_files", "args": {}})
        assert action.action == "list_files"
        assert action.name == "list_files"  # property

    def test_action_default_thought(self):
        """Default empty thought when not provided."""
        action = Action.model_validate({"action": "final_answer", "args": {}})
        assert action.thought == ""


# ---------------------------------------------------------------------------
# VerifierResult tests
# ---------------------------------------------------------------------------

class TestVerifierResult:
    """Tests for VerifierResult."""

    def test_verifier_result_success_alias(self):
        """Accept 'success' as alias for 'passed'."""
        vr = VerifierResult.model_validate({"success": True, "score": 10.0})
        assert vr.passed is True
        assert vr.success is True

    def test_verifier_result_passed_field(self):
        vr = VerifierResult(passed=True, score=8.5)
        assert vr.passed is True
        assert vr.score == 8.5


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    """Tests for Trajectory model."""

    def test_trajectory_creation(self):
        traj = Trajectory(run_id="r1", task_id="t1")
        assert traj.run_id == "r1"
        assert traj.trajectory_id == "r1"  # property

    def test_trajectory_id_alias(self):
        """Accept 'trajectory_id' as alias for 'run_id'."""
        traj = Trajectory(trajectory_id="t123", task_id="t1")
        assert traj.run_id == "t123"

    def test_trajectory_legacy_success_score(self):
        """Convert top-level success/score to verifier_result."""
        traj = Trajectory.model_validate({
            "trajectory_id": "r1",
            "task_id": "t1",
            "success": True,
            "score": 9.5,
        })
        assert traj.verifier_result is not None
        assert traj.verifier_result.passed is True
        assert traj.verifier_result.score == 9.5
        assert traj.success is True
        assert traj.score == 9.5


# ---------------------------------------------------------------------------
# PreferencePair tests
# ---------------------------------------------------------------------------

class TestPreferencePair:
    """Tests for PreferencePair."""

    def test_preference_pair_creation(self):
        pair = PreferencePair(
            pair_id="p1",
            task_id="t1",
            prompt="Fix it",
            chosen="good",
            rejected="bad",
            chosen_score=9.0,
            rejected_score=2.0,
            score_margin=7.0,
        )
        assert pair.score_margin == 7.0
        assert pair.source == "deterministic_verifier"

    def test_preference_simple(self):
        pref = Preference(prompt="Q", chosen="A", rejected="B")
        assert pref.prompt == "Q"
