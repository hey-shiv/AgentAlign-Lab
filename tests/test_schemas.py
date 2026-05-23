"""Tests for schemas module."""

import pytest
from agentalign.schemas import Task, Trajectory, Preference, Step


def test_task_creation():
    """Test creating a task."""
    task = Task(task_id="test_001", description="Test task")
    assert task.task_id == "test_001"
    assert task.description == "Test task"


def test_task_validation():
    """Test task validation."""
    with pytest.raises(ValueError):
        Task(task_id="", description="")  # Should fail on empty


def test_trajectory_creation():
    """Test creating a trajectory."""
    traj = Trajectory(
        trajectory_id="traj_001",
        task_id="task_001",
        success=True,
        score=0.95,
    )
    assert traj.trajectory_id == "traj_001"
    assert traj.success is True


def test_preference_creation():
    """Test creating a preference."""
    pref = Preference(
        prompt="What is 2+2?",
        chosen="4",
        rejected="5",
    )
    assert pref.prompt == "What is 2+2?"
    assert pref.chosen == "4"
