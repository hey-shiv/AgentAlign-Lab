"""Tests for verifier module."""

from agentalign.schemas import Step, Trajectory
from agentalign.verifier.checks import NonEmptyCheck, ValidStepsCheck, run_all_checks


def test_non_empty_check():
    """Test non-empty trajectory check."""
    check = NonEmptyCheck()

    # Valid trajectory
    traj = Trajectory(trajectory_id="t1", task_id="task1", success=True)
    traj.steps.append(Step(step_id=0, action="test", observation="result"))
    assert check.check(traj) is True

    # Empty trajectory
    empty_traj = Trajectory(trajectory_id="t2", task_id="task2", success=False)
    assert check.check(empty_traj) is False


def test_valid_steps_check():
    """Test valid steps check."""
    check = ValidStepsCheck()

    # Valid steps
    traj = Trajectory(trajectory_id="t1", task_id="task1", success=True)
    traj.steps.append(Step(step_id=0, action="test", observation="result"))
    assert check.check(traj) is True

    # Invalid steps (missing observation)
    bad_traj = Trajectory(trajectory_id="t2", task_id="task2", success=True)
    bad_traj.steps.append(Step(step_id=0, action="test", observation=""))
    assert check.check(bad_traj) is False


def test_run_all_checks():
    """Test running all checks."""
    traj = Trajectory(trajectory_id="t1", task_id="task1", success=True)
    traj.steps.append(Step(step_id=0, action="test", observation="result"))

    results = run_all_checks(traj)
    assert all(results.values())
