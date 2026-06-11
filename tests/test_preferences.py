"""Tests for preferences module."""

from agentalign.data.preferences import build_preferences_from_trajectories
from agentalign.schemas import Preference, Trajectory


def test_build_preferences():
    """Test building preference pairs."""
    # Create sample trajectories
    good_trajs = [
        Trajectory(
            trajectory_id=f"good_{i}",
            task_id=f"task_{i}",
            success=True,
            score=0.9,
        )
        for i in range(3)
    ]

    bad_trajs = [
        Trajectory(
            trajectory_id=f"bad_{i}",
            task_id=f"task_{i}",
            success=False,
            score=0.1,
        )
        for i in range(3)
    ]

    prefs = build_preferences_from_trajectories(good_trajs, bad_trajs)

    assert len(prefs) == 3
    assert all(isinstance(p, Preference) for p in prefs)


def test_preference_fields():
    """Test preference has required fields."""
    pref = Preference(
        prompt="Test",
        chosen="Good answer",
        rejected="Bad answer",
    )

    assert pref.prompt == "Test"
    assert pref.chosen == "Good answer"
    assert pref.rejected == "Bad answer"
