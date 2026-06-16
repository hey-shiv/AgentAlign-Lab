"""Tests for the preferences module.

Covers: preference pair building from trajectories with different scores,
min_score_margin filtering, trajectory_to_text formatting.
"""

import pytest

from agentalign.data.preferences import (
    build_preference_pairs,
    build_preferences_from_trajectories,
    trajectory_to_text,
)
from agentalign.schemas import (
    Preference,
    PreferencePair,
    Step,
    Trajectory,
    VerifierResult,
)


def _make_trajectory(
    run_id: str,
    task_id: str,
    passed: bool,
    score: float,
    agent_id: str = "test",
) -> Trajectory:
    """Helper to create a scored trajectory."""
    traj = Trajectory(
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        verifier_result=VerifierResult(passed=passed, score=score),
    )
    traj.steps.append(Step(
        step_index=1,
        thought="Test thought",
        action="final_answer",
        args={"answer": "done"},
        observation="Agent submitted final answer.",
    ))
    return traj


class TestBuildPreferencePairs:
    """Tests for build_preference_pairs."""

    def test_builds_pair_from_high_low_scores(self):
        """Creates a pair from best vs worst trajectory on same task."""
        trajs = [
            _make_trajectory("r1", "t1", True, 9.0),
            _make_trajectory("r2", "t1", False, 2.0),
            _make_trajectory("r3", "t1", False, -1.0),
        ]
        pairs = build_preference_pairs(trajs, min_score_margin=2.0)
        assert len(pairs) >= 1
        pair = pairs[0]
        assert pair.task_id == "t1"
        assert pair.chosen_score > pair.rejected_score
        assert pair.score_margin >= 2.0

    def test_min_margin_filtering(self):
        """Pairs with insufficient margin are filtered out."""
        trajs = [
            _make_trajectory("r1", "t1", True, 5.0),
            _make_trajectory("r2", "t1", True, 4.5),  # margin = 0.5
        ]
        pairs = build_preference_pairs(trajs, min_score_margin=2.0)
        assert len(pairs) == 0

    def test_single_trajectory_no_pair(self):
        """Cannot build a pair from a single trajectory."""
        trajs = [_make_trajectory("r1", "t1", True, 9.0)]
        pairs = build_preference_pairs(trajs, min_score_margin=0.0)
        assert len(pairs) == 0

    def test_multiple_tasks(self):
        """Builds pairs across multiple tasks."""
        trajs = [
            _make_trajectory("r1", "t1", True, 9.0),
            _make_trajectory("r2", "t1", False, 1.0),
            _make_trajectory("r3", "t2", True, 8.0),
            _make_trajectory("r4", "t2", False, 0.0),
        ]
        pairs = build_preference_pairs(trajs, min_score_margin=2.0)
        assert len(pairs) == 2
        task_ids = {p.task_id for p in pairs}
        assert task_ids == {"t1", "t2"}

    def test_pair_id_format(self):
        """Pair IDs follow the expected format."""
        trajs = [
            _make_trajectory("r1", "t1", True, 9.0),
            _make_trajectory("r2", "t1", False, 1.0),
        ]
        pairs = build_preference_pairs(trajs, min_score_margin=2.0)
        assert pairs[0].pair_id.startswith("pref_")


class TestTrajectoryToText:
    """Tests for trajectory_to_text formatting."""

    def test_formats_steps(self):
        """Trajectory steps are formatted as JSON blocks."""
        traj = _make_trajectory("r1", "t1", True, 9.0)
        text = trajectory_to_text(traj)
        assert "final_answer" in text
        assert "Test thought" in text

    def test_empty_trajectory(self):
        """Empty trajectory produces empty string."""
        traj = Trajectory(run_id="r1", task_id="t1")
        text = trajectory_to_text(traj)
        assert text == ""


class TestLegacyPreferences:
    """Tests for backward-compatible preference building."""

    def test_build_preferences_from_trajectories(self):
        good = [
            Trajectory.model_validate({
                "trajectory_id": f"good_{i}",
                "task_id": f"task_{i}",
                "success": True,
                "score": 0.9,
            })
            for i in range(3)
        ]
        bad = [
            Trajectory.model_validate({
                "trajectory_id": f"bad_{i}",
                "task_id": f"task_{i}",
                "success": False,
                "score": 0.1,
            })
            for i in range(3)
        ]
        prefs = build_preferences_from_trajectories(good, bad)
        assert len(prefs) == 3
        assert all(isinstance(p, Preference) for p in prefs)

    def test_preference_fields(self):
        pref = Preference(prompt="Test", chosen="Good answer", rejected="Bad answer")
        assert pref.prompt == "Test"
        assert pref.chosen == "Good answer"
        assert pref.rejected == "Bad answer"
