"""Trajectory scoring utilities.

Implements scoring functions for evaluating trajectory quality.
"""

from agentalign.schemas import Trajectory


class Scorer:
    """Base scorer class."""

    def score(self, trajectory: Trajectory) -> float:
        """Score a trajectory.

        Args:
            trajectory: Trajectory to score

        Returns:
            Score between 0 and 1
        """
        raise NotImplementedError


class SuccessScorer(Scorer):
    """Scorer based on task success."""

    def score(self, trajectory: Trajectory) -> float:
        """Score based on success flag."""
        return 1.0 if trajectory.success else 0.0


class LengthScorer(Scorer):
    """Scorer based on trajectory efficiency."""

    def __init__(self, max_steps: int = 10):
        """Initialize scorer.

        Args:
            max_steps: Maximum expected steps
        """
        self.max_steps = max_steps

    def score(self, trajectory: Trajectory) -> float:
        """Score based on trajectory length."""
        length_ratio = len(trajectory.steps) / self.max_steps
        # Prefer shorter trajectories
        return max(0.0, 1.0 - length_ratio)
