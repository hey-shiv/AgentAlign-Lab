"""Verification checks for trajectories.

Implements various checks to validate trajectory correctness.
"""

from agentalign.schemas import Trajectory


class Check:
    """Base verification check."""

    def check(self, trajectory: Trajectory) -> bool:
        """Perform check.

        Args:
            trajectory: Trajectory to check

        Returns:
            True if check passes
        """
        raise NotImplementedError


class NonEmptyCheck(Check):
    """Verify trajectory is not empty."""

    def check(self, trajectory: Trajectory) -> bool:
        """Check that trajectory has steps."""
        return len(trajectory.steps) > 0


class ValidStepsCheck(Check):
    """Verify all steps are valid."""

    def check(self, trajectory: Trajectory) -> bool:
        """Check that all steps have required fields."""
        for step in trajectory.steps:
            if not step.action or not step.observation:
                return False
        return True


def run_all_checks(trajectory: Trajectory) -> dict[str, bool]:
    """Run all verification checks.

    Args:
        trajectory: Trajectory to verify

    Returns:
        Dictionary of check results
    """
    checks = {
        "non_empty": NonEmptyCheck(),
        "valid_steps": ValidStepsCheck(),
    }

    results = {}
    for name, check in checks.items():
        results[name] = check.check(trajectory)

    return results
