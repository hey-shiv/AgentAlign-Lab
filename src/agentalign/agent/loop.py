"""Agent main loop implementation.

Orchestrates the agent's interaction with the environment.
"""

from agentalign.schemas import Task, Trajectory, Step


class Agent:
    """Main agent class."""

    def __init__(self, model_name: str = "base", max_steps: int = 10):
        """Initialize agent.

        Args:
            model_name: Name of the underlying model
            max_steps: Maximum steps per trajectory
        """
        self.model_name = model_name
        self.max_steps = max_steps

    def run(self, task: Task) -> Trajectory:
        """Run agent on a task.

        Args:
            task: Task to solve

        Returns:
            Completed trajectory
        """
        trajectory = Trajectory(
            trajectory_id=f"traj_{task.task_id}",
            task_id=task.task_id,
            steps=[],
            success=False,
            score=0.0,
        )

        for step_id in range(self.max_steps):
            step = Step(
                step_id=step_id,
                action="placeholder_action",
                observation="placeholder_observation",
            )
            trajectory.steps.append(step)

        trajectory.success = len(trajectory.steps) > 0
        trajectory.score = 0.5 if trajectory.success else 0.0

        return trajectory
