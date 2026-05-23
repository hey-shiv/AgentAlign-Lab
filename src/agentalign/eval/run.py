"""Evaluation runner.

Orchestrates model evaluation on tasks.
"""

from pathlib import Path


class Evaluator:
    """Evaluates models on tasks."""

    def __init__(self, output_dir: Path | str = "outputs/evals"):
        """Initialize evaluator.

        Args:
            output_dir: Directory for evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, tasks, num_samples: int = 100) -> dict:
        """Evaluate model on tasks.

        Args:
            model: Model to evaluate
            tasks: List of tasks
            num_samples: Number of evaluation samples

        Returns:
            Evaluation results
        """
        results = {
            "model": str(model),
            "num_tasks": len(tasks),
            "num_samples": num_samples,
            "metrics": {},
        }

        return results
