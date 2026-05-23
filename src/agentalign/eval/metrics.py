"""Evaluation metrics.

Implements various evaluation metrics.
"""


class Metric:
    """Base metric class."""

    name: str = "base_metric"

    def compute(self, predictions, targets) -> float:
        """Compute metric.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Metric value
        """
        raise NotImplementedError


class AccuracyMetric(Metric):
    """Accuracy metric."""

    name = "accuracy"

    def compute(self, predictions, targets) -> float:
        """Compute accuracy."""
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(predictions)


class SuccessRateMetric(Metric):
    """Success rate metric."""

    name = "success_rate"

    def compute(self, predictions, targets) -> float:
        """Compute success rate."""
        if not predictions:
            return 0.0
        return sum(1 for p in predictions if p) / len(predictions)


METRICS = {
    "accuracy": AccuracyMetric(),
    "success_rate": SuccessRateMetric(),
}
