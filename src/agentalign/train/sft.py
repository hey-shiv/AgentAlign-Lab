"""Supervised Fine-Tuning.

Implements SFT training utilities.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTConfig:
    """SFT training configuration."""

    model_name: str = "qwen-1.5b"
    learning_rate: float = 1.0e-4
    num_epochs: int = 3
    batch_size: int = 32
    max_seq_length: int = 2048
    output_dir: Path | str = "outputs/adapters"


class SFTTrainer:
    """Trainer for SFT."""

    def __init__(self, config: SFTConfig):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

    def train(self, train_data, eval_data=None):
        """Train model with SFT.

        Args:
            train_data: Training dataset
            eval_data: Optional evaluation dataset

        Returns:
            Training results
        """
        # Placeholder implementation
        return {
            "status": "started",
            "config": self.config,
            "output_dir": self.config.output_dir,
        }
