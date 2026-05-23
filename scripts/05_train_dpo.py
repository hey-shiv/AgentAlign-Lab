#!/usr/bin/env python
"""Train model using Direct Preference Optimization.

This script trains a model on preference pairs using DPO.
"""

import argparse
from pathlib import Path
from agentalign.train.dpo import DPOTrainer, DPOConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train with DPO")
    parser.add_argument("--model", type=str, default="qwen-1.5b")
    parser.add_argument("--train-data", type=Path, default="data/preferences/dpo_train.jsonl")
    parser.add_argument("--eval-data", type=Path, default="data/preferences/dpo_val.jsonl")
    parser.add_argument("--output", type=Path, default="outputs/adapters")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5.0e-4)

    args = parser.parse_args()

    print(f"Training {args.model} with DPO...")
    config = DPOConfig(
        model_name=args.model,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        output_dir=args.output,
    )

    trainer = DPOTrainer(config)
    results = trainer.train(args.train_data, args.eval_data)

    print(f"Training results: {results}")
    print("Done!")


if __name__ == "__main__":
    main()
