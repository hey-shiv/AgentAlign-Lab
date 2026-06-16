#!/usr/bin/env python3
"""Script 05: Train DPO adapter.

Loads the DPO config, validates the preference dataset, and either
runs a dry-run (local) or full training (GPU).

Usage:
    python scripts/05_train_dpo.py --dry-run
    python scripts/05_train_dpo.py --preflight
    python scripts/05_train_dpo.py                # Full training (GPU required)
"""

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO adapter")
    parser.add_argument(
        "--config", default="configs/dpo_qwen15_lora.yaml",
        help="Path to DPO config YAML",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset without training")
    parser.add_argument("--preflight", action="store_true", help="Check environment readiness")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    if args.preflight:
        print("=" * 50)
        print("DPO Training Preflight Check")
        print("=" * 50)

        # Check training data
        train_file = config.get("train_file", "data/preferences/dpo_train.jsonl")
        train_path = Path(train_file)
        if train_path.exists():
            n_train = sum(1 for line in train_path.read_text().splitlines() if line.strip())
            print(f"✓ Training data: {n_train} pairs ({train_file})")
        else:
            print(f"✗ Training data not found: {train_file}")

        # Check eval data
        eval_file = config.get("eval_file", "data/preferences/dpo_val.jsonl")
        eval_path = Path(eval_file)
        if eval_path.exists():
            n_eval = sum(1 for line in eval_path.read_text().splitlines() if line.strip())
            print(f"✓ Eval data: {n_eval} pairs ({eval_file})")
        else:
            print(f"  Eval data not found: {eval_file} (optional)")

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("✗ CUDA not available (training requires GPU)")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    print("  MPS (Apple Silicon) detected — not sufficient for QLoRA")
        except ImportError:
            print("✗ PyTorch not installed")

        # Check ML dependencies
        for pkg in ["transformers", "trl", "peft", "datasets", "bitsandbytes"]:
            try:
                __import__(pkg)
                print(f"✓ {pkg} available")
            except ImportError:
                print(f"✗ {pkg} not installed")

        print(f"\nModel: {config.get('model_name')}")
        print(f"Output: {config.get('output_dir')}")
        return

    if args.dry_run:
        from agentalign.train.dpo import run_dpo
        run_dpo(config_path=args.config, dry_run=True)
        return

    # Full training
    from agentalign.train.dpo import run_dpo
    run_dpo(config_path=args.config, dry_run=False)


if __name__ == "__main__":
    main()
