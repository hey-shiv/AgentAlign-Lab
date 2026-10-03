#!/usr/bin/env python3
"""Script 12: Colab training script for DPO.

This script runs the DPO training pipeline, prioritizing the LLM-generated
preference pairs if available, with a fallback to the dummy baseline pairs.
It's designed to be executed directly in a Colab environment.
"""

import argparse
from pathlib import Path
import yaml
from agentalign.train.dpo import run_dpo

def main() -> None:
    parser = argparse.ArgumentParser(description="Colab DPO Training")
    parser.add_argument("--config", default="configs/dpo_qwen15_lora.yaml", help="Config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    # Check for LLM generated preference files first
    train_file_llm = config.get("train_file_llm", "")
    eval_file_llm = config.get("eval_file_llm", "")

    if train_file_llm and Path(train_file_llm).exists():
        print(f"[*] Found LLM-generated training pairs: {train_file_llm}")
        config["train_file"] = train_file_llm
        if eval_file_llm and Path(eval_file_llm).exists():
            config["eval_file"] = eval_file_llm
        else:
            config.pop("eval_file", None)
            
        # Write modified config to a temporary file
        temp_config_path = Path("configs/dpo_qwen15_lora_temp.yaml")
        temp_config_path.write_text(yaml.dump(config))
        config_path = temp_config_path
    else:
        print("[*] LLM pairs not found. Falling back to default scripted baseline pairs.")

    print("\nStarting DPO Training Pipeline...")
    run_dpo(config_path=str(config_path), dry_run=False)
    
    if config_path.name == "dpo_qwen15_lora_temp.yaml":
        config_path.unlink()

if __name__ == "__main__":
    main()
