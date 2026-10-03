#!/usr/bin/env python3
"""Script 14: Merge LoRA Adapter and Export Final Model.

Loads the base model and the fine-tuned LoRA adapter, merges the weights
permanently, and saves the standalone model to disk so it can be uploaded
to HuggingFace or converted to GGUF format for Ollama.
"""

import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA and export model")
    parser.add_argument("--config", default="configs/dpo_qwen15_lora.yaml", help="Path to training config")
    parser.add_argument("--out-dir", default="outputs/final_merged_model", help="Where to save the final model")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_model_name = config["model_name"]
    adapter_path = config["output_dir"]
    out_dir = Path(args.out_dir)

    if not Path(adapter_path).exists():
        print(f"❌ Error: Adapter path {adapter_path} not found.")
        print("Did you finish DPO training (Step 2) yet?")
        return

    print(f"🚀 Loading base model: {base_model_name}")
    # We load in fp16 to ensure the merged weights are high quality but fit in RAM
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU to avoid OOM
        low_cpu_mem_usage=True
    )

    print(f"🔌 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔄 Merging LoRA weights permanently into base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving standalone final model to: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("✅ Successfully exported final merged model!")
    print(f"You can now upload {out_dir} to HuggingFace or run it with Ollama/llama.cpp.")


if __name__ == "__main__":
    main()
