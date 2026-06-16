"""Supervised Fine-Tuning (SFT) with QLoRA.

Trains the base model on chosen trajectories from preference pairs
using TRL's SFTTrainer with PEFT/QLoRA.

Requires the [ml] optional dependency group (torch, transformers, trl, peft, etc.).
"""

import json
from pathlib import Path

import yaml


def run_sft(config_path: str = "configs/dpo_qwen15_lora.yaml") -> None:
    """Run SFT training using chosen trajectories as training examples.

    Loads the config YAML, builds a dataset from chosen preference-pair
    texts, and fine-tunes Qwen2.5-Coder-1.5B-Instruct with QLoRA.

    Args:
        config_path: Path to the training configuration YAML.
    """
    config = yaml.safe_load(Path(config_path).read_text())

    model_name = config.get("model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    output_dir = config.get("output_dir", "outputs/adapters/sft")
    train_file = config.get("train_file", "data/preferences/dpo_train.jsonl")
    max_seq_length = config.get("max_seq_length", 2048)
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("per_device_train_batch_size", 1)
    lr = config.get("learning_rate", 5e-4)
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])

    print(f"[SFT] Loading preference pairs from {train_file}")

    # Load chosen texts from preference pairs
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    examples: list[str] = []
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        # Use the chosen trajectory as the SFT example
        examples.append(pair["prompt"] + "\n" + pair["chosen"])

    print(f"[SFT] Loaded {len(examples)} training examples")

    # --- GPU-only imports ---
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        print(f"[SFT] ML dependencies not available: {exc}")
        print("[SFT] Install with: uv pip install -e '.[ml]'")
        print(f"[SFT] Dry-run complete: would train {model_name} on {len(examples)} examples.")
        return

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[SFT] CUDA not available. Dry-run summary:")
        print(f"  Model: {model_name}")
        print(f"  Examples: {len(examples)}")
        print(f"  Output: {output_dir}")
        return

    # Load model in 4-bit with QLoRA
    print(f"[SFT] Loading model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset
    dataset = Dataset.from_dict({"text": examples})

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        report_to="none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    print("[SFT] Starting training...")
    trainer.train()

    # Save adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SFT] Adapter saved to {output_dir}")
