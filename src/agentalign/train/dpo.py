"""Direct Preference Optimization (DPO) training with QLoRA.

Loads preference pairs and trains Qwen2.5-Coder-1.5B-Instruct using
TRL's DPOTrainer with 4-bit quantization and LoRA adapters.

Requires the [ml] optional dependency group.
"""

import json
from pathlib import Path

import yaml


def run_dpo(
    config_path: str = "configs/dpo_qwen15_lora.yaml",
    dry_run: bool = False,
) -> None:
    """Run DPO training.

    Args:
        config_path: Path to the DPO configuration YAML.
        dry_run: If True, validate the dataset without training.
    """
    config = yaml.safe_load(Path(config_path).read_text())

    model_name = config.get("model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    output_dir = config.get("output_dir", "outputs/adapters/dpo")
    train_file = config.get("train_file", "data/preferences/dpo_train.jsonl")
    eval_file = config.get("eval_file", "data/preferences/dpo_val.jsonl")
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("per_device_train_batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    lr = config.get("learning_rate", 5e-4)
    beta = config.get("beta", 0.1)
    max_length = config.get("max_seq_length", 2048)
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])

    # Load and validate preference pairs
    print(f"[DPO] Loading training data from {train_file}")
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    train_pairs: list[dict] = []
    for line in train_path.read_text().splitlines():
        if line.strip():
            train_pairs.append(json.loads(line))

    eval_pairs: list[dict] = []
    eval_path = Path(eval_file)
    if eval_path.exists():
        for line in eval_path.read_text().splitlines():
            if line.strip():
                eval_pairs.append(json.loads(line))

    print(f"[DPO] Train pairs: {len(train_pairs)}, Eval pairs: {len(eval_pairs)}")

    # Validate pair structure
    for i, pair in enumerate(train_pairs[:3]):
        assert "prompt" in pair, f"Pair {i} missing 'prompt'"
        assert "chosen" in pair, f"Pair {i} missing 'chosen'"
        assert "rejected" in pair, f"Pair {i} missing 'rejected'"
    print("[DPO] Dataset structure validated.")

    if dry_run:
        print("[DPO] Dry-run complete. Dataset is valid.")
        print(f"  Model: {model_name}")
        print(f"  Train pairs: {len(train_pairs)}")
        print(f"  Eval pairs: {len(eval_pairs)}")
        print(f"  Output: {output_dir}")
        return

    # --- GPU-only imports ---
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer
    except ImportError as exc:
        print(f"[DPO] ML dependencies not available: {exc}")
        print("[DPO] Install with: uv pip install -e '.[ml]'")
        return

    # Check CUDA
    if not torch.cuda.is_available():
        print("[DPO] CUDA not available. Cannot train.")
        print("[DPO] Use --dry-run for local validation.")
        return

    # Load model in 4-bit
    print(f"[DPO] Loading model: {model_name}")
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

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # Build HF datasets
    train_dataset = Dataset.from_dict({
        "prompt": [p["prompt"] for p in train_pairs],
        "chosen": [p["chosen"] for p in train_pairs],
        "rejected": [p["rejected"] for p in train_pairs],
    })
    eval_dataset = None
    if eval_pairs:
        eval_dataset = Dataset.from_dict({
            "prompt": [p["prompt"] for p in eval_pairs],
            "chosen": [p["chosen"] for p in eval_pairs],
            "rejected": [p["rejected"] for p in eval_pairs],
        })

    # Training config
    training_args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        max_length=max_length,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("[DPO] Starting training...")
    trainer.train()

    # Save adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DPO] Adapter saved to {output_dir}")
