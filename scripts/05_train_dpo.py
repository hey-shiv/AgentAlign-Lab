import argparse
import json
from pathlib import Path

import yaml

REQUIRED_FIELDS = {"prompt", "chosen", "rejected", "chosen_score", "rejected_score", "score_margin"}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_preference_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            if row["score_margin"] <= 0:
                raise ValueError(f"{path}:{line_number} has non-positive score margin")
            rows.append(row)
    return rows


def config_value(config: dict, key: str, default):
    return config.get(key, default)


def training_environment() -> dict:
    try:
        import bitsandbytes
        import datasets
        import peft
        import torch
        import transformers
        import trl
    except ImportError as exc:
        return {
            "ready": False,
            "reason": f"missing dependency: {exc.name}",
        }

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {
        "ready": cuda_available,
        "reason": "cuda available" if cuda_available else "QLoRA path requires CUDA; no CUDA accelerator detected",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
    }


def write_dry_run_artifact(rows: list[dict], val_rows: list[dict], output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    margins = [row["score_margin"] for row in rows]
    metadata = {
        "status": "dry_run_complete",
        "num_pairs": len(rows),
        "num_val_pairs": len(val_rows),
        "avg_margin": sum(margins) / len(margins) if margins else 0.0,
        "min_margin": min(margins) if margins else 0.0,
        "max_margin": max(margins) if margins else 0.0,
        "model": config.get("model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        "training_environment": training_environment(),
        "note": "Dataset validated. Run without --dry-run on a GPU machine to train the LoRA adapter.",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"DPO dry-run complete. Wrote {output_dir / 'training_metadata.json'}")


def write_training_metadata(output_dir: Path, rows: list[dict], val_rows: list[dict], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "training_complete",
        "num_pairs": len(rows),
        "num_val_pairs": len(val_rows),
        "model": config_value(config, "model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        "training_environment": training_environment(),
        "adapter_files": [
            "adapter_config.json",
            "adapter_model.safetensors",
        ],
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dpo_qwen15_lora.yaml")
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--val", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate DPO data without loading a model.")
    parser.add_argument("--preflight", action="store_true", help="Check whether local DPO training can run.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_file = args.train or config_value(config, "train_file", "data/preferences/dpo_train.jsonl")
    val_file = args.val or config_value(config, "eval_file", "data/preferences/dpo_val.jsonl")
    default_output = "outputs/adapters/dpo_run" if args.dry_run else config_value(
        config,
        "output_dir",
        "outputs/adapters/qwen_dpo_final",
    )
    output_dir = Path(args.output_dir or default_output)
    if args.preflight:
        print(json.dumps(training_environment(), indent=2))
        return

    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"{train_file} not found. Please run scripts 01-04 first.")

    rows = load_preference_rows(train_file)
    val_rows = load_preference_rows(val_file) if Path(val_file).exists() else []
    if not rows:
        raise ValueError("No preference pairs found.")
    print(f"Loaded {len(rows)} preference pairs from {train_file}.")
    if val_rows:
        print(f"Loaded {len(val_rows)} validation preference pairs from {val_file}.")

    if args.dry_run:
        write_dry_run_artifact(rows, val_rows, output_dir, config)
        return

    env = training_environment()
    if not env["ready"]:
        raise SystemExit(
            f"DPO training preflight failed: {env['reason']}.\n"
            "Use --dry-run to validate data locally, or run this command on a CUDA GPU machine."
        )

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing ML training dependency. Install the training stack, then rerun:\n"
            "uv pip install torch transformers trl peft datasets bitsandbytes"
        ) from exc

    model_name = config_value(config, "model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    lora_r = config_value(config, "lora_r", config_value(config, "lora_rank", 8))
    lora_alpha = config_value(config, "lora_alpha", 16)
    lora_dropout = config_value(config, "lora_dropout", 0.05)
    learning_rate = config_value(config, "learning_rate", 5.0e-6)
    beta = config_value(config, "beta", 0.1)
    max_length = config_value(config, "max_seq_length", 1024)
    num_train_epochs = config_value(config, "num_epochs", 1)
    per_device_train_batch_size = config_value(config, "per_device_train_batch_size", config_value(config, "batch_size", 1))
    gradient_accumulation_steps = config_value(config, "gradient_accumulation_steps", 8)
    target_modules = config_value(config, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    warmup_ratio = config_value(config, "warmup_ratio", 0.03)
    logging_steps = config_value(config, "logging_steps", 10)
    save_steps = config_value(config, "save_steps", 100)
    eval_steps = config_value(config, "eval_steps", 100)

    print("Loading DPO dataset...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=val_file, split="train") if val_rows else None

    print(f"Initializing tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        beta=beta,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps if eval_dataset is not None else None,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    write_training_metadata(output_dir, rows, val_rows, config)
    print(f"Training complete. Adapter saved to {output_dir}.")


if __name__ == "__main__":
    main()
