"""v2 training: rejection-sampling SFT, adapter merge, and step-level DPO.

Prompts are rendered with the tokenizer's chat template before training, exactly as
HFPolicy renders them at inference. TRL then masks the loss to the completion and
appends one end-of-turn token (<|im_end|> for Qwen), which is what the model emits
when it stops generating.

On CUDA the base model is loaded in 4-bit NF4 (QLoRA). If bitsandbytes is unavailable
or 4-bit training fails, training falls back to LoRA on an fp16 base, and the mode used
is recorded in the run metadata.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _render(tokenizer, rows: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict[str, list[str]] = {k: [] for k in keys}
    for r in rows:
        out["prompt"].append(tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        ))
        for k in keys:
            if k != "prompt":
                out[k].append(r[k])
    return out


def _filter_length(tokenizer, rows: list[dict], completion_keys: tuple[str, ...], max_length: int) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for r in rows:
        p = tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        p_len = len(tokenizer(p, add_special_tokens=False).input_ids)
        c_len = max(len(tokenizer(r[k], add_special_tokens=False).input_ids) for k in completion_keys)
        if p_len + c_len + 1 <= max_length:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def _load_model(model_path: str, quantize: bool):
    import torch
    from transformers import AutoModelForCausalLM

    device = _device()
    if device == "cuda" and quantize:
        from transformers import BitsAndBytesConfig

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            dtype=torch.float16,
            device_map={"": 0},
        )
    if device == "cuda":
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map={"": 0})
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)


def _filtered(cls, kwargs: dict):
    params = inspect.signature(cls).parameters
    skipped = sorted(k for k in kwargs if k not in params)
    if skipped:
        print(f"[train] {cls.__name__} does not accept {skipped}; skipping them")
    return cls(**{k: v for k, v in kwargs.items() if k in params and v is not None})


def _common_args(cfg: dict, output_dir: str, has_eval: bool) -> dict:
    cuda = _device() == "cuda"
    return {
        "output_dir": output_dir,
        "num_train_epochs": cfg["epochs"],
        "max_steps": cfg.get("max_steps", -1),
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "learning_rate": cfg["lr"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": cfg.get("warmup_ratio", 0.1),
        "logging_steps": cfg.get("logging_steps", 5),
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": cfg.get("eval_steps", 25) if has_eval else None,
        "save_strategy": "no",
        "max_length": cfg["max_length"],
        "fp16": cuda,
        "bf16": False,
        "gradient_checkpointing": cuda,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if cuda else None,
        "report_to": "none",
        "seed": cfg.get("seed", 42),
        "use_cpu": not cuda,
        "dataloader_num_workers": 0,
    }


def _fp32_trainable(model) -> None:
    """AMP's GradScaler needs fp32 master weights; cast any trainable fp16/bf16 params."""
    import torch

    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    if n:
        print(f"[train] cast {n} trainable tensors to fp32")


def _lora(cfg: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def _run(kind: str, model_path: str, train_rows: list[dict], eval_rows: list[dict], cfg: dict, output_dir: str) -> dict:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    keys = ("prompt", "completion") if kind == "sft" else ("prompt", "chosen", "rejected")
    comp = keys[1:]
    train_rows, dropped_train = _filter_length(tokenizer, train_rows, comp, cfg["max_length"])
    eval_rows, dropped_eval = _filter_length(tokenizer, eval_rows, comp, cfg["max_length"])
    print(f"[{kind}] train={len(train_rows)} eval={len(eval_rows)} dropped_for_length={dropped_train + dropped_eval}")
    if not train_rows:
        raise ValueError(f"[{kind}] no training rows left")
    train_ds = Dataset.from_dict(_render(tokenizer, train_rows, keys))
    eval_ds = Dataset.from_dict(_render(tokenizer, eval_rows, keys)) if eval_rows else None

    modes = ["qlora", "lora_fp16"] if (_device() == "cuda" and cfg.get("quantize", True)) else ["lora_fp16"]
    last_error = None
    for mode in modes:
        try:
            model = _load_model(model_path, quantize=(mode == "qlora"))
            args = _common_args(cfg, output_dir, eval_ds is not None)
            if kind == "sft":
                from trl import SFTConfig, SFTTrainer

                args["completion_only_loss"] = True
                trainer = SFTTrainer(model=model, args=_filtered(SFTConfig, args), train_dataset=train_ds,
                                     eval_dataset=eval_ds, processing_class=tokenizer, peft_config=_lora(cfg))
            else:
                from trl import DPOConfig, DPOTrainer

                args["beta"] = cfg["beta"]
                trainer = DPOTrainer(model=model, args=_filtered(DPOConfig, args), train_dataset=train_ds,
                                     eval_dataset=eval_ds, processing_class=tokenizer, peft_config=_lora(cfg))
            _fp32_trainable(trainer.model)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            trainer.train()
            wall = time.time() - start
            final_eval = trainer.evaluate() if eval_ds is not None else {}
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            meta = {
                "kind": kind,
                "mode": mode,
                "base_model": model_path,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "dropped_for_length": dropped_train + dropped_eval,
                "wall_time_seconds": round(wall, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
                "config": cfg,
                "final_eval": final_eval,
                "log_history": trainer.state.log_history,
            }
            (Path(output_dir) / "v2_training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return meta
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to the next mode, then re-raise
            last_error = exc
            print(f"[{kind}] mode {mode} failed: {type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"[{kind}] all training modes failed") from last_error


def train_sft(model_path: str, train_rows: list[dict], eval_rows: list[dict], cfg: dict, output_dir: str) -> dict:
    return _run("sft", model_path, train_rows, eval_rows, cfg, output_dir)


def train_dpo(model_path: str, train_rows: list[dict], eval_rows: list[dict], cfg: dict, output_dir: str) -> dict:
    return _run("dpo", model_path, train_rows, eval_rows, cfg, output_dir)


def merge_adapter(base_model: str, adapter_dir: str, out_dir: str) -> str:
    """Merge a LoRA adapter into an fp16 copy of the base model and save it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir
