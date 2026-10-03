"""HuggingFace model loader for real LLM inference."""

import torch
from typing import Callable

def make_hf_callable(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device: str = "mps",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    adapter_path: str | None = None,
) -> Callable[[str], str]:
    """Create a model callable for the ReAct loop using a real HF model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf_model] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[hf_model] Requested device: {device}")
    
    # Auto-detect if requested device is not available
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[hf_model] MPS not available, falling back to {device}")

    # Auto device map handles cuda nicely, but mps requires explicit mapping
    device_map = "auto" if device != "mps" else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    if device in ["mps", "cpu"]:
        model.to(device)

    if adapter_path:
        from peft import PeftModel
        print(f"[hf_model] Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    def hf_callable(prompt: str) -> str:
        # Wrap the raw prompt in the model's chat template
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        
        do_sample = temperature > 0.0
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    return hf_callable
