"""Batched Hugging Face generation for the v2 rollout engine."""

from __future__ import annotations

import time


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFPolicy:
    """Generates one action per prompt, batching prompts under a token budget.

    Prompts are wrapped in the tokenizer's chat template exactly as v1's
    hf_model.make_hf_callable does, so v1 and v2 rollouts see the same input.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        token_budget: int = 120_000,
        max_batch: int = 64,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or pick_device()
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.token_budget = token_budget
        self.max_batch = max_batch
        self.calls = 0
        self.generated = 0
        self.seconds = 0.0

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            from agentalign.v2.compat import patch_incompatible_torchao

            patch_incompatible_torchao()

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()

    def render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def _generate_batch(self, texts: list[str]) -> list[str]:
        torch = self.torch
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            self.token_budget = max(self.token_budget // 2, 4_000)
            print(f"[policy] OOM on batch of {len(texts)}; splitting, token budget now {self.token_budget}")
            mid = len(texts) // 2
            return self._generate_batch(texts[:mid]) + self._generate_batch(texts[mid:])
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        self.generated += int((new_tokens != self.tokenizer.pad_token_id).sum())
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def __call__(self, prompts: list[str]) -> list[str]:
        start = time.monotonic()
        texts = [self.render(p) for p in prompts]
        lengths = [len(self.tokenizer(t).input_ids) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lengths[i])
        results: list[str | None] = [None] * len(texts)
        i = 0
        while i < len(order):
            longest = lengths[order[i]] + self.max_new_tokens
            size = max(1, min(self.max_batch, self.token_budget // longest))
            chunk = order[i:i + size]
            for idx, text in zip(chunk, self._generate_batch([texts[j] for j in chunk])):
                results[idx] = text
            i += size
        self.calls += 1
        self.seconds += time.monotonic() - start
        return [r if r is not None else "" for r in results]
