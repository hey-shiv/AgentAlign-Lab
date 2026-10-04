"""Environment compatibility shims for cloud images.

Kaggle's image ships torchao 0.10.0. peft 0.19 raises ImportError whenever it injects a
LoRA layer into an unquantized model and finds a torchao older than 0.16, even though
this project never uses torchao. Treat an incompatible torchao as absent.
"""

from __future__ import annotations


def patch_incompatible_torchao() -> bool:
    """Make peft ignore a torchao it cannot use. Returns True if a patch was applied."""
    import importlib.metadata as metadata
    import importlib.util

    if importlib.util.find_spec("torchao") is None:
        return False
    try:
        from packaging.version import Version

        if Version(metadata.version("torchao")) >= Version("0.16.0"):
            return False
    except Exception:  # noqa: BLE001 - unknown version: be safe and disable it for peft
        pass

    import peft.import_utils as import_utils

    def _unavailable() -> bool:
        return False

    import_utils.is_torchao_available = _unavailable
    try:
        import peft.tuners.lora.torchao as lora_torchao

        lora_torchao.is_torchao_available = _unavailable
    except ImportError:
        pass
    print("[compat] incompatible torchao found; peft will ignore it")
    return True
