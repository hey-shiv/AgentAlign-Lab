#!/usr/bin/env python3
"""Prepare the Hugging Face Space directory for deployment."""

import shutil
from pathlib import Path

def prepare_space():
    space_dir = Path("outputs/hf_space")
    if space_dir.exists():
        shutil.rmtree(space_dir)
    space_dir.mkdir(parents=True)

    # Copy the dashboard code
    code = Path("src/agentalign/dashboard/app.py").read_text()
    # ZeroGPU Spaces fail at startup unless a @spaces.GPU function exists.
    # The dashboard needs no GPU, so register a no-op before launch() blocks.
    stub = (
        "try:\n"
        "    import spaces\n\n"
        "    @spaces.GPU\n"
        "    def _zerogpu_noop():\n"
        "        return None\n"
        "except ImportError:\n"
        "    pass\n\n\n"
    )
    marker = 'if __name__ == "__main__":'
    assert marker in code
    (space_dir / "app.py").write_text(code.replace(marker, stub + marker, 1))

    # Create requirements.txt
    (space_dir / "requirements.txt").write_text(
        "gradio==5.49.1\n"
        "huggingface_hub<1.0\n"
        "pandas>=2.0.0\n"
        "pydantic>=2.0.0\n"
    )

    # Space config: sdk_version must match the pinned gradio, otherwise the
    # Space installs an old gradio that breaks against newer huggingface_hub
    # (ImportError: cannot import name 'HfFolder').
    (space_dir / "README.md").write_text(
        "---\n"
        "title: AgentAlign Dashboard\n"
        "colorFrom: blue\n"
        "colorTo: indigo\n"
        "sdk: gradio\n"
        "sdk_version: 5.49.1\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "---\n"
    )

    # We also need to copy the minimal schemas and data loading code
    # to avoid needing the full package installed, or we can just copy
    # the agentalign package into the space.
    agentalign_dst = space_dir / "agentalign"
    shutil.copytree("src/agentalign", agentalign_dst)
    
    # Copy data directories needed for replaying trajectories
    for data_dir in ["data/tasks", "data/trajectories", "data/preferences"]:
        src = Path(data_dir)
        if src.exists():
            shutil.copytree(src, space_dir / data_dir)
            
    # Copy evaluation outputs
    if Path("outputs/evals").exists():
        shutil.copytree("outputs/evals", space_dir / "outputs" / "evals")

    print(f"Hugging Face Space bundle prepared at: {space_dir}")
    print("To deploy:")
    print("1. Go to huggingface.co/spaces and create a new Gradio Space")
    print("2. Upload the contents of outputs/hf_space to the Space")

if __name__ == "__main__":
    prepare_space()
