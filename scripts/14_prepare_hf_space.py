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
    shutil.copy("src/agentalign/dashboard/app.py", space_dir / "app.py")

    # Create requirements.txt
    (space_dir / "requirements.txt").write_text(
        "gradio>=4.0.0\n"
        "pandas>=2.0.0\n"
        "pydantic>=2.0.0\n"
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

    print(f"✅ Hugging Face Space bundle prepared at: {space_dir}")
    print("To deploy:")
    print("1. Go to huggingface.co/spaces and create a new Gradio Space")
    print("2. Upload the contents of outputs/hf_space to the Space")

if __name__ == "__main__":
    prepare_space()
