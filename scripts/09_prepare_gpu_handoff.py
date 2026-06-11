import argparse
import json
import shutil
from pathlib import Path

HANDOFF_PATHS = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "docs/BEGINNER_TECHNICAL_INTERVIEW_GUIDE.md",
    "configs/dpo_qwen15_lora.yaml",
    "data/preferences",
    "data/tasks",
    "data/trajectories/scored_train",
    "data/trajectories/scored_val",
    "data/trajectories/scored_test",
    "src/agentalign",
    "scripts",
    "tests",
    "notebooks/02_colab_dpo_training.ipynb",
    "outputs/adapters/dpo_run/training_metadata.json",
]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copy2(src, dst)


def build_manifest() -> dict:
    return {
        "project": "AgentAlign Lab",
        "purpose": "CUDA GPU handoff for QLoRA/DPO training and held-out evaluation.",
        "data": {
            "train_preference_pairs": count_jsonl(Path("data/preferences/dpo_train.jsonl")),
            "val_preference_pairs": count_jsonl(Path("data/preferences/dpo_val.jsonl")),
            "train_tasks": count_jsonl(Path("data/tasks/train.jsonl")),
            "val_tasks": count_jsonl(Path("data/tasks/val.jsonl")),
            "test_tasks": count_jsonl(Path("data/tasks/test.jsonl")),
            "scored_train_trajectories": sum(
                count_jsonl(path) for path in Path("data/trajectories/scored_train").glob("*.jsonl")
            ),
            "scored_val_trajectories": sum(
                count_jsonl(path) for path in Path("data/trajectories/scored_val").glob("*.jsonl")
            ),
            "scored_test_trajectories": sum(
                count_jsonl(path) for path in Path("data/trajectories/scored_test").glob("*.jsonl")
            ),
        },
        "commands": [
            "python -m pip install -U pip",
            "python -m pip install -e '.[ml,dashboard]'",
            "python scripts/05_train_dpo.py --preflight",
            "python scripts/05_train_dpo.py --train data/preferences/dpo_train.jsonl --val data/preferences/dpo_val.jsonl --output-dir outputs/adapters/qwen_dpo_final",
            "python scripts/02_run_agent.py --split test --agent qwen_base --model hf --out runs/test_qwen --clear",
            "python scripts/02_run_agent.py --split test --agent qwen_dpo --model hf_adapter --adapter-path outputs/adapters/qwen_dpo_final --out runs/test_qwen --clear",
            "python scripts/03_score_trajectories.py --runs-dir runs/test_qwen --out-dir data/trajectories/scored_test_qwen --clear",
            "python scripts/06_eval_models.py --runs-dir data/trajectories/scored_test_qwen --split test --agent qwen_dpo --compare-agent qwen_base",
            "python scripts/08_audit_mvp.py",
            "python scripts/10_import_colab_artifacts.py /path/to/qwen_dpo_final_artifacts.zip --clear",
        ],
        "expected_training_artifacts": [
            "outputs/adapters/qwen_dpo_final/adapter_config.json",
            "outputs/adapters/qwen_dpo_final/adapter_model.safetensors",
        ],
        "notes": [
            "For Colab, upload outputs/gpu_handoff.zip and open notebooks/02_colab_dpo_training.ipynb.",
            "Run on a CUDA GPU machine. The local CPU/MPS path is intentionally preflight-blocked for QLoRA.",
            "After training, copy outputs/adapters/qwen_dpo_final back into the repository.",
            "Then run held-out adapter evaluation from the same GPU-capable environment.",
        ],
    }


def count_copied_files(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def write_runbook(path: Path, manifest: dict) -> None:
    lines = [
        "# AgentAlign Lab GPU Handoff",
        "",
        "This bundle contains the DPO preference data, config, and scripts needed to run the remaining CUDA-only training step.",
        "",
        "## Data",
        "",
        f"- Train preference pairs: {manifest['data']['train_preference_pairs']}",
        f"- Validation preference pairs: {manifest['data']['val_preference_pairs']}",
        f"- Train tasks: {manifest['data']['train_tasks']}",
        f"- Validation tasks: {manifest['data']['val_tasks']}",
        f"- Test tasks: {manifest['data']['test_tasks']}",
        f"- Scored train trajectories: {manifest['data']['scored_train_trajectories']}",
        f"- Scored validation trajectories: {manifest['data']['scored_val_trajectories']}",
        f"- Scored test trajectories: {manifest['data']['scored_test_trajectories']}",
        "",
        "## Commands",
        "",
    ]
    for command in manifest["commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend([
        "## Expected Adapter Artifacts",
        "",
        *[f"- `{artifact}`" for artifact in manifest["expected_training_artifacts"]],
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in manifest["notes"]],
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="outputs/gpu_handoff")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--no-zip", action="store_true", help="Skip writing outputs/gpu_handoff.zip.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for rel_path in HANDOFF_PATHS:
        src = Path(rel_path)
        if not src.exists():
            continue
        dst = out_dir / rel_path
        copy_path(src, dst)
        copied.append(rel_path)

    manifest = build_manifest()
    manifest["files"] = copied
    write_runbook(out_dir / "README_GPU_HANDOFF.md", manifest)
    copied_file_count = count_copied_files(out_dir)
    manifest["copied_file_count"] = copied_file_count
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_zip:
        archive_base = out_dir.with_suffix("")
        archive_path = shutil.make_archive(str(archive_base), "zip", out_dir)
        print(f"Wrote uploadable archive: {archive_path}")
    print(f"Prepared GPU handoff bundle at {out_dir}")
    print(f"Copied {copied_file_count} files from {len(copied)} top-level paths.")


if __name__ == "__main__":
    main()
