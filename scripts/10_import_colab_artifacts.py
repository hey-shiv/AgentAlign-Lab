import argparse
import json
import shutil
import zipfile
from pathlib import Path

REQUIRED_ARTIFACTS = [
    "outputs/adapters/qwen_dpo_final/adapter_config.json",
    "outputs/adapters/qwen_dpo_final/adapter_model.safetensors",
]
REQUIRED_DIRS = [
    "data/trajectories/scored_test_qwen",
    "runs/test_qwen",
]


def safe_member_path(member: str) -> Path:
    path = Path(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {member}")
    return path


def archive_members(archive_path: Path) -> set[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return {member for member in archive.namelist() if not member.endswith("/")}


def validate_archive(archive_path: Path) -> list[str]:
    members = archive_members(archive_path)
    missing = [path for path in REQUIRED_ARTIFACTS if path not in members]
    for required_dir in REQUIRED_DIRS:
        if not any(member.startswith(f"{required_dir}/") for member in members):
            missing.append(f"{required_dir}/")
    return missing


def extract_archive(archive_path: Path, repo_root: Path, clear: bool) -> None:
    if clear:
        for rel_path in [*REQUIRED_DIRS, "outputs/adapters/qwen_dpo_final"]:
            target = repo_root / rel_path
            if target.exists():
                shutil.rmtree(target)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            rel_path = safe_member_path(member.filename)
            if not (
                str(rel_path).startswith("outputs/adapters/qwen_dpo_final/")
                or str(rel_path).startswith("data/trajectories/scored_test_qwen/")
                or str(rel_path).startswith("runs/test_qwen/")
            ):
                continue
            target = repo_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def count_jsonl_files(path: Path) -> tuple[int, int]:
    files = sorted(path.glob("*.jsonl")) if path.exists() else []
    lines = 0
    for file_path in files:
        lines += sum(1 for line in file_path.read_text().splitlines() if line.strip())
    return len(files), lines


def import_summary(repo_root: Path) -> dict:
    scored_files, scored_lines = count_jsonl_files(repo_root / "data/trajectories/scored_test_qwen")
    raw_files, raw_lines = count_jsonl_files(repo_root / "runs/test_qwen")
    return {
        "adapter_config": (repo_root / REQUIRED_ARTIFACTS[0]).exists(),
        "adapter_model": (repo_root / REQUIRED_ARTIFACTS[1]).exists(),
        "scored_test_qwen_files": scored_files,
        "scored_test_qwen_trajectories": scored_lines,
        "raw_test_qwen_files": raw_files,
        "raw_test_qwen_trajectories": raw_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=str, help="Path to qwen_dpo_final_artifacts.zip downloaded from Colab.")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root to import artifacts into.")
    parser.add_argument("--clear", action="store_true", help="Remove existing adapter/eval artifacts before import.")
    parser.add_argument("--dry-run", action="store_true", help="Validate archive without extracting it.")
    args = parser.parse_args()

    archive_path = Path(args.archive).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")

    missing = validate_archive(archive_path)
    if missing:
        raise SystemExit(
            "Archive is missing required Colab artifacts:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    if args.dry_run:
        print("Archive validation passed. No files extracted.")
        return

    extract_archive(archive_path, repo_root, clear=args.clear)
    summary = import_summary(repo_root)
    print(json.dumps(summary, indent=2))
    if not summary["adapter_config"] or not summary["adapter_model"]:
        raise SystemExit("Adapter import failed: required adapter files are missing after extraction.")
    if summary["scored_test_qwen_trajectories"] == 0:
        raise SystemExit("Evaluation import failed: no scored qwen test trajectories found after extraction.")
    print("Colab artifacts imported. Run: python scripts/08_audit_mvp.py")


if __name__ == "__main__":
    main()
