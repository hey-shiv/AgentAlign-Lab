import argparse
import json
from collections import Counter
from pathlib import Path

from agentalign.data.trajectories import load_trajectories
from agentalign.schemas import PreferencePair
from agentalign.tasks.load import load_task

EXPECTED_FAMILIES = {
    "python_bugfix": 25,
    "data_transformation": 15,
    "config_repair": 10,
    "log_extraction": 5,
    "safety_trap": 5,
}
EXPECTED_SPLITS = {"train": 42, "val": 9, "test": 9}
MIN_TRAJECTORIES = 300
MIN_PREFERENCE_PAIRS = 300
ADAPTER_DIR = Path("outputs/adapters/qwen_dpo_final")
REQUIRED_ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
ADAPTER_EVAL_DIR = Path("data/trajectories/scored_test_qwen")
GPU_HANDOFF_DIR = Path("outputs/gpu_handoff")
GPU_HANDOFF_ZIP = Path("outputs/gpu_handoff.zip")


def pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def load_split(path: Path) -> list[str]:
    task_ids = []
    if not path.exists():
        return task_ids
    with path.open() as f:
        for line in f:
            if line.strip():
                task_ids.append(json.loads(line)["task_id"])
    return task_ids


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def load_all_trajectories(scored_dir: Path):
    trajectories = []
    if not scored_dir.exists():
        return trajectories
    for path in sorted(scored_dir.glob("*.jsonl")):
        trajectories.extend(load_trajectories(path))
    return trajectories


def load_preferences(path: Path):
    if not path.exists():
        return []
    pairs = []
    for line in path.read_text().splitlines():
        if line.strip():
            pairs.append(PreferencePair.model_validate_json(line))
    return pairs


def training_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def gpu_handoff_ready(train_pair_count: int) -> tuple[bool, str]:
    handoff_root = GPU_HANDOFF_DIR if (GPU_HANDOFF_DIR / "manifest.json").exists() else Path(".")
    manifest_path = handoff_root / "manifest.json"
    local_archive_required = handoff_root == GPU_HANDOFF_DIR

    if local_archive_required and not GPU_HANDOFF_ZIP.exists():
        return False, f"missing {GPU_HANDOFF_ZIP}"
    if not manifest_path.exists():
        return False, f"missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    required_paths = [
        handoff_root / "pyproject.toml",
        handoff_root / "docs/BEGINNER_TECHNICAL_INTERVIEW_GUIDE.md",
        handoff_root / "configs/dpo_qwen15_lora.yaml",
        handoff_root / "data/preferences/dpo_train.jsonl",
        handoff_root / "data/preferences/dpo_val.jsonl",
        handoff_root / "data/tasks/train.jsonl",
        handoff_root / "data/tasks/val.jsonl",
        handoff_root / "data/tasks/test.jsonl",
        handoff_root / "data/trajectories/scored_train",
        handoff_root / "data/trajectories/scored_val",
        handoff_root / "data/trajectories/scored_test",
        handoff_root / "src/agentalign",
        handoff_root / "scripts/05_train_dpo.py",
        handoff_root / "scripts/08_audit_mvp.py",
        handoff_root / "notebooks/02_colab_dpo_training.ipynb",
        handoff_root / "README_GPU_HANDOFF.md",
        handoff_root / "outputs/adapters/dpo_run/training_metadata.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    task_count = len(list((handoff_root / "data/tasks").glob("*.json")))
    handoff_train_pairs = count_jsonl(handoff_root / "data/preferences/dpo_train.jsonl")
    handoff_train_trajectories = sum(
        count_jsonl(path) for path in (handoff_root / "data/trajectories/scored_train").glob("*.jsonl")
    )
    ok = (
        not missing
        and task_count == sum(EXPECTED_FAMILIES.values())
        and handoff_train_pairs == train_pair_count
        and handoff_train_trajectories >= MIN_TRAJECTORIES
        and manifest.get("data", {}).get("train_preference_pairs") == train_pair_count
    )
    detail = json.dumps({
        "missing": missing,
        "mode": "local_archive" if local_archive_required else "extracted_handoff",
        "task_json_files": task_count,
        "train_trajectories": handoff_train_trajectories,
        "train_preference_pairs": handoff_train_pairs,
        "zip": str(GPU_HANDOFF_ZIP) if local_archive_required else "already extracted",
    }, sort_keys=True)
    return ok, detail


def adapter_eval_ready() -> tuple[bool, str]:
    trajectories = load_all_trajectories(ADAPTER_EVAL_DIR)
    base = [traj for traj in trajectories if traj.agent_id == "qwen_base"]
    tuned = [traj for traj in trajectories if traj.agent_id == "qwen_dpo"]
    base_tasks = {traj.task_id for traj in base}
    tuned_tasks = {traj.task_id for traj in tuned}
    ok = (
        len(base_tasks) == EXPECTED_SPLITS["test"]
        and len(tuned_tasks) == EXPECTED_SPLITS["test"]
        and base_tasks == tuned_tasks
    )
    detail = json.dumps({
        "dir": str(ADAPTER_EVAL_DIR),
        "qwen_base_runs": len(base),
        "qwen_base_tasks": len(base_tasks),
        "qwen_dpo_runs": len(tuned),
        "qwen_dpo_tasks": len(tuned_tasks),
    }, sort_keys=True)
    return ok, detail


def audit() -> tuple[list[tuple[str, bool, str]], bool]:
    checks = []

    tasks = [load_task(path) for path in sorted(Path("data/tasks").glob("*.json"))]
    family_counts = Counter(task.family for task in tasks)
    checks.append((
        "task_suite_count",
        len(tasks) == sum(EXPECTED_FAMILIES.values()),
        f"{len(tasks)} tasks",
    ))
    checks.append((
        "task_family_distribution",
        dict(family_counts) == EXPECTED_FAMILIES,
        json.dumps(dict(family_counts), sort_keys=True),
    ))

    all_split_ids = []
    for split, expected_count in EXPECTED_SPLITS.items():
        task_ids = load_split(Path(f"data/tasks/{split}.jsonl"))
        all_split_ids.extend(task_ids)
        checks.append((f"{split}_split_count", len(task_ids) == expected_count, f"{len(task_ids)} tasks"))
    checks.append((
        "split_no_overlap",
        len(all_split_ids) == len(set(all_split_ids)),
        f"{len(set(all_split_ids))} unique split task ids",
    ))

    scored_train = load_all_trajectories(Path("data/trajectories/scored_train"))
    scored_val = load_all_trajectories(Path("data/trajectories/scored_val"))
    scored_test = load_all_trajectories(Path("data/trajectories/scored_test"))
    total_trajectories = len(scored_train) + len(scored_val) + len(scored_test)
    checks.append((
        "trajectory_volume",
        total_trajectories >= MIN_TRAJECTORIES,
        f"{total_trajectories} scored trajectories",
    ))
    checks.append((
        "train_trajectory_volume",
        len(scored_train) >= MIN_TRAJECTORIES,
        f"{len(scored_train)} train trajectories",
    ))
    checks.append((
        "held_out_test_exists",
        len(scored_test) > 0 and len({traj.task_id for traj in scored_test}) == EXPECTED_SPLITS["test"],
        f"{len(scored_test)} test trajectories across {len({traj.task_id for traj in scored_test})} tasks",
    ))

    train_pairs = load_preferences(Path("data/preferences/dpo_train.jsonl"))
    val_pairs = load_preferences(Path("data/preferences/dpo_val.jsonl"))
    checks.append((
        "preference_pair_volume",
        len(train_pairs) >= MIN_PREFERENCE_PAIRS,
        f"{len(train_pairs)} train preference pairs",
    ))
    checks.append(("validation_pairs_exist", len(val_pairs) > 0, f"{len(val_pairs)} validation pairs"))

    handoff_ok, handoff_detail = gpu_handoff_ready(len(train_pairs))
    checks.append(("colab_gpu_handoff_bundle", handoff_ok, handoff_detail))

    metadata = training_metadata(Path("outputs/adapters/dpo_run/training_metadata.json"))
    checks.append((
        "dpo_dry_run_metadata",
        metadata.get("status") == "dry_run_complete" and metadata.get("num_pairs") == len(train_pairs),
        json.dumps({
            "status": metadata.get("status"),
            "num_pairs": metadata.get("num_pairs"),
            "env_ready": metadata.get("training_environment", {}).get("ready"),
        }, sort_keys=True),
    ))

    try:
        from agentalign.dashboard.app import build_app

        dashboard_ok = type(build_app()).__name__ == "Blocks"
    except Exception:
        dashboard_ok = False
    checks.append(("dashboard_builds", dashboard_ok, "Gradio Blocks app"))

    local_train_ready = metadata.get("training_environment", {}).get("ready") is True
    checks.append((
        "cuda_training_ready",
        local_train_ready,
        metadata.get("training_environment", {}).get("reason", "missing training metadata"),
    ))
    adapter_files = [ADAPTER_DIR / file_name for file_name in REQUIRED_ADAPTER_FILES]
    adapter_ready = all(path.exists() for path in adapter_files)
    checks.append((
        "dpo_adapter_artifacts",
        adapter_ready,
        ", ".join(str(path) for path in adapter_files),
    ))
    adapter_eval_ok, adapter_eval_detail = adapter_eval_ready()
    checks.append(("dpo_held_out_adapter_eval", adapter_eval_ok, adapter_eval_detail))

    required_without_cuda = [
        check
        for check in checks
        if check[0] not in {"cuda_training_ready", "dpo_adapter_artifacts", "dpo_held_out_adapter_eval"}
    ]
    local_loop_ready = all(passed for _, passed, _ in required_without_cuda)
    return checks, local_loop_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable audit output.")
    args = parser.parse_args()

    checks, local_loop_ready = audit()
    if args.json:
        print(json.dumps({
            "local_loop_ready": local_loop_ready,
            "checks": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in checks
            ],
        }, indent=2))
        return

    print("AgentAlign Lab MVP Audit")
    print(f"Local data/eval loop ready: {pass_fail(local_loop_ready)}")
    print()
    for name, passed, detail in checks:
        print(f"{pass_fail(passed):4} {name}: {detail}")

    check_status = dict((name, passed) for name, passed, _ in checks)
    if local_loop_ready and not check_status["cuda_training_ready"]:
        print()
        print("Remaining full-plan gap: real QLoRA/DPO training requires a CUDA GPU environment.")
    if local_loop_ready and not check_status["dpo_adapter_artifacts"]:
        print("Remaining adapter gap: trained LoRA adapter artifacts are not present locally.")
    if local_loop_ready and not check_status["dpo_held_out_adapter_eval"]:
        print("Remaining evaluation gap: base-vs-adapter held-out evaluation is not present locally.")


if __name__ == "__main__":
    main()
