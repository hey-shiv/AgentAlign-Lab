import argparse
import json
from pathlib import Path

from agentalign.data.preferences import generate_preference_pairs, save_preferences
from agentalign.data.trajectories import load_trajectories
from agentalign.tasks.load import load_task


def load_split_task_ids(split: str | None) -> set[str] | None:
    if not split:
        return None
    path = Path(f"data/tasks/{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    task_ids = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                task_ids.add(json.loads(line)["task_id"])
    return task_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="data/trajectories/scored")
    parser.add_argument("--out", type=str, default="data/preferences/dpo_train.jsonl")
    parser.add_argument("--min-margin", type=float, default=2.0)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    split_task_ids = load_split_task_ids(args.split)
    all_trajectories = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        for trajectory in load_trajectories(path):
            if split_task_ids is None or trajectory.task_id in split_task_ids:
                all_trajectories.append(trajectory)

    if not all_trajectories:
        print("No trajectories found.")
        return

    # Load tasks associated with trajectories
    tasks_dict = {}
    for traj in all_trajectories:
        if traj.task_id not in tasks_dict:
            task_path = Path(f"data/tasks/{traj.task_id}.json")
            if task_path.exists():
                tasks_dict[traj.task_id] = load_task(task_path)

    pairs = generate_preference_pairs(all_trajectories, tasks_dict, min_margin=args.min_margin)

    if pairs:
        save_preferences(pairs, args.out)
        print(f"Saved {len(pairs)} preference pairs to {args.out}.")
    else:
        print("Could not generate any preference pairs with the given trajectories and margin.")

if __name__ == "__main__":
    main()
