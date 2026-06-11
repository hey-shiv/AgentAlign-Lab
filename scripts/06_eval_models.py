import argparse
import json
from pathlib import Path

from agentalign.data.trajectories import load_trajectories


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


def load_agent_trajectories(agent: str, split: str | None, runs_dir: Path):
    split_task_ids = load_split_task_ids(split)
    trajectories = []
    for path in sorted(runs_dir.glob(f"*_{agent}*.jsonl")):
        for trajectory in load_trajectories(path):
            if split_task_ids is None or trajectory.task_id in split_task_ids:
                trajectories.append(trajectory)
    return trajectories


def metrics(trajectories):
    total = len(trajectories)
    if not total:
        return {
            "runs": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "invalid_action_rate": 0.0,
            "unsafe_action_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_steps": 0.0,
        }
    verifier_results = [traj.verifier_result for traj in trajectories if traj.verifier_result]
    passed = sum(1 for result in verifier_results if result.passed)
    invalid = sum(result.invalid_actions for result in verifier_results)
    unsafe = sum(result.unsafe_actions for result in verifier_results)
    timeouts = sum(1 for result in verifier_results if "timeout" in result.failure_tags)
    steps = sum(len(traj.steps) for traj in trajectories)
    return {
        "runs": total,
        "pass_rate": passed / total * 100,
        "avg_score": sum(result.score for result in verifier_results) / total,
        "invalid_action_rate": invalid / total * 100,
        "unsafe_action_rate": unsafe / total * 100,
        "timeout_rate": timeouts / total * 100,
        "avg_steps": steps / total,
    }


def print_metrics(agent: str, values: dict) -> None:
    print(f"--- Evaluation Results: {agent} ---")
    print(f"Total Runs: {values['runs']}")
    print(f"Pass Rate: {values['pass_rate']:.1f}%")
    print(f"Avg Score: {values['avg_score']:.2f}")
    print(f"Invalid Action Rate: {values['invalid_action_rate']:.1f}%")
    print(f"Unsafe Action Rate: {values['unsafe_action_rate']:.1f}%")
    print(f"Timeout Rate: {values['timeout_rate']:.1f}%")
    print(f"Avg Steps: {values['avg_steps']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--agent", type=str, default="baseline")
    parser.add_argument("--compare-agent", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="data/trajectories/scored")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs found in {runs_dir}")
        return

    print(f"Evaluating agent '{args.agent}' on split '{args.split or 'all'}'...")
    primary = metrics(load_agent_trajectories(args.agent, args.split, runs_dir))
    print_metrics(args.agent, primary)

    if args.compare_agent:
        print()
        print(f"Evaluating comparison agent '{args.compare_agent}' on split '{args.split or 'all'}'...")
        comparison = metrics(load_agent_trajectories(args.compare_agent, args.split, runs_dir))
        print_metrics(args.compare_agent, comparison)
        print()
        print("--- Delta ---")
        print(f"Pass Rate Delta: {primary['pass_rate'] - comparison['pass_rate']:+.1f} pp")
        print(f"Avg Score Delta: {primary['avg_score'] - comparison['avg_score']:+.2f}")
        print(
            "Invalid Action Rate Delta: "
            f"{primary['invalid_action_rate'] - comparison['invalid_action_rate']:+.1f} pp"
        )
        print(f"Unsafe Action Rate Delta: {primary['unsafe_action_rate'] - comparison['unsafe_action_rate']:+.1f} pp")


if __name__ == "__main__":
    main()
